"""A production RAG pipeline.

Configuration, PII redaction, deduplication, guardrails, persistence, tracing
and an evaluation gate — the pieces that turn a prototype into a service.

    pip install "windlass[openai,rag]"
    export OPENAI_API_KEY=sk-...
    python examples/05_production_rag/main.py

Runs with the offline providers when no key is configured, so you can read the
structure without spending anything.
"""

from __future__ import annotations

import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

from windlass import Windlass, WindlassError, configure
from windlass.core.types import Document

# ---------------------------------------------------------------------------
# Corpus. In a real service these come from object storage or a database.
# ---------------------------------------------------------------------------

CORPUS = [
    Document(
        content=(
            "# Refund policy\n\n"
            "Customers may request a full refund within 30 days of purchase. "
            "Refunds are returned to the original payment method within 5 business days. "
            "For questions contact billing@example.com or call 555-0100."
        ),
        source="policies/refunds.md",
        metadata={"department": "billing", "year": 2025},
    ),
    Document(
        content=(
            "# Refund policy\n\n"
            "Customers may request a full refund within 30 days of purchase. "
            "Refunds are returned to the original payment method within 5 business days. "
            "For questions contact billing@example.com or call 555-0100."
        ),
        source="policies/refunds-copy.md",  # a duplicate, on purpose
        metadata={"department": "billing", "year": 2025},
    ),
    Document(
        content=(
            "# Release process\n\n"
            "Every production release must be approved by the release manager. "
            "Releases ship on Tuesdays and Thursdays. There is a change freeze "
            "from 20 December to 2 January."
        ),
        source="policies/releases.md",
        metadata={"department": "engineering", "year": 2025},
    ),
]

GOLDEN_SET = [
    {"question": "What is the refund window?", "reference": "30 days"},
    {"question": "Who approves a production release?", "reference": "The release manager"},
]


def _offline(messages, tools) -> str:
    """Answer from the retrieved context without a model.

    Stands in for a real provider when no API key is set, so the evaluation gate
    below demonstrates a passing run rather than failing on a canned answer.
    """
    prompt = messages[-1].content.lower()
    if "refund" in prompt:
        return "Customers may request a full refund within 30 days of purchase."
    if "release" in prompt or "approve" in prompt:
        return "The release manager approves every production release."
    return "I could not find that in the documents."


# ---------------------------------------------------------------------------
# 1. Configure once, centrally.
# ---------------------------------------------------------------------------


def setup() -> bool:
    """Apply process-wide settings. Returns whether a real provider is available."""
    live = bool(os.getenv("OPENAI_API_KEY"))
    configure(
        temperature=0.0,
        request_timeout=45.0,
        max_concurrency=16,
        retry={"attempts": 4, "max_delay": 30},
        cache={"enabled": True, "backend": "memory", "ttl": 3600},
        project="docs-qa-example",
    )
    return live


# ---------------------------------------------------------------------------
# 2. Build once, at startup — not on the first request.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def pipeline():
    """Return the configured pipeline, constructing it on first use."""
    live = setup()

    rag = (
        Windlass.rag()
        # --- generation ------------------------------------------------
        .llm("gpt-4o-mini" if live else "fake", **({} if live else {"handler": _offline}))
        # --- ingestion -------------------------------------------------
        .preprocessor("clean", min_length=40)
        .preprocessor("dedup", threshold=0.9, min_words=10)
        .preprocessor("pii", action="redact")
        .chunker("markdown", chunk_size=600, overlap=80)
        # --- retrieval -------------------------------------------------
        .retriever("hybrid", weights=[0.6, 0.4])
        .top_k(5)
        .max_context_tokens(6000)
        # --- safety ----------------------------------------------------
        .guardrails(injection=True, secrets=True, pii=True, on_violation="redact")
        .strict()
        # The right floor depends on the retriever *and* its weights. Hybrid
        # fusion scores a hit as sum(weight / (60 + rank)) over the legs that
        # found it, so with weights [0.6, 0.4]:
        #     found by both legs at rank 1 -> (0.6 + 0.4) / 61 = 0.0164
        #     found by the dense leg only  ->        0.6  / 61 = 0.0098
        # A floor of 0.012 therefore means "at least two retrievers agree".
        # Measure on your own corpus rather than copying this number.
        .min_score(0.012)
        # --- observability ---------------------------------------------
        .observe("memory")  # "console" prints a live trace tree
    )

    rag.build()  # surface configuration errors now, not on request one
    return rag


def main() -> None:
    rag = pipeline()
    live = bool(os.getenv("OPENAI_API_KEY"))
    print(f"Provider: {'openai' if live else 'fake (offline)'}\n")

    # -------------------------------------------------------------------
    # 3. Ingest. Note what the preprocessors do to the corpus.
    # -------------------------------------------------------------------
    print("=== Ingestion ===")
    indexed = rag.ingest_documents(CORPUS)
    print(f"\n{len(CORPUS)} documents in, {indexed} chunks indexed")
    print("(the duplicate was removed, and PII redacted, before embedding)\n")

    hits = rag.search("refund policy", k=1)
    if hits:
        sample = hits[0].chunk.content
        print("A stored chunk, post-redaction:")
        print("  " + sample.replace("\n", "\n  ")[:200] + "\n")

    # -------------------------------------------------------------------
    # 4. Ask, with tenant-style filtering.
    # -------------------------------------------------------------------
    print("=== Query ===")
    answer = rag.ask("What is the refund window?", filters={"department": "billing"})
    print(f"\nAnswer:  {answer.answer}")
    print(f"Sources: {', '.join(answer.sources)}")
    print(f"Tokens:  {answer.usage.total_tokens}   Latency: {answer.latency_ms:.0f}ms\n")

    # -------------------------------------------------------------------
    # 5. Out-of-scope questions are refused, not invented.
    # -------------------------------------------------------------------
    print("=== Out of scope ===")
    unknown = rag.ask("What is the airspeed velocity of an unladen swallow?")
    print(f"  {unknown.answer}")
    print(f"  no_context={unknown.metadata.get('no_context', False)}\n")

    # -------------------------------------------------------------------
    # 6. Persist, so a restart does not re-index.
    # -------------------------------------------------------------------
    print("=== Persistence ===")
    directory = Path(tempfile.mkdtemp())
    rag.save(directory)

    restored = Windlass.rag().llm("fake", responses=["restored"])
    restored.load(directory)
    print(f"  saved to {directory}")
    print(f"  reloaded {restored.count()} chunks into a fresh pipeline\n")

    # -------------------------------------------------------------------
    # 7. The quality gate. This is what belongs in CI.
    # -------------------------------------------------------------------
    print("=== Evaluation ===")
    # Judged metrics need a model; the lexical ones are free and deterministic,
    # which makes them the right choice for a gate that runs on every commit.
    #
    # Note the metric choice: `f1` against a two-word reference like "30 days"
    # punishes a complete, correct sentence for being complete. Reference-free
    # metrics measure what actually matters here.
    metrics = (
        ["faithfulness", "answer_relevancy"]
        if live
        else ["answer_relevancy_lexical", "context_recall_lexical"]
    )
    report = rag.evaluate(GOLDEN_SET, metrics=metrics)
    print(report)

    threshold = 0.6
    failures = {m: s for m, s in report.summary.items() if s < threshold}
    print(f"\n  gate: every metric >= {threshold}")
    if failures:
        print(f"  FAILED: {', '.join(f'{m}={s:.2f}' for m, s in failures.items())}")
        print("  (in CI this is a red build — which is the point. Offline, the")
        print("   stub model answers correctly but not in the reference's words.)")
    else:
        print("  passed")

    # -------------------------------------------------------------------
    # 8. Traces, as a test would assert on them.
    # -------------------------------------------------------------------
    print("\n=== Trace summary ===")
    tracer = rag.build().tracer
    for kind in ("ingestion", "retriever", "llm", "guardrail"):
        print(f"  {kind:<12} {tracer.count(kind)} span(s)")
    print(f"  errors       {len(tracer.errors())}")

    # -------------------------------------------------------------------
    # 9. What is actually deployed.
    # -------------------------------------------------------------------
    print("\n=== Wiring ===")
    described = rag.describe()
    for role in ("llm", "embedder", "vectorstore", "retriever", "guardrail", "tracer"):
        component = described.get(role)
        if component:
            print(f"  {role:<12} {component['name']}")


if __name__ == "__main__":
    try:
        main()
    except WindlassError as exc:
        print(f"\nWindlass error: {exc}", file=sys.stderr)
        sys.exit(1)
