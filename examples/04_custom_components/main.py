"""Custom components — writing your own chunker, retriever, guardrail and LLM.

Every one registers the same way the built-ins do, and composes the same way.
Runs with no API key and no optional dependencies.

    python examples/04_custom_components/main.py
"""

from __future__ import annotations

import re
from typing import Any

from windlass import (
    LLM,
    Chunk,
    Chunker,
    Completion,
    Guardrail,
    GuardrailResult,
    Retriever,
    ScoredChunk,
    Windlass,
    register,
)

# ---------------------------------------------------------------------------
# 1. A chunker. One method: split_text.
#    Document iteration, metadata propagation, offset tracking, id assignment
#    and concurrency are all handled by the base class.
# ---------------------------------------------------------------------------


@register.chunker("by-sentence", description="One chunk per group of sentences.")
class SentenceChunker(Chunker):
    """Groups sentences into fixed-size chunks.

    Args:
        sentences_per_chunk: How many sentences to put in each chunk.
        **config: Forwarded to Chunker.
    """

    def __init__(self, *, sentences_per_chunk: int = 2, **config: Any) -> None:
        config.setdefault("min_chunk_size", 0)
        super().__init__(**config)
        self.sentences_per_chunk = sentences_per_chunk

    def split_text(self, text: str) -> list[str]:
        """Split into groups of whole sentences."""
        from windlass.core.text import split_sentences

        sentences = split_sentences(text)
        step = self.sentences_per_chunk
        return [" ".join(sentences[i : i + step]) for i in range(0, len(sentences), step)]


# ---------------------------------------------------------------------------
# 2. A retriever backed by something that is not a vector database.
#    This is the most common real plugin: you already have search.
# ---------------------------------------------------------------------------


@register.retriever("keyword-index", description="A simple in-memory keyword index.")
class KeywordIndexRetriever(Retriever):
    """Retrieves by counting exact keyword occurrences.

    Stands in for "we already have Elasticsearch / Solr / a search API".

    Args:
        **config: Forwarded to Retriever.
    """

    #: Setting this makes traces and hybrid provenance say "keyword-index"
    #: instead of the interface's generic default.
    provider_name = "keyword-index"

    def __init__(self, **config: Any) -> None:
        super().__init__(**config)
        self.corpus: list[Chunk] = []

    async def aindex(self, chunks) -> int:
        """Store chunks in this retriever's own index."""
        self.corpus.extend(chunks)
        return len(chunks)

    async def aretrieve_chunks(self, query, k, *, filters=None, **kwargs):
        """Score by how many query words appear in each chunk."""
        words = {w for w in re.findall(r"\w+", query.lower()) if len(w) > 2}
        scored = []
        for chunk in self.corpus:
            body = chunk.content.lower()
            hits = sum(1 for w in words if w in body)
            if hits:
                scored.append(ScoredChunk(chunk=chunk, score=hits / max(1, len(words))))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]


# ---------------------------------------------------------------------------
# 3. A guardrail. Report what you found; leave the policy to on_violation,
#    so one detector works in both blocking and redacting configurations.
# ---------------------------------------------------------------------------


@register.guardrail("no-competitors", description="Blocks named competitors.")
class CompetitorGuardrail(Guardrail):
    """Catches mentions of named competitors.

    Args:
        names: Competitor names to catch.
        **config: Forwarded to Guardrail.
    """

    def __init__(self, *, names: list[str] | None = None, **config: Any) -> None:
        super().__init__(**config)
        self.names = [n.lower() for n in (names or [])]

    async def acheck(self, content, *, stage="input", context=None) -> GuardrailResult:
        """Find competitor names and mask them."""
        found = [n for n in self.names if n in content.lower()]
        redacted = content
        for name in found:
            redacted = re.sub(re.escape(name), "[COMPETITOR]", redacted, flags=re.IGNORECASE)
        return GuardrailResult(
            allowed=not found,
            content=redacted,
            detections=[{"rule": "competitor", "name": n} for n in found],
            rule="competitor" if found else None,
            stage=stage,
        )


# ---------------------------------------------------------------------------
# 4. An LLM provider. One method: agenerate.
# ---------------------------------------------------------------------------


@register.llm("shouty", description="Answers in capitals. For demonstration only.")
class ShoutyLLM(LLM):
    """A model that repeats the question in capitals.

    Args:
        **config: Forwarded to LLM.
    """

    provider_name = "shouty"
    supports_tools = False

    @classmethod
    def default_model(cls) -> str:
        return "shouty-1"

    async def agenerate(self, messages, *, tools=None, **kwargs) -> Completion:
        """Return the last user message, shouted."""
        last = next(
            (m.content for m in reversed(messages) if m.role.value == "user"),
            "",
        )
        # The RAG prompt ends with "Question: <q>" then "Answer:", so pull out
        # just the question rather than shouting the scaffolding back.
        question = last.split("Question:")[-1].split("Answer:")[0].strip()
        return Completion(content=question.upper(), model=self.model)


# ---------------------------------------------------------------------------


def main() -> None:
    # -------------------------------------------------------------------
    print("--- Registered alongside the built-ins ---")
    print("  chunkers:  ", Windlass.list("chunker"))
    print("  retrievers:", Windlass.list("retriever"))
    print("  llms:      ", Windlass.list("llm"))

    # -------------------------------------------------------------------
    print("\n--- The custom chunker, used by name ---")
    chunker = Windlass.chunker("by-sentence", sentences_per_chunk=2)
    text = (
        "Windlass is a framework. It unifies the AI ecosystem. "
        "Every component is replaceable. Nothing is tightly coupled."
    )
    for i, chunk in enumerate(chunker.split_text(text)):
        print(f"  {i}: {chunk}")

    # -------------------------------------------------------------------
    print("\n--- The custom retriever, in a pipeline ---")
    rag = Windlass.rag().llm("shouty").chunker("by-sentence").retriever("keyword-index").top_k(2)
    rag.ingest_text(text, source="about.txt")

    for hit in rag.search("component replaceable"):
        print(f"  [{hit.score:.2f}] {hit.chunk.content}")

    print(f"\n  answer: {rag.ask('What is replaceable?').answer}")

    # -------------------------------------------------------------------
    print("\n--- The custom guardrail ---")
    guard = Windlass.guardrail("no-competitors", names=["Acme", "Globex"], on_violation="redact")
    print(" ", guard.validate("We are faster than Acme and cheaper than Globex."))

    # -------------------------------------------------------------------
    print("\n--- Composed with a built-in guardrail ---")
    policy = Windlass.guardrail("rules", on_violation="redact") & Windlass.guardrail(
        "no-competitors", names=["Acme"], on_violation="redact"
    )
    print(" ", policy.validate("Mail ada@example.com about the Acme deal."))

    # -------------------------------------------------------------------
    print("\n--- The custom retriever inside hybrid fusion ---")
    from windlass.providers.embeddings.hash import HashEmbedder
    from windlass.providers.vectordb.memory import InMemoryVectorStore

    embedder = HashEmbedder(dimensions=128)
    store = InMemoryVectorStore(dimensions=128)

    dense = Windlass.retriever("vector", embedder=embedder, vectorstore=store, top_k=3)
    keyword = KeywordIndexRetriever(top_k=3)

    hybrid = Windlass.retriever("hybrid", retrievers=[dense, keyword], top_k=2)
    hybrid.index(Windlass.chunker("by-sentence").chunk_text(text))

    for hit in hybrid.retrieve("replaceable component").hits:
        print(f"  [{hit.score:.4f}] {hit.retriever:<22} {hit.chunk.content[:44]}…")

    # -------------------------------------------------------------------
    print("\n--- Full specs, as the CLI sees them ---")
    for spec in Windlass.catalog("chunker"):
        marker = "*" if spec.origin == "user" else " "
        print(f"  {marker} {spec.name:<14} {spec.description}")
    print("\n  (* = registered by this example)")


if __name__ == "__main__":
    main()
