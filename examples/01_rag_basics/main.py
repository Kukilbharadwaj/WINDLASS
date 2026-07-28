"""RAG basics — ingest, retrieve, generate.

Runs with no API key and no optional dependencies, using the built-in
dependency-free providers.

    python examples/01_rag_basics/main.py
"""

from __future__ import annotations

from windlass import Windlass

CORPUS = {
    "billing.md": (
        "# Billing\n\n"
        "## Refunds\n\n"
        "Customers may request a full refund within 30 days of purchase. "
        "Refunds are processed to the original payment method within 5 business days.\n\n"
        "## Annual plans\n\n"
        "Annual plans are refundable pro rata after the first 30 days. "
        "Contact billing support with the invoice number to start the process."
    ),
    "releases.md": (
        "# Release process\n\n"
        "## Approval\n\n"
        "Every production release must be approved by the release manager. "
        "Emergency hotfixes may be approved by any two senior engineers.\n\n"
        "## Windows\n\n"
        "Releases ship on Tuesdays and Thursdays. There is a change freeze "
        "from 20 December to 2 January."
    ),
    "support.md": (
        "# Support\n\n"
        "## Response times\n\n"
        "Enterprise customers receive a first response within 2 hours during "
        "business hours. Error E1042 indicates an expired API token and is "
        "resolved by rotating the key in the dashboard."
    ),
}


def main() -> None:
    # ---------------------------------------------------------------
    # 1. Build a pipeline. Every argument here has a working default —
    #    Windlass.rag() alone would also run.
    # ---------------------------------------------------------------
    rag = (
        Windlass.rag()
        .llm("fake", responses=["Customers may request a full refund within 30 days."])
        .chunker("markdown", chunk_size=400)
        .retriever("hybrid")
        .top_k(3)
        .strict()  # refuse rather than invent when nothing is found
        .min_score(0.02)  # …and "found" has to mean "actually relevant"
    )

    # ---------------------------------------------------------------
    # 2. Ingest. Loading, cleaning, chunking, embedding and indexing
    #    all happen here.
    # ---------------------------------------------------------------
    total = 0
    for name, text in CORPUS.items():
        total += rag.ingest_text(text, source=name)

    print(f"Indexed {total} chunks from {len(CORPUS)} documents\n")

    # ---------------------------------------------------------------
    # 3. Retrieve without generating. This is the first thing to look
    #    at when an answer is wrong: if the right chunk is not here,
    #    no prompt change will help.
    # ---------------------------------------------------------------
    print("--- Retrieval only ---")
    for hit in rag.search("What is the refund window?", k=3):
        source = hit.chunk.metadata.get("source", "?")
        print(f"  {hit.rank}. [{hit.score:.3f}] {source:<12} {hit.chunk.content[:60]}…")

    # ---------------------------------------------------------------
    # 4. Ask. Guardrails, retrieval, prompt assembly, generation.
    # ---------------------------------------------------------------
    print("\n--- Full answer ---")
    answer = rag.ask("What is the refund window?")

    print(f"Answer:   {answer.answer}")
    print(f"Sources:  {', '.join(answer.sources)}")
    print(f"Contexts: {len(answer.contexts)}")
    print(f"Tokens:   {answer.usage.total_tokens}")
    print(f"Latency:  {answer.latency_ms:.0f}ms")

    # ---------------------------------------------------------------
    # 5. Metadata filtering — the basis of multi-tenancy.
    # ---------------------------------------------------------------
    print("\n--- Filtered retrieval ---")
    hits = rag.search("approval", k=5, filters={"source": "releases.md"})
    for hit in hits:
        print(f"  {hit.chunk.metadata['source']:<12} {hit.chunk.content[:60]}…")

    # ---------------------------------------------------------------
    # 6. Strict mode plus a score floor: refuse rather than invent.
    #
    #    Note that strict() alone would not fire here. Dense retrieval
    #    returns its nearest neighbours however far away they are, so
    #    the result set is almost never empty — min_score() is what
    #    makes "found nothing relevant" mean what you expect.
    # ---------------------------------------------------------------
    print("\n--- Nothing relevant indexed ---")
    answer = rag.ask("What is the airspeed velocity of an unladen swallow?")
    print(f"  {answer.answer}")
    print(f"  no_context={answer.metadata.get('no_context', False)}")

    # ---------------------------------------------------------------
    # 7. Ingestion is idempotent — chunk ids are content hashes.
    # ---------------------------------------------------------------
    before = rag.count()
    for name, text in CORPUS.items():
        rag.ingest_text(text, source=name)
    print(f"\nRe-ingested the same corpus: {before} chunks -> {rag.count()} chunks")


if __name__ == "__main__":
    main()
