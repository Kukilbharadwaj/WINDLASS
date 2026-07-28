"""Hybrid retrieval — why dense-only search misses things, and what fusion fixes.

Runs with no API key and no optional dependencies.

    python examples/03_hybrid_retrieval/main.py
"""

from __future__ import annotations

from windlass import Windlass
from windlass.core.types import Chunk
from windlass.providers.embeddings.hash import HashEmbedder
from windlass.providers.retrievers.bm25 import BM25Retriever
from windlass.providers.retrievers.hybrid import HybridRetriever
from windlass.providers.retrievers.vector import VectorRetriever
from windlass.providers.vectordb.memory import InMemoryVectorStore

KNOWLEDGE_BASE = [
    "Error E1042 indicates an expired API token. Rotate the key in the dashboard.",
    "If your credentials stop working, generate a fresh token from the settings page.",
    "Authentication failures are usually caused by a stale or revoked access key.",
    "The billing dashboard shows invoices, payment methods and usage history.",
    "Rate limiting returns HTTP 429. Back off exponentially and retry.",
    "Webhooks are delivered at least once; make your handler idempotent.",
]

QUERIES = [
    ("E1042", "an exact error code — no paraphrase to find"),
    ("my login keeps failing", "a paraphrase — no shared vocabulary"),
    ("expired token", "both: the exact term and a concept"),
]


def build():
    """Build three retrievers over one shared corpus."""
    embedder = HashEmbedder(dimensions=256)
    store = InMemoryVectorStore(dimensions=embedder.dimension())

    chunks = [Chunk(content=text, document_id=f"kb-{i}") for i, text in enumerate(KNOWLEDGE_BASE)]
    for chunk, vector in zip(chunks, embedder.embed([c.content for c in chunks]), strict=True):
        chunk.embedding = vector
    store.add(chunks)

    dense = VectorRetriever(embedder=embedder, vectorstore=store, top_k=3)

    lexical = BM25Retriever(top_k=3)
    lexical.index(chunks)

    hybrid = HybridRetriever(retrievers=[dense, lexical], top_k=3)

    return dense, lexical, hybrid


def main() -> None:
    dense, lexical, hybrid = build()

    for query, why in QUERIES:
        print(f"\n{'=' * 72}\nQuery: {query!r}   ({why})\n{'=' * 72}")

        for label, retriever in (("dense ", dense), ("bm25  ", lexical), ("hybrid", hybrid)):
            result = retriever.retrieve(query)
            if not result.hits:
                print(f"  {label}  (nothing matched)")
                continue
            top = result.hits[0]
            print(f"  {label}  [{top.score:.4f}] {top.chunk.content[:62]}…")

    # -------------------------------------------------------------------
    # Why RRF rather than score normalisation.
    # -------------------------------------------------------------------
    print(f"\n{'=' * 72}\nWhy fuse ranks instead of scores\n{'=' * 72}")

    dense_hits = dense.retrieve("expired token").hits
    bm25_hits = lexical.retrieve("expired token").hits

    print("  dense scores:", [round(h.score, 3) for h in dense_hits])
    print("  bm25 scores: ", [round(h.score, 3) for h in bm25_hits])
    print(
        "\n  These live on incompatible scales — cosine similarity is bounded in\n"
        "  [-1, 1], BM25 is unbounded and corpus-dependent. Normalising them needs\n"
        "  statistics that change on every ingest. Their *ranks* need nothing."
    )

    # -------------------------------------------------------------------
    # Provenance: which leg found each hit.
    # -------------------------------------------------------------------
    print(f"\n{'=' * 72}\nWhich leg contributed each hit\n{'=' * 72}")
    for hit in hybrid.retrieve("expired token").hits:
        print(f"  [{hit.score:.4f}] {hit.retriever:<14} {hit.chunk.content[:52]}…")
    print("\n  A chunk found by both legs is ranked above one found by only one.")

    # -------------------------------------------------------------------
    # In a pipeline, this is one word.
    # -------------------------------------------------------------------
    print(f"\n{'=' * 72}\nThe same thing through the builder\n{'=' * 72}")
    rag = Windlass.rag().llm("fake", responses=["Rotate the key."]).retriever("hybrid").top_k(2)
    for text in KNOWLEDGE_BASE:
        rag.ingest_text(text)

    for hit in rag.search("E1042"):
        print(f"  [{hit.score:.4f}] {hit.chunk.content[:60]}…")


if __name__ == "__main__":
    main()
