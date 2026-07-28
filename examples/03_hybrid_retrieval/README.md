# 03 — Hybrid retrieval

Why dense-only search quietly fails, and what Reciprocal Rank Fusion fixes.

## Run

```bash
pip install windlass
python examples/03_hybrid_retrieval/main.py
```

## The problem

Vector search and BM25 fail in **opposite** ways:

| Query | Dense embeddings | BM25 |
|---|---|---|
| `E1042` | poor — an error code has no semantic neighbourhood | **strong** — exact term match |
| `my login keeps failing` | **strong** — matches "authentication failures" | poor — no shared vocabulary |
| `expired token` | good | good |

Pick either alone and you have a system that is confidently wrong about half your queries. Users notice the half you did not test.

## What the example shows

1. **The same three queries** run through dense, BM25 and hybrid retrieval side by side.
2. **Why RRF rather than score normalisation** — the two score scales are genuinely incompatible.
3. **Provenance** — which leg found each hit, and why agreement between legs ranks higher.
4. **The one-word version** — `.retriever("hybrid")` in a pipeline.

## Why Reciprocal Rank Fusion

A cosine similarity of 0.81 and a BM25 score of 14.3 are not comparable quantities. Normalising them into a common range needs corpus statistics — min, max, distribution — that change every time you ingest a document.

RRF sidesteps that entirely by combining **ranks**:

```
score(chunk) = Σ  weight_i / (k + rank_i)
              legs
```

Ranks need no normalisation, no corpus statistics, and no re-tuning after an ingest. The constant `k` (60 by default) damps the influence of the very top positions.

The consequence you can see in the output: a chunk found by *both* legs outranks one found by only one, even if its individual scores were lower. Agreement between independent retrievers is itself evidence.

## In practice

```python
rag = Windlass.rag().retriever("hybrid")                    # equal weight
rag = Windlass.rag().retriever("hybrid", weights=[0.7, 0.3])  # favour dense
```

Raise the dense weight for conversational corpora; raise the sparse weight for technical documentation full of identifiers.

The legs run **concurrently**, so hybrid costs about as much wall-clock time as its slower leg — not the sum.

## Adding your own leg

Hybrid fuses any number of retrievers of any kind. Your existing search service can be one of them:

```python
rag = Windlass.rag().retriever(
    "hybrid",
    retrievers=[dense_retriever, bm25_retriever, MyCompanySearch(client=...)],
    weights=[0.4, 0.3, 0.3],
)
```

See example 04 and the [plugin guide](../../docs/guides/plugins.md).

## The rule of thumb

**If your corpus contains identifiers — error codes, SKUs, function names, ticket numbers, product names — use hybrid retrieval.** Dense-only search will miss them, and you will not find out from your own testing.
