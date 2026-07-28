# 01 — RAG basics

Ingest documents, retrieve relevant chunks, generate an answer. The whole loop, with nothing installed but Windlass itself.

## Run

```bash
pip install windlass
python examples/01_rag_basics/main.py
```

## What it shows

1. **Building a pipeline** — the fluent chain, and how it reads as a description of the system.
2. **Ingestion** — loading, chunking, embedding and indexing in one call.
3. **Retrieval on its own** — the first thing to inspect when an answer is wrong.
4. **Full generation** — with sources, token counts and latency.
5. **Metadata filtering** — the mechanism behind multi-tenancy.
6. **Strict mode** — refusing to answer when nothing relevant was retrieved.
7. **Idempotent ingestion** — re-ingesting unchanged content does not duplicate it.

## Expected output

```
Indexed 6 chunks from 3 documents

--- Retrieval only ---
  1. [0.033] billing.md   # Billing > Refunds

Customers may request a full ref…
  2. [0.016] releases.md  # Release process > Approval

Every production rele…
  3. [0.016] support.md   # Support > Response times

Enterprise customers rec…

--- Full answer ---
Answer:   Customers may request a full refund within 30 days.
Sources:  billing.md
Contexts: 3
Tokens:   ...
Latency:  ...ms

--- Filtered retrieval ---
  releases.md  # Release process > Approval

Every production release must…

--- Nothing relevant indexed ---
  I could not find anything relevant in the indexed documents to answer that.

Re-ingested the same corpus: 6 chunks -> 6 chunks
```

Exact scores will vary; the ordering will not.

## Notes on what you are seeing

**The heading path.** Chunks begin with `# Billing > Refunds` because the `markdown` chunker prefixes each chunk with its position in the document. That context is nearly free and measurably improves retrieval.

**Fused scores.** Scores look small because hybrid retrieval uses Reciprocal Rank Fusion, which combines *ranks* rather than raw scores. A cosine similarity of 0.81 and a BM25 score of 14.3 are not comparable quantities; their ranks are.

**The scripted model.** `llm("fake", ...)` returns a fixed answer. That is deliberate — it lets you verify that chunking, embedding, indexing and retrieval all work before spending a token.

## Things to try

Swap in a real model:

```python
.llm("gpt-4o-mini")          # needs windlass[openai] and OPENAI_API_KEY
.llm("ollama", model="llama3.2")   # needs a local Ollama, no extra install
```

Watch what changes:

```python
.chunker("recursive")        # no heading path — compare the retrieved text
.retriever("vector")         # dense only — try searching for "E1042"
.strict(False)               # let the model answer without context
.observe("console")          # print a trace tree
```

That "E1042" experiment is the point of example 03.
