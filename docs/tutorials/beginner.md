# Beginner: a documentation Q&A bot

Build a bot that answers questions about a set of documents, understand every line, then make it good.

**You need:** Python 3.12+ and ten minutes. No API key.

```bash
pip install windlass
```

## Step 1 — Prove the plumbing works

Create `bot.py`:

```python
from windlass import Windlass

rag = Windlass.rag()

rag.ingest_text(
    "Windlass is a modular AI application framework. It unifies the modern AI "
    "ecosystem behind one consistent API. Install it with: pip install windlass"
)

print(rag.ask("How do I install Windlass?"))
```

```bash
python bot.py
```

You get a scripted answer, because the default model is the `fake` provider. That is the point of this step: you have verified chunking, embedding, indexing and retrieval **before** spending a token.

Look at what was actually retrieved:

```python
answer = rag.ask("How do I install Windlass?")

print("Answer:", answer.answer)
print("Retrieved:")
for hit in answer.contexts:
    print(f"  {hit.score:.3f}  {hit.chunk.content[:70]}")
```

If the right text is in there, your pipeline works and only the model is missing.

## Step 2 — Add a real model

=== "Local (free)"

    ```bash
    ollama pull llama3.2
    ```

    ```python
    rag = Windlass.rag().llm("ollama", model="llama3.2")
    ```

=== "OpenAI"

    ```bash
    pip install "windlass[openai]"
    export OPENAI_API_KEY=sk-...
    ```

    ```python
    rag = Windlass.rag().llm("gpt-4o-mini")
    ```

=== "Anthropic"

    ```bash
    pip install "windlass[anthropic]"
    export ANTHROPIC_API_KEY=sk-ant-...
    ```

    ```python
    rag = Windlass.rag().llm("claude-sonnet-4-5")
    ```

Run it again. Same code, real answer.

## Step 3 — Ingest real documents

```bash
mkdir documents
```

Put some Markdown or text files in there, then:

```python
from windlass import Windlass

rag = Windlass.rag().llm("gpt-4o-mini")

chunks = rag.ingest("./documents")
print(f"Indexed {chunks} chunks")

print(rag.ask("What is this project about?"))
```

For PDFs, Word documents and spreadsheets:

```bash
pip install "windlass[loaders]"
```

No code change — format detection is automatic.

## Step 4 — Make it interactive

```python
from windlass import Windlass

rag = Windlass.rag().llm("gpt-4o-mini").memory()
rag.ingest("./documents")

print("Ask me about the documents. Type 'exit' to quit.\n")

while True:
    question = input("you › ").strip()
    if question.lower() in {"exit", "quit"}:
        break
    if not question:
        continue

    print("bot › ", end="", flush=True)
    for token in rag.stream_ask(question, thread_id="cli"):
        print(token, end="", flush=True)
    print("\n")
```

Two additions matter here. `.memory()` makes follow-up questions work — "does that apply to annual plans?" now resolves against the previous turn. `stream_ask` prints tokens as they arrive, which makes the wait feel far shorter than it is.

## Step 5 — Make it good

The version above works. This version is worth deploying.

```python
from windlass import Windlass

rag = (
    Windlass.rag()
    .llm("gpt-4o-mini", temperature=0)      # (1)!
    .chunker("markdown")                    # (2)!
    .retriever("hybrid")                    # (3)!
    .preprocessor("clean")                  # (4)!
    .memory()
    .top_k(6)                               # (5)!
    .strict()                               # (6)!
    .observe("console")                     # (7)!
)

rag.ingest("./documents", show_progress=True)
```

1. Temperature 0 for factual Q&A. Creativity is not what you want when citing documents.
2. Heading-aware chunking. Each chunk carries its heading path, which is nearly free context and measurably improves retrieval.
3. Dense + BM25 fused. Embeddings handle paraphrase; BM25 finds the exact error code or product name that embeddings miss.
4. Normalises Unicode and whitespace, strips page numbers and boilerplate. Typically 5–15% fewer characters to embed, with no loss of meaning.
5. Six chunks instead of five. Cheap to try, easy to measure.
6. **Refuse to answer when nothing was retrieved.** The single most effective anti-hallucination setting available — it removes the opportunity to invent rather than asking the model not to.
7. Print a trace tree so you can see where the time goes.

## Step 6 — Show your sources

An answer you cannot verify is an answer you cannot trust.

```python
answer = rag.ask(question)

print(answer.answer)

if answer.sources:
    print("\nSources:")
    for source in answer.sources:
        print(f"  - {source}")

print(f"\n({answer.usage.total_tokens} tokens, {answer.latency_ms:.0f}ms)")
```

The default prompt already asks the model to cite the `[n]` markers Windlass puts on each context block.

## Step 7 — Don't re-index every time

Ingestion is the slow part. Persist the index and load it on startup:

```bash
pip install "windlass[faiss]"
```

```python
from pathlib import Path
from windlass import Windlass

INDEX = Path("./index")

rag = (
    Windlass.rag()
    .llm("gpt-4o-mini", temperature=0)
    .embedding("huggingface")
    .vectordb("faiss", persist_path=str(INDEX))
    .chunker("markdown")
    .retriever("hybrid")
    .strict()
)

if INDEX.exists():
    rag.load(INDEX)
    print(f"Loaded {rag.count()} chunks")
else:
    rag.ingest("./documents", show_progress=True)
    rag.save(INDEX)
    print(f"Indexed and saved {rag.count()} chunks")
```

Re-running `ingest` later is also cheap: chunk ids are content hashes, so unchanged content upserts instead of duplicating.

## Step 8 — Check that it works

```python
def test_finds_the_installation_instructions():
    hits = rag.search("how do I install")
    assert any("pip install" in h.chunk.content for h in hits)
```

Test *retrieval*, not the model's wording. Retrieval is yours and it is deterministic; the model's phrasing is neither.

When you have a handful of real questions, add a quality gate:

```python
report = rag.evaluate(
    [
        {"question": "How do I install it?", "reference": "pip install windlass"},
        {"question": "What is it for?", "reference": "A modular AI application framework"},
    ],
    metrics=["faithfulness", "answer_relevancy"],
)
print(report)
assert report.summary["faithfulness"] >= 0.8
```

## The finished bot

```python
"""A documentation Q&A bot."""

from pathlib import Path

from windlass import Windlass

INDEX = Path("./index")
DOCUMENTS = Path("./documents")


def build():
    rag = (
        Windlass.rag()
        .llm("gpt-4o-mini", temperature=0)
        .embedding("huggingface")
        .vectordb("faiss", persist_path=str(INDEX))
        .chunker("markdown")
        .retriever("hybrid")
        .preprocessor("clean")
        .memory()
        .top_k(6)
        .strict()
    )

    if INDEX.exists():
        rag.load(INDEX)
    else:
        rag.ingest(DOCUMENTS, show_progress=True)
        rag.save(INDEX)

    return rag


def main():
    rag = build()
    print(f"Ready — {rag.count()} chunks indexed. Type 'exit' to quit.\n")

    while True:
        try:
            question = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question.lower() in {"exit", "quit"}:
            return
        if not question:
            continue

        print("bot › ", end="", flush=True)
        for token in rag.stream_ask(question, thread_id="cli"):
            print(token, end="", flush=True)
        print()

        answer = rag.ask(question, thread_id="cli")
        if answer.sources:
            print("      sources:", ", ".join(answer.sources))
        print()


if __name__ == "__main__":
    main()
```

## What you learned

- A pipeline is a chain of replaceable components, and you can read what a system does from the chain.
- The defaults let you verify plumbing before spending money.
- Retrieval quality is what determines answer quality — inspect `answer.contexts` first, always.
- `.strict()` is the highest-leverage safety setting.
- Persist the index; re-ingestion is idempotent.

## Next

- [Intermediate](intermediate.md) — production concerns: multi-tenancy, guardrails, evaluation gates, deployment.
- [RAG concepts](../concepts/rag.md) — chunking and retrieval in depth.
