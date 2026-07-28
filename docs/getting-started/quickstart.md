# Quickstart

Ten minutes, no API key, no extras. Everything on this page runs on `pip install windlass` alone.

## 1. Your first RAG pipeline

```python
from windlass import Windlass

rag = Windlass.rag()

rag.ingest_text(
    "Windlass is a modular AI application framework. It unifies the modern AI "
    "ecosystem behind one consistent, extensible API."
)

answer = rag.ask("What is Windlass?")
print(answer)
```

Three things just happened, all with defaults:

1. The text was **chunked** (recursive, 1000 characters, 200 overlap).
2. Each chunk was **embedded** (hashed n-grams — deterministic, no dependency) and written to an in-memory **vector store**.
3. The question was embedded, the nearest chunks **retrieved**, and a prompt built and sent to the model.

The default model is the `fake` provider, so the answer is scripted. That is deliberate: it lets you verify the *plumbing* before you spend a token.

### Look at what was retrieved

The answer is only as good as the retrieval, so inspect it directly:

```python
answer = rag.ask("What is Windlass?")

for hit in answer.contexts:
    print(f"{hit.score:.3f}  {hit.chunk.content[:80]}")

print(answer.sources)            # deduplicated source list
print(answer.usage.total_tokens)
print(f"{answer.latency_ms:.0f}ms")
```

Or skip generation entirely:

```python
for hit in rag.search("modular framework", k=3):
    print(hit.rank, hit.score, hit.chunk.content[:60])
```

## 2. Point it at a real model

Install a provider and name it:

```bash
pip install "windlass[openai]"
export OPENAI_API_KEY=sk-...
```

```python
rag = Windlass.rag().llm("gpt-4o-mini")
rag.ingest("./documents")
print(rag.ask("What is our refund policy?"))
```

No API key? Run a model locally instead — Ollama needs no extra install:

```bash
ollama pull llama3.2
```

```python
rag = Windlass.rag().llm("ollama", model="llama3.2")
```

## 3. Ingest real documents

```python
rag.ingest("./documents")          # a directory of mixed formats
rag.ingest("report.pdf")           # one file
rag.ingest("https://example.com")  # a web page
rag.ingest(["a.pdf", "b.docx"])    # several sources
```

Format detection is automatic: each file is routed to the loader that claims its extension. A folder of PDFs, spreadsheets and Markdown works with no configuration.

Most formats need `pip install "windlass[loaders]"`. Text, Markdown, JSON, CSV and HTML work out of the box.

## 4. Configure the pipeline

Every stage is a method on the chain:

```python
rag = (
    Windlass.rag()
    .llm("gpt-4o-mini", temperature=0)
    .embedding("huggingface", model="BAAI/bge-small-en-v1.5")
    .chunker("semantic", chunk_size=1000)
    .vectordb("faiss", persist_path="./index")
    .retriever("hybrid")
    .preprocessor("clean")
    .preprocessor("pii", action="redact")
    .guardrails()
    .observe("console")
    .top_k(8)
    .strict()
)
```

Read that top to bottom and you know exactly what the system does. That is the point.

!!! tip "`.strict()` is the anti-hallucination switch"
    With it on, retrieval finding nothing returns a refusal instead of calling the model with an empty context. Removing the model's opportunity to invent is more effective than asking it not to.

## 5. Your first agent

```python
from windlass import Windlass, tool

@tool
def calculate(expression: str) -> float:
    """Evaluate an arithmetic expression.

    Args:
        expression: A Python arithmetic expression, e.g. "2 + 2 * 3".
    """
    return eval(expression, {"__builtins__": {}}, {})   # (1)!

agent = Windlass.agent().llm("gpt-4o-mini").tool(calculate)
print(agent.run("What is 17% of 2,400?"))
```

1. A locked-down `eval` is fine for a demo. In production, use a real expression parser — a tool is code an LLM can trigger, so treat its input as untrusted.

The schema the model sees is derived from the signature and docstring:

```python
>>> calculate.schema()["function"]
{'name': 'calculate',
 'description': 'Evaluate an arithmetic expression.',
 'parameters': {'type': 'object',
                'properties': {'expression': {'type': 'string',
                                              'description': 'A Python arithmetic expression, e.g. "2 + 2 * 3".'}},
                'additionalProperties': False,
                'required': ['expression']}}
```

### See what it did

```python
response = agent.run("What is 17% of 2,400?")

print(response.output)
for step in response.steps:
    for call, result in zip(step.tool_calls, step.tool_results):
        print(f"  {call.name}({call.arguments}) -> {result.content}")
print(response.usage.total_tokens)
```

## 6. Give the agent memory

```python
agent = Windlass.agent().llm("gpt-4o-mini").memory()

agent.run("My name is Ada.", thread_id="user-42")
print(agent.run("What is my name?", thread_id="user-42"))
```

`thread_id` scopes the conversation, so one agent serves many concurrent users.

## 7. Stream

```python
for token in rag.stream_ask("Summarise the Q3 report"):
    print(token, end="", flush=True)
```

```python
for event in agent.stream("Research this and summarise"):
    if event.type == "text":
        print(event.delta, end="", flush=True)
    elif event.type == "tool_call":
        print(f"\n[calling {event.tool_call.name}]")
```

## 8. Go async

Every method has an `a`-prefixed coroutine:

```python
import asyncio

async def main():
    rag = Windlass.rag().llm("gpt-4o-mini")
    await rag.aingest("./documents")

    answers = await asyncio.gather(
        rag.aask("What is the refund policy?"),
        rag.aask("Who approves releases?"),
        rag.aask("When is the next audit?"),
    )
    for answer in answers:
        print(answer.answer)

asyncio.run(main())
```

The blocking API is derived from the async one, so it is safe to call from inside a running event loop — a notebook, FastAPI, LangServe. Windlass detects the loop and dispatches accordingly.

## 9. Measure it

```python
report = rag.evaluate(
    [
        {"question": "What is the refund window?", "reference": "30 days"},
        {"question": "Who signs off on releases?", "reference": "The release manager"},
    ],
    metrics=["faithfulness", "answer_relevancy", "f1"],
)
print(report)
```

```
EvaluationReport(2 samples, pass_rate=100.0%)
  answer_relevancy         0.900
  f1                       0.833
  faithfulness             0.950
```

Run this in CI and a retrieval regression fails the build instead of reaching your users.

## 10. Reach past the abstraction

```python
index  = rag.native_store()      # the raw faiss.Index
client = rag.native_llm()        # the raw openai.AsyncOpenAI
graph  = agent.native_graph()    # a real LangGraph StateGraph (with .graph())
```

Windlass gets out of the way when you need it to.

## Next steps

- [Configuration](configuration.md) — settings, environment, config files.
- [Architecture](../concepts/architecture.md) — how the pieces fit together.
- [Beginner tutorial](../tutorials/beginner.md) — build a documentation Q&A bot end to end.
- [Writing plugins](../guides/plugins.md) — add your own component.
