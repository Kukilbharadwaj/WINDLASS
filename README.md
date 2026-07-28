<div align="center">

# Windlass

**The modular AI application framework.**

One elegant, consistent, extensible API for agents, RAG, tools, MCP, guardrails, evaluation and observability.

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230)](https://github.com/astral-sh/ruff)
[![Typed](https://img.shields.io/badge/typing-PEP%20561-blue)](https://peps.python.org/pep-0561/)

</div>

---

## The problem

Building an AI product today means integrating LangGraph, LangChain, FastMCP, Hugging Face, FAISS, ChromaDB, Pinecone, BM25, RAGAS, DeepEval, LangSmith, Langfuse and NeMo Guardrails. Every one has its own object model, configuration style, installation story and design philosophy. You spend your time on glue code instead of on your product — and when you want to swap FAISS for Pinecone, or OpenAI for a local model, the glue is what you have to rewrite.

**Windlass is not a wrapper.** It is an architecture: thirteen interfaces, a component registry, a dependency-injection container, and a fluent builder that assembles them. Every part is replaceable, nothing is tightly coupled, and the underlying library is always one method call away.

## Install

```bash
pip install windlass
```

That is the whole core: `pydantic`, `pydantic-settings`, `httpx` and `typing-extensions`. No torch, no LangChain, no vector database driver. It already gives you a working end-to-end RAG pipeline and a working agent, offline, with no API keys — because Windlass ships dependency-free implementations of every essential component.

Add what you actually need:

```bash
pip install "windlass[openai]"          # OpenAI models
pip install "windlass[agent]"           # LangGraph agent runtime
pip install "windlass[rag]"             # embeddings, vector stores, loaders
pip install "windlass[mcp]"             # MCP client
pip install "windlass[evaluation]"      # RAGAS + DeepEval
pip install "windlass[guardrails]"      # NVIDIA NeMo Guardrails
pip install "windlass[observability]"   # LangSmith + Langfuse
pip install "windlass[agent,mcp,openai]"  # combine freely
pip install "windlass[full]"            # everything
```

Unused dependencies are never installed, and never imported. Use a feature whose extra is missing and you get an actionable error, not a traceback:

```
MissingDependencyError: The Pinecone vector store is not installed.

Hint: Run:

    pip install "windlass[pinecone]"
```

## RAG in five lines

```python
from windlass import Windlass

rag = Windlass.rag()
rag.ingest("./documents")           # PDFs, DOCX, XLSX, Markdown, HTML, CSV, images…
print(rag.ask("What changed in the pricing model last quarter?"))
```

Configure any part of it without leaving the chain:

```python
rag = (
    Windlass.rag()
    .loader("pdf")
    .preprocessor("clean").preprocessor("pii", action="redact")
    .chunker("semantic", chunk_size=1000, overlap=200)
    .embedding("huggingface", model="BAAI/bge-small-en-v1.5")
    .vectordb("pinecone", collection="docs")
    .retriever("hybrid", weights=[0.6, 0.4])
    .guardrails()
    .observe("langfuse")
    .top_k(8)
    .strict()                        # refuse to answer without context
)

rag.ingest("./documents")
answer = rag.ask("Explain transformers")

print(answer.answer)
print(answer.sources)                # cited sources, deduplicated
print(answer.usage.total_tokens)
```

## Agents in five lines

```python
from windlass import Windlass, tool

@tool
def get_weather(city: str) -> dict:
    """Look up the current weather.

    Args:
        city: City name, e.g. "Paris".
    """
    return weather_api.current(city)

agent = Windlass.agent().llm("gpt-4o").tool(get_weather).memory()
print(agent.run("Should I take an umbrella in Paris today?"))
```

The JSON schema is derived from your type hints and docstring. There is no second source of truth to keep in sync.

```python
agent = (
    Windlass.agent()
    .llm("claude-sonnet-4-5")
    .tool(get_weather, search_docs, send_email)
    .mcp(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "."])
    .system("You are a research assistant. Always cite your sources.")
    .memory("summary")
    .guardrails(pii=True, on_violation="redact")
    .checkpoint("sqlite", path="./state.db")
    .human_in_the_loop()
    .observe("langfuse")
)
```

## Three levels, always

Every feature in Windlass works at three levels of control. You choose how much you need.

**Level 1 — it just works.**

```python
rag.ask("What is RAG?")
```

**Level 2 — configure it.**

```python
rag.chunker("semantic", chunk_size=1000, overlap=200)
```

**Level 3 — reach past it.** Windlass simplifies libraries; it never hides them.

```python
graph = agent.native_graph()            # a real LangGraph StateGraph
graph.add_node("critic", critic_fn)
graph.add_conditional_edges("critic", route, {"retry": "agent", "done": END})
agent.recompile()

index  = rag.native_store()             # the raw faiss.Index
client = rag.native_llm()               # the raw openai.AsyncOpenAI
```

## What is included

| Area | Implementations |
|---|---|
| **LLMs** | OpenAI · Anthropic · Gemini · Groq · Ollama · Fake (offline) |
| **Agents** | Built-in ReAct runtime · LangGraph · Supervisor multi-agent · HITL · checkpointing · streaming |
| **Tools** | Automatic schema generation · parallel execution · approval gating · MCP proxying |
| **MCP** | FastMCP (stdio/SSE/HTTP) · multi-server aggregation · tools, resources, prompts |
| **Embeddings** | HuggingFace Inference API · HuggingFace local · OpenAI · Hash (offline) |
| **Loaders** | PDF · DOCX · PPTX · XLSX · CSV · Markdown · HTML · JSON · images (OCR) · audio · web · YouTube |
| **Preprocessing** | Cleaning · PII detection & redaction · deduplication · language detection · OCR fallback · table extraction · LLM metadata |
| **Chunking** | Recursive · semantic · token · Markdown · code · parent-child |
| **Retrieval** | BM25 · dense vector · hybrid (RRF) · contextual · HyDE · multi-query · MMR |
| **Vector DBs** | FAISS · ChromaDB · Pinecone · in-memory (offline) |
| **Memory** | Buffer · sliding window · summarising · vector long-term · composite |
| **Guardrails** | Rule-based (PII, injection, secrets, keywords) · NVIDIA NeMo Guardrails |
| **Evaluation** | Built-in lexical & LLM-judged metrics · RAGAS · DeepEval |
| **Observability** | Console · in-memory · null · multi-backend fan-out · LangSmith · Langfuse |

## Everything is replaceable

Every component is resolved through a registry. Register your own and it becomes indistinguishable from a built-in one — no Windlass source is touched, no subclassing of internals, no monkeypatching.

```python
from windlass import register, Chunker

@register.chunker("by-sentence")
class SentenceChunker(Chunker):
    """Splits on sentence boundaries."""

    def split_text(self, text: str) -> list[str]:
        return [s for s in text.split(". ") if s]

rag = Windlass.rag().chunker("by-sentence")
```

Ship it as a package and it installs into other people's Windlass:

```toml
[project.entry-points."windlass.chunker"]
by-sentence = "my_package.chunkers:SentenceChunker"
```

The same applies to all fifteen kinds: `llm`, `embedding`, `loader`, `preprocessor`, `chunker`, `retriever`, `vectordb`, `memory`, `guardrail`, `evaluator`, `tracer`, `tool`, `mcp`, `cache`, `checkpointer`.

## Async, streaming, and production concerns

Every interface is async-first, with blocking variants derived from the same implementation — so the two can never drift.

```python
answer = await rag.aask("...")
async for token in rag.astream_ask("..."):
    print(token, end="")

async for event in agent.astream("..."):
    if event.type == "tool_call":
        print(f"calling {event.tool_call.name}")
```

`run_sync` detects an already-running event loop and dispatches to a background loop, so the blocking API is safe inside Jupyter, FastAPI and LangServe.

Built in throughout: bounded-concurrency batching, exponential backoff with jitter on transient failures only, optional response and embedding caching, lazy imports, thread safety, and structured tracing on every stage.

## Evaluate what you built

```python
report = rag.evaluate(
    [
        {"question": "What is the refund window?", "reference": "30 days"},
        {"question": "Who signs off on releases?", "reference": "The release manager"},
    ],
    metrics=["faithfulness", "answer_relevancy", "context_precision_lexical"],
)

print(report)
# EvaluationReport(2 samples, pass_rate=100.0%)
#   answer_relevancy         0.900
#   context_precision_lexical 1.000
#   faithfulness             0.950
```

`faithfulness` is the one to watch — it is the direct measure of hallucination, and it catches a retrieval regression before your users do.

## Testing

An AI application is testable when its model is replaceable. Windlass ships the doubles.

```python
from windlass.testing import fake_rag, fake_agent, capture_spans, call

def test_pipeline_retrieves_the_right_section():
    rag = fake_rag(["Refunds are available within 30 days."], answers=["30 days."])
    answer = rag.ask("What is the refund window?")
    assert "30 days" in answer.contexts[0].chunk.content

def test_agent_calls_search_before_answering():
    agent = fake_agent(
        ["", "Found it."],
        tools=[search],
        tool_calls=[[call("search", query="refunds")], []],
    )
    with capture_spans() as spans:
        assert agent.run("find the refund policy").output == "Found it."
    assert spans.count("tool") == 1
```

No network, no keys, no flakiness.

## Command line

```bash
windlass doctor                        # diagnose an install: extras, credentials, smoke test
windlass list retriever                # what is registered, with descriptions
windlass config                        # effective settings, secrets masked
windlass ask "..." --docs ./documents  # one-shot RAG query
windlass chat --llm ollama             # interactive agent session
```

## Configuration

Configuration comes from defaults, then `windlass.toml`, then the environment, then your code — each layer overriding the last. Conventional provider variables are read as-is, so an existing `OPENAI_API_KEY` in your shell just works.

```python
from windlass import configure

configure(
    default_llm="anthropic",
    temperature=0.0,
    max_concurrency=16,
    retry={"attempts": 5},
    cache={"enabled": True, "backend": "disk"},
)
```

## Documentation

The full documentation site — architecture, concepts, tutorials from beginner to advanced, complete API reference, plugin development guide, best practices, troubleshooting and migration guides — is built with MkDocs:

```bash
pip install "windlass[docs]"
mkdocs serve
```

Start with `docs/getting-started/quickstart.md`, then `docs/concepts/architecture.md`.

## Contributing

Windlass is built to be extended, and contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, the architecture rules that keep the framework decoupled, and what a good pull request looks like.

```bash
git clone https://github.com/Kukilbharadwaj/WINDLASS && cd WINDLASS
python -m venv venv && venv/Scripts/activate   # or: source venv/bin/activate
pip install -e ".[dev]"
pre-commit install
pytest
```

## License

Apache 2.0. See [LICENSE](LICENSE).
