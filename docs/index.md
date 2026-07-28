# Windlass

**The modular AI application framework.** One elegant, consistent, extensible API for agents, RAG, tools, MCP, guardrails, evaluation and observability.

---

## The problem it solves

Shipping an AI product today means integrating LangGraph, LangChain, FastMCP, Hugging Face, FAISS, ChromaDB, Pinecone, BM25, RAGAS, DeepEval, LangSmith, Langfuse and NeMo Guardrails. Each has its own object model, configuration style, installation story and design philosophy.

The cost is not the learning curve. The cost is that **the glue becomes your architecture**. When you want to swap FAISS for Pinecone, or OpenAI for a local model, the glue is what you rewrite — and it is spread across your entire codebase.

Windlass is not a wrapper around those libraries. It is an architecture:

- **Thirteen interfaces**, one per kind of replaceable thing.
- **A component registry**, so implementations are looked up by name, never imported directly.
- **A dependency-injection container**, so wiring is data rather than code.
- **A fluent builder**, so the common case is five lines.

Nothing in Windlass imports a concrete provider. That single rule is what makes swapping a provider a string change instead of a refactor.

## Five lines to a RAG pipeline

```python
from windlass import Windlass

rag = Windlass.rag()
rag.ingest("./documents")
print(rag.ask("What changed in the pricing model last quarter?"))
```

## Five lines to an agent

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

The tool's JSON schema comes from its type hints and docstring. There is no second source of truth to keep in sync.

## It works before you configure anything

```bash
pip install windlass
```

The core install is `pydantic` and `httpx`. Nothing else. And yet the example above runs — because Windlass ships dependency-free implementations of every *essential* component: a scripted model, hashed embeddings, an exact in-memory vector store, BM25, recursive chunking, rule-based guardrails, lexical evaluation metrics.

That is not a toy mode. It is what makes the framework testable offline, learnable without an API key, and honest about which parts genuinely need a heavyweight dependency.

## Three levels, always

<div class="grid cards" markdown>

- **Level 1 — it just works**

    ```python
    rag.ask("What is RAG?")
    ```

- **Level 2 — configure it**

    ```python
    rag.chunker("semantic", chunk_size=1000, overlap=200)
    ```

- **Level 3 — reach past it**

    ```python
    graph = agent.native_graph()   # a real LangGraph StateGraph
    index = rag.native_store()     # the raw faiss.Index
    ```

</div>

Windlass simplifies libraries. It never hides them. See [Three levels of API](concepts/three-levels.md).

## Where to go next

<div class="grid cards" markdown>

- :material-download: **[Installation](getting-started/installation.md)** — the extras system, and why nothing you do not use is installed.
- :material-rocket-launch: **[Quickstart](getting-started/quickstart.md)** — a working RAG pipeline and agent in ten minutes.
- :material-sitemap: **[Architecture](concepts/architecture.md)** — how the registry, container and builders fit together.
- :material-school: **[Tutorials](tutorials/beginner.md)** — beginner, intermediate and advanced, each fully runnable.
- :material-puzzle: **[Writing plugins](guides/plugins.md)** — make your own component indistinguishable from a built-in one.
- :material-book-open-variant: **[API reference](reference/windlass/index.md)** — generated from the docstrings in the source.

</div>

## Design principles

| Principle | What it means in practice |
|---|---|
| **Program against interfaces** | No module in Windlass imports a concrete provider. Ever. |
| **Composition over inheritance** | Pipelines are assembled from components, not built by subclassing. |
| **Open for extension** | Every kind is a registry entry; yours registers the same way built-ins do. |
| **Lazy by default** | `import windlass` pulls in no optional dependency and costs less than importing `pydantic-settings`. |
| **Async-first** | Every interface defines `a*` coroutines; the blocking API is derived from them, so the two cannot drift. |
| **Errors that tell you what to do** | A missing extra prints the exact `pip install` command. No raw `ImportError` ever reaches your code. |
| **Degrade, don't crash** | A dead MCP server, a failing hybrid leg, a broken tracer: log and continue. Partial capability beats none. |

## License

Apache 2.0.
