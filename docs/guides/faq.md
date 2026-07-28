# FAQ

## About the project

### Is this a wrapper around LangChain?

No. Windlass has no dependency on LangChain and does not use it internally. LangGraph is one optional agent runtime among two; the built-in runtime needs nothing.

The distinction that matters: a wrapper re-exposes another library's object model with different names. Windlass defines its own interfaces and data model, and adapts libraries *into* them. That is why you can swap FAISS for Pinecone, or the built-in agent loop for LangGraph, without your application noticing.

### Why another framework?

Because the integration cost is real, and it becomes your architecture. Every AI library has its own object model, configuration style and installation story. Wiring them together is a week of work, and it is the week you rewrite every time you change one of them.

Windlass's contribution is not any individual adapter — it is the rule that **nothing imports a concrete provider**, and the registry, container and builder that make that rule practical.

### What does it actually give me over calling the SDKs directly?

For a single-provider prototype: not much, and you should call the SDK directly.

It earns its place when you have more than one of: several providers to support, a corpus of mixed formats to ingest, hybrid retrieval, guardrails on both stages, evaluation in CI, tracing, human approval on consequential actions, or a need to test any of that offline. Those are the pieces that are individually simple and collectively a project.

### Is it production ready?

The framework is: typed, tested (over 650 tests, mypy clean), with real error handling, retries, thread safety and observability throughout. It is pre-1.0, which means the public API may still change between minor versions — every break is recorded in the changelog with a migration path.

### What is the license?

Apache 2.0.

---

## Getting started

### Do I need an API key to try it?

No. `pip install windlass` gives you a complete RAG pipeline and agent that run offline, using a scripted model, hashed embeddings and an in-memory store. That is how the test suite runs, and how the tutorials start.

### Which model should I use?

For RAG answers, a mid-tier model at temperature 0 is usually right — the context does the work. `gpt-4o-mini` and `claude-haiku` are strong defaults. Reserve a frontier model for synthesis and reasoning.

For local, Ollama needs no extra install: `Windlass.rag().llm("ollama", model="llama3.2")`.

### Which embedding model?

`BAAI/bge-small-en-v1.5` (384 dims, fast, English) is the default and is genuinely good. Step up to `BAAI/bge-m3` for multilingual or for the highest quality; it is considerably heavier.

Embedding model choice matters more than model choice for retrieval quality. Changing it requires re-ingesting.

### Which vector store?

Start with `memory`. It does exact search, has no recall loss, and is fine to roughly 100k chunks. Move to `faiss` when you outgrow it locally, `chroma` when you want persistence without a server, `pinecone` when you want somebody else to run it.

The interface does not change, so this is a one-word decision you can defer.

---

## Usage

### How do I use a model Windlass does not list?

If it is OpenAI-compatible (vLLM, LM Studio, OpenRouter, Together, most self-hosted gateways):

```python
Windlass.llm("openai", base_url="https://my-gateway/v1", api_key="...")
```

Otherwise, write an adapter — it is one method. See [Writing plugins](plugins.md).

### Can I use my existing search service instead of a vector database?

Yes, and it is one of the most common real plugins. Implement `Retriever.aretrieve_chunks`, register it, and it composes into hybrid retrieval alongside dense search like any built-in. There is a worked example in the [plugin guide](plugins.md#a-retriever-backed-by-your-existing-search-service).

### Can I use Windlass with FastAPI / Django / Celery?

Yes. Every method has both an async and a blocking form, and the blocking form is safe inside a running event loop — Windlass detects it and dispatches to a background loop thread. In async code, prefer the `a*` methods to avoid the hop.

Build the pipeline once at startup, not per request.

### Is it thread-safe?

Yes. The registry, container, in-memory store, BM25 index, caches, memory backends and checkpointers are all guarded by locks. Components are safe to share across threads and concurrent requests.

### How do I mix RAG and agents?

A RAG pipeline is just a tool:

```python
@tool
def search_docs(query: str) -> list[dict]:
    """Search internal documentation.

    Args:
        query: What to look for.
    """
    return [{"content": h.chunk.content, "source": h.chunk.metadata.get("source")}
            for h in rag.search(query, k=5)]

agent = Windlass.agent().llm("gpt-4o").tool(search_docs)
```

Nothing special is needed to combine them.

### How do I do multi-tenancy?

Two mechanisms, used together: tag at ingestion, filter at retrieval.

```python
rag.ingest(path, metadata={"tenant": tenant})
rag.ask(question, filters={"tenant": tenant})
```

For strong isolation, give each tenant its own namespace or collection, so a filtering bug cannot cross the boundary at all.

### Can I change the prompt?

```python
rag.prompt("""Your instructions here.

<context>
{context}
</context>

Question: {question}

Answer:""")
```

`{context}` and `{question}` are required.

### How do I stop it hallucinating?

In order of effect:

1. `rag.strict()` — refuse when nothing is retrieved. This removes the *opportunity*.
2. Fix retrieval so the right context is actually present.
3. Measure `faithfulness` and gate on it in CI.
4. Then, prompt instructions.

Prompt engineering is fourth on that list for a reason.

### How much does it cost to run?

Windlass itself is free. Your spend is provider calls, and the levers are: context size (`top_k` × chunk size, the largest controllable cost), model choice per job, and caching embeddings.

Track it:

```python
metrics.histogram("rag.tokens", answer.usage.total_tokens)
```

---

## Extending

### How do I add my own component?

Subclass the interface, implement one or two methods, register a name. See [Writing plugins](plugins.md).

### How do I ship a component for others to use?

Publish an entry point:

```toml
[project.entry-points."windlass.retriever"]
my-search = "my_package.retrievers:MySearch"
```

Installing your package makes it available in Windlass. No code change on either side.

### What if I need something Windlass genuinely does not model?

Take the underlying object:

```python
graph  = agent.native_graph()   # a real LangGraph StateGraph
index  = rag.native_store()     # the raw faiss.Index
client = rag.native_llm()       # the raw openai.AsyncOpenAI
```

`native()` is on the base `Component` class, so **every** component built on an interface has it — including ones you have not read the source of. See [Three levels](../concepts/three-levels.md).

### Can I add a whole new *kind* of component?

```python
REGISTRY.add_kind("reranker")

@register.component("reranker", "cohere")
class CohereReranker: ...
```

---

## Testing and operations

### How do I test without calling a model?

```python
from windlass.testing import fake_rag, fake_agent, call

rag = fake_rag(["Refunds within 30 days."], answers=["30 days."])
agent = fake_agent(["", "Done."], tools=[my_tool],
                   tool_calls=[[call("my_tool", x=1)], []])
```

See [Testing](testing.md).

### How do I know if quality regressed?

```python
report = rag.evaluate(GOLDEN_SET, metrics=["faithfulness", "answer_relevancy"])
assert report.summary["faithfulness"] >= 0.85
```

Twenty golden questions and a threshold turn "it feels worse" into a red build.

### Does Windlass send telemetry?

No. There is a `telemetry` setting, off by default, and Windlass ships no telemetry backend — the flag exists so downstream distributions can honour it.

### Where do my documents go?

Wherever you configure. The default in-memory store keeps everything in your process. Model calls go to whichever provider you name; local providers (Ollama, HuggingFace embeddings) mean your documents never leave the machine.

---

## Comparisons

### LangChain?

LangChain is larger and has more integrations. Windlass is smaller, typed throughout, has a tiny core install, and does not require you to learn a new abstraction per component kind — there are thirteen interfaces, and that is the whole surface.

They are not mutually exclusive: a LangChain object can be wrapped in a Windlass adapter in a few lines.

### LlamaIndex?

LlamaIndex is deeply focused on RAG and has more indexing strategies. Windlass treats RAG and agents as equal citizens behind one API, with a stronger emphasis on the plugin architecture and on offline testability.

### Just calling the SDKs?

For one provider and one use case, calling the SDK is the right answer and you should do it. Windlass pays off at the point where the integration itself becomes the thing you maintain.

---

## Contributing

### How do I contribute?

See [CONTRIBUTING.md](https://github.com/Kukilbharadwaj/WINDLASS/blob/main/CONTRIBUTING.md). The most valuable contributions are provider adapters, loaders, and documentation from someone who has just been confused by something.

### Can I add a provider for X?

Yes, please. Providers follow a consistent shape — read `providers/llm/openai.py` as the reference. The architecture rules are in `CONTRIBUTING.md`, and there is one that is non-negotiable: **no module may import a concrete provider**.
