# Three levels of API

Every feature in Windlass works at three levels of control. This is not a layering accident — it is the central design commitment, and it is what keeps the framework useful past the tutorial.

## Level 1 — it just works

Sensible defaults, no configuration.

```python
rag = Windlass.rag()
rag.ingest("./documents")
print(rag.ask("What is our refund policy?"))
```

```python
agent = Windlass.agent().llm("gpt-4o").tool(search)
print(agent.run("Find the Q3 numbers"))
```

The defaults are chosen to be *correct*, not merely present: recursive chunking at 1000/200, dense retrieval at top-5, a 6000-token context budget, retries on transient failures only, and a refusal to answer when nothing was retrieved.

## Level 2 — configure it

Same API, more arguments. Nothing is hidden behind a different entry point.

```python
rag = (
    Windlass.rag()
    .chunker("semantic", chunk_size=1000, overlap=200, threshold=0.6)
    .embedding("huggingface", model="BAAI/bge-m3", batch_size=128)
    .vectordb("pinecone", collection="docs", namespace="tenant-42")
    .retriever("hybrid", weights=[0.6, 0.4], fetch_k=40)
    .top_k(8)
    .max_context_tokens(12_000)
    .strict()
)
```

Every builder method takes provider-specific keyword arguments and forwards them. When a provider grows an option, you can use it immediately — no Windlass release required.

### Instances count as configuration

Anywhere a name works, a configured instance works:

```python
from windlass.providers.vectordb.chroma import ChromaVectorStore

store = ChromaVectorStore(host="chroma.internal", port=8000, collection="prod")
rag = Windlass.rag().vectordb(store)
```

And so does a factory, if construction needs to be deferred:

```python
rag = Windlass.rag().vectordb(lambda: ChromaVectorStore(host=discover_host()))
```

## Level 3 — reach past it

!!! quote "The promise"
    **Windlass simplifies libraries. It never hides them.**

Every adapter exposes the object it wraps. When the abstraction does not cover what you need, you take the underlying object and use it directly — without forking Windlass, subclassing internals, or abandoning everything else the framework does for you.

### The escape hatches

```python
rag.native_llm()         # openai.AsyncOpenAI, anthropic.AsyncAnthropic, …
rag.native_store()       # faiss.Index, chromadb Collection, Pinecone Index
rag.native_retriever()   # whatever the retriever wraps

agent.native_llm()
agent.native_graph()     # a real LangGraph StateGraph

component.native()       # on any component, always
```

### Building an agent graph by hand

The built-in runtime is a loop. When you need conditional routing, subgraphs or an explicit state machine, build with `.graph()` and edit the real thing:

```python
from langgraph.graph import END

agent = Windlass.agent().llm("gpt-4o").tool(search).graph()

graph = agent.native_graph()          # the uncompiled StateGraph

graph.add_node("critic", critique)
graph.add_edge("tools", "critic")
graph.add_conditional_edges(
    "critic",
    lambda state: "retry" if state["needs_work"] else "done",
    {"retry": "agent", "done": END},
)

agent.recompile()                     # pick up the changes
response = agent.run("Research and write a summary")
```

You still get Windlass's tools, memory, guardrails, checkpointing and tracing. You have changed only the topology.

### Using a provider capability Windlass does not model

```python
client = rag.native_llm()                      # openai.AsyncOpenAI
images = await client.images.generate(prompt="a diagram of the pipeline")
```

### Tuning a FAISS index directly

```python
index = rag.native_store()                     # faiss.Index
index.nprobe = 32                              # recall/latency trade-off
faiss.write_index(index, "tuned.faiss")
```

### Adding a tracer span of your own

```python
tracer = rag.build().tracer

with tracer.span("business-rules", kind="chain") as span:
    span.set_input(order)
    result = apply_rules(order)
    span.set_output(result)
```

Your span nests inside the surrounding Windlass trace automatically.

## Why this matters

Most frameworks give you Level 1 and Level 2. The failure mode is familiar: the abstraction covers 90% of what you need, and the last 10% forces a rewrite — because the framework owns the objects and will not give them back.

Windlass treats that last 10% as a first-class requirement. `native()` is on the base `Component` class, which means **every** component built on an interface has it — including ones you have not read the source of. (The two exceptions are `cache` and `checkpointer`, which wrap no SDK and so have nothing to hand back.)

The practical consequence: adopting Windlass is not a bet that its abstractions will cover every future requirement. It is a bet that they will cover the common case — and you keep the underlying library either way.

## Choosing a level

| Situation | Level |
|---|---|
| Prototyping, learning, a demo | 1 |
| Production with known requirements | 2 |
| A provider feature Windlass does not model | 3 |
| Custom graph topology, conditional routing | 3 |
| A tuning knob specific to one index type | 3 |
| Behaviour you want everywhere, repeatedly | Write a [plugin](../guides/plugins.md) instead |

That last row is the important one. Level 3 is an escape hatch, not a design pattern. If you reach for it in the same way three times, the right move is to write a component and register it — then it is Level 2 for you and everyone else on your team.
