# Migration

## Upgrading Windlass

Windlass follows semantic versioning. Before 1.0, minor versions may change public API; every break is recorded in the [changelog](https://github.com/windlass/windlass/blob/main/CHANGELOG.md) with a migration path.

```bash
pip install --upgrade windlass
windlass doctor
pytest
```

Two habits make upgrades cheap:

**Pin your extras, not just the package.**

```
windlass[openai,rag,observability]>=0.1,<0.2
```

**Keep a golden set.** Twenty questions with reference answers, run through `evaluate()`, will tell you within a minute whether an upgrade changed behaviour you care about.

---

## Coming from LangChain

Windlass has no LangChain dependency, but the concepts map cleanly.

### Concepts

| LangChain | Windlass |
|---|---|
| `ChatOpenAI(...)` | `Windlass.llm("openai", ...)` |
| `OpenAIEmbeddings()` | `Windlass.embedding("openai")` |
| `RecursiveCharacterTextSplitter` | `Windlass.chunker("recursive")` |
| `FAISS.from_documents(...)` | `Windlass.rag().vectordb("faiss")` + `.ingest()` |
| `vectorstore.as_retriever()` | `Windlass.rag().retriever("vector")` |
| `EnsembleRetriever` | `Windlass.rag().retriever("hybrid")` |
| `RetrievalQA` / LCEL chain | `Windlass.rag()` |
| `ConversationBufferMemory` | `Windlass.agent().memory("buffer")` |
| `@tool` | `@tool` — same idea, schema from type hints |
| `AgentExecutor` | `Windlass.agent()` |
| `PyPDFLoader` | `Windlass.loader("pdf")`, or automatic detection |
| Callbacks | `.observe("langfuse")` |

### A RAG chain

=== "LangChain"

    ```python
    from langchain_community.document_loaders import DirectoryLoader
    from langchain_community.vectorstores import FAISS
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains import RetrievalQA

    documents = DirectoryLoader("./docs").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    store = FAISS.from_documents(chunks, OpenAIEmbeddings())

    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        retriever=store.as_retriever(search_kwargs={"k": 5}),
    )
    print(chain.invoke({"query": "What is the refund policy?"})["result"])
    ```

=== "Windlass"

    ```python
    from windlass import Windlass

    rag = (
        Windlass.rag()
        .llm("gpt-4o-mini")
        .embedding("openai")
        .vectordb("faiss")
        .chunker("recursive", chunk_size=1000, overlap=200)
        .top_k(5)
    )
    rag.ingest("./docs")
    print(rag.ask("What is the refund policy?"))
    ```

### An agent

=== "LangChain"

    ```python
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are helpful."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, max_iterations=10)
    print(executor.invoke({"input": "What is the weather in Paris?"})["output"])
    ```

=== "Windlass"

    ```python
    agent = (
        Windlass.agent()
        .llm("gpt-4o")
        .tool(*tools)
        .system("You are helpful.")
        .max_iterations(10)
    )
    print(agent.run("What is the weather in Paris?"))
    ```

### Keeping LangChain components

A LangChain object wraps into a Windlass adapter in a few lines:

```python
from windlass import Chunk, Retriever, ScoredChunk, register


@register.retriever("langchain")
class LangChainRetriever(Retriever):
    """Adapts any LangChain retriever to the Windlass interface."""

    def __init__(self, *, retriever=None, **config):
        super().__init__(**config)
        self.lc = retriever

    async def aretrieve_chunks(self, query, k, *, filters=None, **kwargs):
        documents = await self.lc.ainvoke(query)
        return [
            ScoredChunk(chunk=Chunk(content=d.page_content, metadata=d.metadata), score=1.0)
            for d in documents[:k]
        ]
```

```python
rag = Windlass.rag().retriever(LangChainRetriever(retriever=my_lc_retriever))
```

Which means migration can be incremental: adapt what you have, replace it when convenient.

### Notes

- **Documents.** Windlass `Document` has `content`, not `page_content`. `Chunk` is a separate type — LangChain uses `Document` for both.
- **Async.** Every Windlass method has an `a*` form, and the blocking form is safe inside a running loop.
- **Callbacks.** Replaced by the tracer interface — one `.observe(...)` covers the whole pipeline.
- **LCEL.** There is no equivalent, deliberately. Windlass pipelines are configured objects rather than composed expressions; for custom topology use `.graph()` and edit the LangGraph directly.

---

## Coming from LlamaIndex

| LlamaIndex | Windlass |
|---|---|
| `SimpleDirectoryReader` | `rag.ingest("./docs")` — detection is automatic |
| `VectorStoreIndex.from_documents(...)` | `rag.ingest(...)` |
| `index.as_query_engine()` | `rag.ask(...)` |
| `index.as_retriever()` | `rag.search(...)` |
| `SentenceSplitter` | `Windlass.chunker("recursive")` |
| `SemanticSplitterNodeParser` | `Windlass.chunker("semantic")` |
| `QueryFusionRetriever` | `Windlass.retriever("hybrid")` |
| `Settings.llm = ...` | `configure(default_llm=...)` |
| `ChatEngine` | `rag.memory()` + `thread_id` |
| Node postprocessors | Preprocessors, or a retriever `rerank=` |

=== "LlamaIndex"

    ```python
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
    from llama_index.llms.openai import OpenAI

    Settings.llm = OpenAI(model="gpt-4o-mini")
    documents = SimpleDirectoryReader("./docs").load_data()
    index = VectorStoreIndex.from_documents(documents)
    print(index.as_query_engine().query("What is the refund policy?"))
    ```

=== "Windlass"

    ```python
    from windlass import Windlass

    rag = Windlass.rag().llm("gpt-4o-mini")
    rag.ingest("./docs")
    print(rag.ask("What is the refund policy?"))
    ```

Note that LlamaIndex's `Settings` is a global; Windlass's `configure()` sets defaults that any builder can override locally, which matters when one process serves several pipelines.

---

## Coming from raw SDK code

If you are calling `openai` directly, the pieces map like this:

```python
# Before
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": question}],
    tools=tool_schemas,
)

# After
agent = Windlass.agent().llm("gpt-4o").tool(*tools)
response = agent.run(question)
```

What you gain: the tool loop, schema generation, retries with jitter on transient failures only, guardrails, memory, tracing, and provider portability.

What you keep: the client itself, via `agent.native_llm()`.

Migrating incrementally is fine — Windlass components work standalone, so you can adopt one piece (say, chunking and retrieval) without restructuring the rest.

---

## Version history

### 0.1.0

Initial release. See the [changelog](https://github.com/windlass/windlass/blob/main/CHANGELOG.md).
