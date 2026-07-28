# Components

Everything replaceable in Windlass is a **component**. There are fifteen kinds. Each has exactly one interface, and every implementation — built-in or yours — satisfies the same contract.

## The catalogue

Run `windlass list` to see what your install actually has. Components whose extra is missing are marked.

### `llm` — language models

| Name | Extra | Notes |
|---|---|---|
| `fake` | — | Deterministic and scripted. The reason the test suite runs offline. |
| `echo` | — | Returns the last user message. Useful for wiring tests. |
| `openai` | `openai` | Also Azure, vLLM, LM Studio, OpenRouter — anything OpenAI-compatible |
| `anthropic` | `anthropic` | Claude, with extended thinking support |
| `gemini` | `gemini` | Gemini Developer API and Vertex AI |
| `groq` | `groq` | Open-weight models at very high throughput |
| `ollama` | — | Local models over HTTP; no extra needed |

Interface: [`LLM`][windlass.interfaces.llm.LLM]

### `embedding` — embedding models

| Name | Extra | Notes |
|---|---|---|
| `hash` | — | Hashed n-grams. Deterministic, dependency-free, not semantic |
| `huggingface` | `embeddings` | Local `sentence-transformers`; applies E5/BGE/Nomic instruction prefixes for you |
| `openai` | `openai` | Supports Matryoshka dimension reduction |

Interface: [`Embedder`][windlass.interfaces.embedding.Embedder]

### `vectordb` — vector stores

| Name | Extra | Notes |
|---|---|---|
| `memory` | — | Exact search, JSON persistence. Genuinely fine to ~100k chunks with NumPy |
| `faiss` | `faiss` | Flat, IVF or HNSW; millions of vectors on one machine |
| `chroma` | `chroma` | Embedded or client/server, native metadata filtering |
| `pinecone` | `pinecone` | Managed, distributed, namespace isolation |

Interface: [`VectorStore`][windlass.interfaces.vectordb.VectorStore]

### `chunker` — text splitting

| Name | Extra | Notes |
|---|---|---|
| `recursive` | — | Splits on the largest natural boundary that fits. The right default for prose |
| `token` | — | Same, measured in model tokens |
| `semantic` | — | Splits at topic shifts detected from sentence embeddings |
| `markdown` | — | Heading-aware; prefixes each chunk with its heading path |
| `code` | — | Function and class boundaries for 13 languages |
| `parent_child` | — | Indexes small chunks that expand to their parent at query time |

Interface: [`Chunker`][windlass.interfaces.chunker.Chunker]

### `retriever` — retrieval strategies

| Name | Extra | Notes |
|---|---|---|
| `bm25` | — | Okapi BM25 over an in-process inverted index |
| `vector` | — | Dense search, optional MMR diversification |
| `hybrid` | — | Fuses any number of retrievers with Reciprocal Rank Fusion |
| `contextual` | — | LLM-enriched chunks at index time; HyDE and multi-query at search time |

Interface: [`Retriever`][windlass.interfaces.retriever.Retriever]

### `loader` — document loading

`text` · `markdown` · `json` · `csv` · `html` · `web` (all dependency-free), plus `pdf` · `docx` · `pptx` · `xlsx` · `youtube` (`loaders`), `image` (`ocr`) and `audio` (`audio`).

Format detection is automatic — you rarely name a loader. Interface: [`Loader`][windlass.interfaces.loader.Loader]

### `preprocessor` — document processing

| Name | Extra | Notes |
|---|---|---|
| `clean` | — | Unicode and whitespace normalisation, boilerplate removal |
| `pii` | — | Detects and redacts personal data before it reaches the index |
| `dedup` | — | Exact and near-duplicate removal via shingle similarity |
| `language` | — | Detects language; can filter on it |
| `metadata` | — | Counts, reading time, keywords |
| `tables` | — | Separates tables from prose |
| `ocr` | `ocr` | Re-reads scanned pages when text extraction came back empty |
| `llm_metadata` | — | Title, summary, topics and entities via a model |

Preprocessors compose with `|`. Interface: [`Preprocessor`][windlass.interfaces.preprocessor.Preprocessor]

### `memory` — conversation and long-term

| Name | Notes |
|---|---|
| `buffer` | The full transcript per thread |
| `window` | The last *n* messages; never splits a tool call from its result |
| `summary` | Summarises overflow with an LLM |
| `vector` | Durable facts recalled by similarity, not recency |
| `composite` | Combines several |

Interface: [`Memory`][windlass.interfaces.memory.Memory]

### `guardrail` — safety

| Name | Extra | Notes |
|---|---|---|
| `rules` | — | PII, prompt injection, leaked secrets, banned terms. Microseconds, no model call |
| `nemo` | `guardrails` | NVIDIA NeMo: topical rails, Colang flows, jailbreak detection |

Guardrails compose with `&`. Interface: [`Guardrail`][windlass.interfaces.guardrail.Guardrail]

### `evaluator` — quality measurement

| Name | Extra | Notes |
|---|---|---|
| `builtin` | — | Lexical metrics (free) plus LLM-judged ones (faithfulness, relevancy, correctness) |
| `ragas` | `evaluation` | Reference-free RAG metrics |
| `deepeval` | `evaluation` | Metrics with pass/fail verdicts and explanations |

Interface: [`Evaluator`][windlass.interfaces.evaluator.Evaluator]

### `tracer` — observability

`console` · `memory` · `null` (dependency-free), plus `langsmith` and `langfuse` (`observability`).

Interface: [`Tracer`][windlass.interfaces.tracer.Tracer]

### `mcp` — Model Context Protocol

| Name | Extra | Notes |
|---|---|---|
| `fastmcp` | `mcp` | stdio, SSE and streamable HTTP |
| `static` | — | In-process, for tests |
| `multi` | — | Aggregates several servers with automatic namespacing |

Interface: [`MCPClient`][windlass.interfaces.mcp.MCPClient]

### `tool`, `cache`, `checkpointer`

`tool` ships no built-ins by design — a generic built-in tool would either be useless or a security footgun. `cache` offers `memory`, `disk` and `null`; `checkpointer` offers `memory` and `sqlite`.

## What every component shares

All fifteen kinds extend [`Component`][windlass.interfaces.base.Component]:

```python
component.name              # identifier used in traces and errors
component.config            # the configuration it was built with
component.describe()        # JSON-safe summary, secrets masked
component.native()          # the underlying SDK object (Level 3)
await component.aclose()    # release sockets, handles, pools
```

`describe()` is worth knowing about. It powers `rag.describe()`, which prints a complete picture of what a deployment is actually running — invaluable when two environments behave differently:

```python
>>> Windlass.rag().retriever("hybrid").describe()
{'llm': {'kind': 'llm', 'name': 'fake', 'class': '...', 'config': {...}},
 'retriever': {'kind': 'retriever', 'name': 'hybrid', 'legs': [...]},
 'vectorstore': {...},
 'top_k': 5, ...}
```

Secrets are masked by key name, so it is safe to log.

## Constructing one directly

Builders are the ergonomic path, but every component is available on its own:

```python
llm       = Windlass.llm("openai", model="gpt-4o")
embedder  = Windlass.embedding("huggingface")
store     = Windlass.vectordb("faiss", dimensions=384)
chunker   = Windlass.chunker("semantic", embedder=embedder)
guardrail = Windlass.guardrail(pii=True)
tracer    = Windlass.tracer("langfuse")
```

Useful when you are embedding one piece of Windlass into a system that already has its own orchestration.

## Discovering at runtime

```python
Windlass.list()                  # {kind: [names]} for everything
Windlass.list("retriever")       # ['bm25', 'contextual', 'hybrid', 'vector']
Windlass.kinds()                 # every kind
Windlass.catalog("chunker")      # full specs: description, aliases, required extra
```

```python
for spec in Windlass.catalog("vectordb"):
    print(f"{spec.name:<12} {spec.extra or 'core':<10} {spec.description}")
```

## Adding your own

See [Writing plugins](../guides/plugins.md). The short version:

```python
from windlass import register, Retriever, ScoredChunk

@register.retriever("my-search")
class MySearch(Retriever):
    """Queries our existing search service."""

    async def aretrieve_chunks(self, query, k, *, filters=None, **kwargs):
        hits = await search_service.query(query, limit=k)
        return [ScoredChunk(chunk=to_chunk(h), score=h.score) for h in hits]
```

```python
rag = Windlass.rag().retriever("my-search")
```

From that point it is indistinguishable from a built-in: it composes into hybrid retrieval, it is traced, it is configurable, and it shows up in `windlass list`.
