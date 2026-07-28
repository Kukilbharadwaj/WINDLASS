# Performance

Where the time and the money actually go, and what to do about it.

## Measure first

```python
rag = Windlass.rag().observe("console")
answer = rag.ask("...")
```

```
▸ rag.ask  (chain)
  ⌕ retrieve  (retriever)
    · 12ms
  ◆ generate  (llm)
    · 840ms  312 tok
  · 856ms
```

In almost every RAG system, generation dominates and retrieval is noise. Optimising retrieval from 12ms to 6ms is not the win; it is a distraction from the 840ms next to it.

```python
answer.latency_ms                        # end to end
answer.metadata["retrieval_ms"]          # retrieval only
answer.usage.total_tokens
```

## Import cost

`import windlass` costs less than importing `pydantic-settings`, which it depends on. That is deliberate: providers are registered by dotted path and imported only when used.

```bash
python -X importtime -c "import windlass" 2>&1 | tail -5
```

If you see torch or chromadb in that output, something is importing eagerly — that is a bug worth reporting.

## Ingestion

Ingestion is the slow part, and it is where batching pays.

### Batch size

```python
configure(batch_size=128)                              # global
Windlass.rag().embedding("huggingface", batch_size=128) # per component
```

On a GPU, raise it until memory complains. On CPU, 32–64 is usually the sweet spot. For API-backed embeddings, the ceiling is the provider's request limit.

### Concurrency

```python
configure(max_concurrency=16)
```

Batches run concurrently up to this ceiling. Raising it helps until you hit the provider's rate limit, at which point it hurts — every 429 costs a backoff.

### Cache embeddings

```python
configure(cache={"enabled": True, "backend": "disk", "ttl": 604_800})
```

Embeddings are deterministic. On a corpus that changes 5% a week, this turns re-ingestion from expensive into nearly free.

### Idempotence is already handled

Chunk ids are content hashes, so re-ingesting unchanged content upserts rather than duplicating. You do not need to track what you have already indexed.

### Concurrency is not just for the network

Blocking parsers (pypdf, python-docx, openpyxl) run on the thread pool, so a folder of 500 PDFs uses the whole machine instead of one core.

### Streaming ingestion for very large corpora

```python
loader = Windlass.loader()
async for document in loader.astream_load("./huge-corpus"):
    await rag.aingest_documents([document])
```

Keeps peak memory flat regardless of corpus size.

## Retrieval

### Pick the right store

| Store | Good to | Search | Notes |
|---|---|---|---|
| `memory` | ~100k chunks | exact | No recall loss. Genuinely production-viable at this scale |
| `faiss` (flat) | ~1M | exact | Fast, no training needed |
| `faiss` (ivf) | 10M+ | approximate | Needs `39 × nlist` vectors to train |
| `faiss` (hnsw) | 10M+ | approximate | Fast, higher memory, no training |
| `chroma` | ~1M | approximate | Persistent, native filtering, no server needed |
| `pinecone` | unbounded | approximate | Network latency per query; eventually consistent |

The in-memory store is `O(n)` per search, but with NumPy the whole matrix is scored in one vectorised operation — 100k chunks lands in the low tens of milliseconds. Without NumPy, expect roughly two orders of magnitude slower, so install `windlass[rag]` (which brings NumPy) for anything real.

### Filters make search faster

A metadata filter narrows the candidate set *before* scoring:

```python
rag.ask(question, filters={"tenant": tenant, "year": {"$gte": 2024}})
```

### Tune FAISS directly when you need to

```python
index = rag.native_store()
index.nprobe = 32          # more clusters scanned: better recall, slower
```

### Hybrid costs its slower leg, not the sum

Legs run concurrently. If BM25 answers in 3ms and dense in 15ms, hybrid is ~15ms.

### Over-fetch honestly

With a reranker or MMR, Windlass fetches `fetch_k` candidates then narrows to `top_k`. Raising `fetch_k` improves recall at the cost of one larger rerank call:

```python
rag.retriever("vector", diversity=0.3, fetch_k=50)
```

## Generation

This is where the time is.

### Stream

```python
for token in rag.stream_ask(question):
    print(token, end="", flush=True)
```

Time to *first* token is what users experience. Streaming does not make the request faster; it makes the wait tolerable.

### Cut the context

```python
rag.max_context_tokens(6000)
rag.top_k(5)
```

Prompt tokens are the largest controllable cost in most RAG systems. `top_k=20` over 2000-character chunks is 40k characters on every request. Measure whether it actually improves answers.

### Right-size the model

Research, classification, metadata extraction and summarisation rarely need your best model:

```python
researcher = Windlass.agent().llm("gpt-4o-mini")
writer     = Windlass.agent().llm("claude-sonnet-4-5")
```

### Batch offline work

```python
answers = await rag.llm.abatch(prompts, concurrency=16)
```

Bounded on purpose — unbounded fan-out over a few hundred prompts is the fastest way to get rate limited.

## Agents

### Parallel tools

On by default. Models routinely request three independent lookups in one turn; running them serially triples the user's wait for no reason.

```python
agent.parallel_tools(False)   # only when tools share mutable state
```

### Every iteration is a round trip

```python
agent.max_iterations(10)
```

An agent taking eight steps has made eight model calls. If the average step count is creeping up, a tool description has stopped working — that is a quality problem showing up as a latency problem.

### Trim memory

```python
agent.memory("window", window=20)       # not "buffer"
agent.memory("summary")                 # for long conversations
```

Buffer memory grows without bound, and every message is prompt tokens on every subsequent call.

## Concurrency and threading

### The sync bridge

`rag.ask(...)` inside an async web handler works — Windlass detects the running loop and dispatches to a background loop thread. But it costs a thread hop. In async code, `await rag.aask(...)` directly.

### Thread safety

The registry, container, in-memory store, BM25 index, caches, memory backends and checkpointers are all guarded by locks. Components are safe to share between threads and across concurrent requests.

### Workers versus concurrency

```bash
uvicorn app:app --workers 4
```

Four workers means four copies of any local embedding model in memory. If that is too much, use an API-backed embedding provider or drop to one worker with higher async concurrency.

## Memory footprint

| What | Roughly |
|---|---|
| `import windlass` | a few MB |
| `bge-small-en-v1.5` | ~130 MB |
| `bge-m3` | ~2.2 GB |
| 100k chunks in the memory store, 384-dim | ~200 MB vectors plus the text |
| FAISS flat, 1M × 384 | ~1.5 GB |

Bake local models into your image so the first request does not pay for a download:

```dockerfile
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/bge-small-en-v1.5')"
```

## A checklist

- [ ] Traced, and you know which stage dominates
- [ ] Embedding cache on
- [ ] `batch_size` and `max_concurrency` tuned to the provider
- [ ] Context budget set deliberately, not by default
- [ ] Streaming on any interactive path
- [ ] The right model for each job, not the best model everywhere
- [ ] Window or summary memory, never unbounded buffer
- [ ] Store sized for the corpus, not for the demo
- [ ] Local models baked into the image
- [ ] Tokens per request tracked as a metric
