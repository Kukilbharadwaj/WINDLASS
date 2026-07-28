# Troubleshooting

Start here:

```bash
windlass doctor
```

It reports which extras are installed, which credentials are configured, what the defaults resolve to, and runs a complete offline pipeline as a smoke test. Most problems are visible in that output.

---

## Installation

### `MissingDependencyError`

```
MissingDependencyError: The Pinecone vector store is not installed.

Hint: Run:

    pip install "windlass[pinecone]"
```

Working as designed — run the command it printed. Check what you have with `windlass list`; uninstalled components are marked.

### A raw `ImportError` from an optional package

That is a bug. Windlass translates optional-dependency import failures at a single choke point (`windlass.core.lazy`). Please report it with the traceback.

### `windlass: command not found`

The console script is installed into the environment's `Scripts`/`bin`. Either activate the environment, or run `python -m windlass.cli`.

### `import windlass` is slow

Measure it:

```bash
python -X importtime -c "import windlass" 2>&1 | tail -5
```

Windlass's own overhead is a few milliseconds; the floor is `pydantic-settings`, which costs more on its own than the whole framework. If you see torch or chromadb in that output, something is importing eagerly — that is a bug worth reporting.

---

## Configuration

### `ComponentNotFoundError`

```
ComponentNotFoundError: No chunker named 'sematic' is registered.

Hint: Available chunkers: code, markdown, parent_child, recursive, semantic, token
```

A typo, or a plugin that did not load. Check with `Windlass.list("chunker")` and `Windlass.plugins(strict=True)`.

### `AuthenticationError` despite the key being set

Check what Windlass actually sees:

```bash
windlass config | grep -i key
```

Common causes: the variable is set in a different shell, a `.env` file in the working directory is overriding it, or the key belongs to an account without access to that model.

### `configure()` seems not to take effect

`configure()` merges onto current settings and returns the new object. Read them back with `settings()`, and remember explicit builder arguments always win:

```python
configure(temperature=0.7)
Windlass.rag().llm("openai", temperature=0.0)   # this pipeline uses 0.0
```

### Settings leak between tests

They are a process-wide singleton. Reset them:

```python
@pytest.fixture(autouse=True)
def clean_settings():
    reset_settings()
    yield
    reset_settings()
```

---

## Retrieval quality

### The answer is wrong or made up

Look at what was retrieved first — this is the answer to most quality complaints:

```python
answer = rag.ask("...")
for hit in answer.contexts:
    print(f"{hit.score:.3f}  {hit.retriever:<12}  {hit.chunk.content[:80]}")
```

If the right chunk is not there, it is a **retrieval** problem and no prompt change will fix it.

Then, in order:

1. `rag.strict()` — refuse rather than invent when nothing is retrieved.
2. `rag.retriever("hybrid")` — especially if the query contains an identifier or product name.
3. Raise `top_k`.
4. Change the chunker (`markdown` for docs, `parent_child` for mixed content).
5. Upgrade the embedding model and re-ingest.

### Retrieval returns nothing

```python
print(rag.count())    # is anything indexed at all?
```

If zero, ingestion failed silently — see below. If not zero, check that a filter is not excluding everything:

```python
rag.search("query", k=10)                            # unfiltered
rag.search("query", k=10, filters={"tenant": "x"})   # filtered
```

### Retrieval got worse after changing the embedding model

Expected. Vectors from two different models are not comparable. Clear the store and re-ingest:

```python
rag.build().vectorstore.clear()
rag.ingest("./documents")
```

### The same passage appears five times in the results

Your corpus has duplicates, or your chunks overlap heavily.

```python
rag.preprocessor("dedup", threshold=0.9)     # at ingestion
rag.retriever("vector", diversity=0.3)       # MMR at query time
```

### Exact terms are not found

Dense retrieval misses identifiers, codes and SKUs. That is what BM25 is for:

```python
rag.retriever("hybrid")
```

---

## Ingestion

### `IngestionError: No documents were produced`

The path is wrong, the directory is empty, or no loader claims those extensions.

```python
from windlass.rag.loading import extension_map
print(sorted(extension_map()))               # what this install can read
```

Most formats need `pip install "windlass[loaders]"`.

### A PDF produced no text

`pypdf` extracts *embedded* text. A scanned PDF contains images, not text.

```bash
pip install "windlass[ocr]"
```

```python
rag.preprocessor("ocr")     # re-reads pages whose text extraction came back empty
```

PDF page rendering additionally needs `pdf2image` and Poppler.

### Everything was dropped during preprocessing

```
IngestionError: Every document was dropped during preprocessing.
```

A filter is too strict — usually `min_length` on the cleaner, or a language filter:

```python
rag.preprocessor("clean", min_length=10)    # default is 20
```

### Ingestion is slow

```python
configure(batch_size=128, max_concurrency=16)
configure(cache={"enabled": True, "backend": "disk"})
```

See [Performance](performance.md).

### One corrupt file aborts the run

It should not — the default is `on_error="skip"`. If you set `raise`, one bad file will stop everything:

```python
rag.loader("pdf", on_error="skip")
```

---

## Agents

### `MaxIterationsExceeded`

The agent looped without finishing. Raising the ceiling treats the symptom; the cause is usually one of:

- **A tool description the model misreads.** It keeps calling the wrong tool. Rewrite the description as a prompt.
- **A tool returning something unusable.** A forty-field object teaches the model to guess. Return three fields.
- **A task that needs decomposition.** Split into specialists with a supervisor.

Look at the trace:

```python
for step in response.steps:
    print(step.index, step.thought[:80], [c.name for c in step.tool_calls])
```

### The agent never calls a tool

- Does the model support tool calling? Windlass raises at build time if not.
- Is the description clear enough for the model to know when to use it?
- Is the tool bound? `agent.build().tools.names()`

### `ToolNotFoundError`, or the model invents tool names

It hallucinated a name. Windlass returns the error to the model with the list of real tools, so it usually recovers on the next turn. If it happens repeatedly, your tool names are too similar or too generic.

### A tool times out

```python
@tool(timeout=120.0)
def slow_thing(): ...
```

The default is 60 seconds. A hung tool fails the call, not the run.

### `AgentInterrupt` when you did not expect one

A tool declares `requires_approval=True`, or `.human_in_the_loop()` is on. Inspect and resume:

```python
agent.pending_approvals(thread_id)
agent.resume(thread_id, approved=True)
```

### `resume()` says there is no checkpoint

Resuming requires a checkpointer, and `memory` does not survive a process restart:

```python
agent.checkpoint("sqlite", path="./state.db")
```

---

## Streaming

### Streaming yields one chunk then stops

If this happens with a **custom** sync bridge, the cause is `asyncio.run` per item: it calls `loop.shutdown_asyncgens()` on exit, which finalises the suspended generator. Use `windlass.core.concurrency.iter_sync`, which pins the whole iteration to one loop.

Inside Windlass this is already handled — if you see it from a Windlass method, please report it.

### `RuntimeError: asyncio.run() cannot be called from a running event loop`

Not from Windlass — `run_sync` detects a running loop and dispatches to a background thread. If you see it, something else in your stack is calling `asyncio.run`. In async code, prefer the `a*` methods directly.

---

## Guardrails

### Legitimate content is being blocked

```python
rag.guardrails(on_violation="warn")     # log without blocking, to calibrate
```

Then narrow the policy:

```python
rag.guardrails(pii=True, injection=False, secrets=True)
```

### PII redaction is too aggressive

```python
rag.preprocessor("pii", kinds=["email", "credit_card", "ssn"])
```

The phone pattern is the usual over-matcher — it is deliberately broad, since a missed phone number is worse than a false positive.

### PII is not detected

Regex detection is precise for *structured* identifiers and blind to unstructured ones — it will not find "Ada Lovelace" as a name.

```bash
pip install "windlass[pii]"
```

```python
rag.preprocessor("pii", use_presidio=True)
```

---

## Providers

### Rate limits

```python
configure(retry={"attempts": 5, "max_delay": 60})
configure(max_concurrency=4)
```

Windlass honours a provider's advertised `Retry-After` header over its own backoff. Do not add another retry loop on top; you will multiply the two.

### `ProviderTimeoutError`

```python
configure(request_timeout=180.0)
```

Local models loading for the first time routinely need more than 60 seconds. `keep_alive` keeps Ollama models resident between requests.

### `Cannot reach the Ollama daemon`

```bash
ollama serve
ollama pull llama3.2
```

### A model rejects the tool schema

Some models are strict about JSON Schema. Simplify the annotation — deeply nested Pydantic models are the usual culprit — and check what is actually being sent:

```python
print(json.dumps(my_tool.schema(), indent=2))
```

---

## Getting help

Include this in a bug report:

```bash
windlass doctor
windlass config          # secrets are masked
python -c "import windlass, sys; print(windlass.__version__, sys.version)"
```

Plus the full traceback. Windlass errors carry structured context worth including:

```python
except WindlassError as exc:
    print(exc.message, exc.hint, exc.context)
```
