# Changelog

All notable changes to Windlass are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Before 1.0, minor versions may change the public API. Every break gets an entry
here and a note in the [migration guide](docs/guides/migration.md).

## [Unreleased]

### Changed

- **The project is now called Windlass** (was "Harness", distributed as
  `harness-ai`). The old name collided with an existing PyPI distribution and
  with a well-known CI/CD vendor, neither of which was survivable for a package
  people are meant to `import`. Everything renamed consistently: the
  distribution is `windlass`, the import is `windlass`, the facade is
  `Windlass`, the CLI is `windlass`, environment variables are `WINDLASS_*`,
  the config file is `windlass.toml`, and plugin entry-point groups are
  `windlass.plugins` / `windlass.<kind>`.

- **Every optional dependency now has an upper version bound.** Unbounded pins
  are how two of the defects below shipped: `langgraph>=0.2.28` resolved to
  `1.2.9` and `langfuse>=2.50` resolved to `4.14.1`, each across a major
  version, into adapters written against the older API.

### Added

- **`hf_inference` embeddings — the HuggingFace Inference API as a first-class
  provider.** The existing `huggingface` provider runs sentence-transformers
  locally, which means torch, a model download and the RAM to hold it. This one
  calls the hosted endpoint and needs **no optional dependency at all**, since
  `httpx` is already core. BGE and E5 instruction prefixes are declared through
  the interface's hooks, so the query/document asymmetry those models need is
  applied on the correct side rather than being a silent retrieval-quality bug.

- `huggingface_api_key` setting, reading `HUGGINGFACE_API_KEY` or `HF_TOKEN`,
  and reported by `windlass doctor`.

- **`multi` tracer — fan one trace out to several backends.** Teams rarely have
  exactly one place traces should land: a platform team standardises on
  LangSmith while a product team already reads Langfuse, and during a migration
  you want both. `observe("multi", backends=["langfuse", "langsmith"])` sends
  every span to each. A backend that raises is logged once and skipped for the
  rest of the run, never propagated — one misconfigured exporter must not take
  down the others or the application.

- **`web` extras group** — `beautifulsoup4` and `lxml` only. Scraping a page
  should not also install a PDF parser, an Excel reader and a YouTube client,
  which is what the broader `loaders` group pulls in.

- **`ProviderError` now takes `status_code`.** Retry classification reads it, so
  an adapter that omitted it silently turned a transient 429 or 503 into a hard
  failure of the whole run. Previously the only way to set it was to assign the
  attribute after construction — undiscoverable from the signature, and invisible
  to a type checker.

### Fixed

- **The LangGraph runtime never ran.** `.graph()` raised `NameError: name
  'Annotated' is not defined` before a single node executed. The module uses
  `from __future__ import annotations`, so the state `TypedDict`'s annotations
  are strings, and LangGraph resolves them with `get_type_hints` against the
  *defining module's* globals — but `Annotated` was imported inside the builder
  method. The state class now lives at module scope. A headline feature had
  never once worked with `langgraph` installed, and the suite was green
  throughout, because no test ever built a graph.

- **Every dependency-container binding was silently ignored.** The builders
  passed the configured *default name* whenever the caller had not named a
  component, and `Container.component()` only consults its bindings when the
  spec is `None`, so that branch was unreachable from the builders. This broke
  `Windlass.container().bind_instance(...)` — which the public API documents as
  the place for application-wide wiring — and `capture_spans()`, the shipped
  test helper, which collected nothing. Resolution order is now explicit:
  builder call, then container binding, then configured default.

- **`resume()` dropped the approved tool call from `steps`.** `aresume`
  delegates to `arun`, which starts a fresh step list, so the one action a human
  explicitly authorised survived only in `messages`. Any audit trail reading
  `steps` showed an effect with no step accounting for it.

- **A tracer could raise on the way out.** `_bounded_flush` starts a daemon
  thread, and Python refuses new threads once interpreter shutdown has begun —
  exactly when an `atexit` flush runs. Hosted tracers printed a traceback on
  every clean exit, contradicting the adapter's own guarantee never to break the
  application.

- **`LANGFUSE_BASE_URL` is now read.** The vendor SDK honours it, Windlass did
  not, so a project on a non-default region silently disagreed with itself:
  `windlass config` reported the default cloud host while the vendor client
  quietly used the right one. They only agreed by luck.

---

Six defects found by building a full third-party application against the
framework (multi-agent claims adjudication over Groq, HuggingFace Inference,
Pinecone, Langfuse and LangSmith). All six were invisible to the existing
suite, because each needs a real socket, a second installed distribution, or a
current vendor SDK to reproduce.

### Added

- **Agents recover from an unparseable tool call instead of dying.** A model
  that emits a tool call the provider cannot parse now raises the new
  `MalformedToolCallError`, which `AgentRuntime` catches, feeds back to the
  model as a correction, and retries — the same treatment a hallucinated tool
  name already got. Groq's `tool_use_failed` is detected and mapped to it.

  The common cause is a model trying to express a data dependency the protocol
  cannot represent, nesting one call inside another's arguments
  (`settle(amount=<function=compute>{...}</function>)`) because the value it
  needs does not exist until the first call has run. Observed with Llama 3.3 on
  Groq; it previously ended the run with a bare `ProviderError`.

  Recoveries are bounded by `AgentBuilder.tool_call_retries()` (default 2) and
  each one consumes an iteration, so a model that never recovers still
  terminates. `tool_call_retries(0)` restores the previous behaviour.

- `tests/test_observability.py` — the hosted tracers had no tests at all.
  Twenty-six cover version detection, span-kind translation, nesting, usage
  reporting and flush deadlines against fakes shaped like each Langfuse SDK
  generation.

### Fixed

- **Langfuse tracing exported nothing on langfuse 3.x/4.x, then hung the
  process.** The adapter targeted the v2 `trace()`/`span()`/`generation()`
  surface, which no longer exists; every call raised `AttributeError` into the
  adapter's own `except Exception`, so `doctor` reported healthy while zero
  observations reached Langfuse. Separately, `flush()` blocked forever inside
  Langfuse's `queue.join()` when a worker had stopped with items outstanding —
  contradicting the documented guarantee that a tracer can never break the
  application, and wedging any process that flushed. The adapter now detects
  the installed generation and uses `start_observation(as_type=...)` on v3/v4
  — whose observation types map almost one-to-one onto Windlass span kinds —
  falling back to the v2 surface, and **raising at construction** when neither
  is present rather than silently discarding every span. Every vendor `flush()`
  now runs under a deadline, LangSmith included.

- **Blocking calls no longer close the event loop they run on.** `run_sync`
  used `asyncio.run` when no loop was active, creating and destroying a loop
  per call. Every provider that keeps a long-lived `httpx.AsyncClient` — Ollama
  directly, and the OpenAI, Anthropic, Groq and Gemini SDKs underneath — pools
  keep-alive sockets on the loop that opened them, so the *second* blocking
  call raised `RuntimeError: Event loop is closed` from deep inside httpx. All
  calls now share the background loop, which is the same reasoning `iter_sync`
  already documented. Mock transports could never catch this: they open no
  sockets, so loop identity is now asserted directly.
- **A cache passed to an `Embedder` constructor is no longer discarded.**
  `cache or NullCache()` looked equivalent to a `None` check and was not: every
  `Cache` implements `__len__`, so a freshly constructed (empty) cache is falsy.
  `Windlass.embedding(..., cache=MemoryCache())` silently never cached. Only
  `set_cache()` worked, which is the path the existing test happened to use.
- **A hand-constructed `Registry` is isolated again.** Discovery of
  entry-point plugins ran on first lookup for *any* registry, so installing any
  third-party Windlass plugin changed the contents of every registry in the
  process — including the clean ones tests build, which made the suite's result
  depend on what else was in the environment. Discovery is now opt-in
  (`Registry(discover=True)`); the process-wide `REGISTRY` opts in, and
  `load_plugins()` remains available on demand for any registry.
- **`RAGBuilder.min_score()` works with `.retriever(instance)`.** The threshold
  was passed as construction config, which an already-built retriever cannot
  accept, so the pairing failed at build time with an error that never
  mentioned `min_score`. It is now applied to the live object, matching how
  `AgentBuilder` already handles a pre-built MCP client.
- **`PineconeVectorStore.clear()` succeeds on a namespace that does not exist.**
  Pinecone creates namespaces lazily and returns 404 for a delete-all against
  one never written to, so teardown failed precisely when there was nothing to
  tear down. Detection matches on status and message rather than exception
  class, which moved between SDK generations.

## [0.1.1] — 2026-07-29

Documentation and packaging only. No functional change — upgrading from 0.1.0
changes nothing about how the framework behaves.

### Fixed

- **The API reference rendered a broken link on the `multi` tracer page.** The
  module's doctest output was written as `[1, 1]`, which is Markdown
  reference-link syntax, so mkdocstrings tried to resolve it as a cross-reference
  and `mkdocs build --strict` aborted. The example now yields `(1, 1)` and
  asserts exactly the same thing.

- **Corrected inaccurate claims throughout the documentation**, each verified
  against the source: the core install is four packages rather than two;
  `agent.draw()` and `rag.aingest_documents()` are runtime methods, not builder
  methods, and the examples now call them through `build()`;
  `rag.chunker(strategy=...)` silently produced a recursive chunker because
  `spec` is positional-only; the `MissingDependencyError` sample quoted the
  wrong extras group; test counts and component counts were stale; and `cache`
  and `checkpointer` were described as extending `Component`, which they do not.

- Import ordering in five example scripts, so `ruff check` passes cleanly.

### Changed

- Project URLs now point at the real repository and the live documentation site
  at <https://kukilbharadwaj.github.io/WINDLASS/>.

## [0.1.0] — 2026-07-26

First release.

### Core

- Component registry with lazy, dotted-path registration, aliases and
  case-insensitive lookup. Importing a provider is deferred until it is used.
- Plugin discovery via the `windlass.plugins` and `windlass.<kind>` entry-point
  groups. A failing plugin logs a warning and is skipped unless
  `strict_plugins` is set.
- Hierarchical dependency-injection container. `component()` accepts a registry
  name, a live instance or a factory.
- Settings from defaults, a config file (TOML/JSON/YAML), the environment and
  explicit arguments — in that precedence order. Conventional provider
  variables such as `OPENAI_API_KEY` are read as-is.
- One exception hierarchy rooted at `WindlassError`, each error carrying an
  actionable `hint` and structured `context`.
- Optional dependencies loaded through a single choke point, so a missing extra
  produces a `MissingDependencyError` naming the exact `pip install` command.
- Async/sync bridge that detects a running event loop and dispatches to a
  background loop thread, making the blocking API safe in Jupyter, FastAPI and
  LangServe.
- Bounded-concurrency helpers, exponential backoff with jitter on transient
  failures only, memory and disk caches, and a context-aware logger.

### Interfaces

- Thirteen component contracts: `LLM`, `Embedder`, `Loader`, `Preprocessor`,
  `Chunker`, `Retriever`, `VectorStore`, `Memory`, `Guardrail`, `Evaluator`,
  `Tracer`, `Tool`, `MCPClient`.
- All async-first, with blocking variants derived from the same implementation.
- `native()` on every component, returning the wrapped SDK object.

### Providers

- **LLMs** — OpenAI (and OpenAI-compatible gateways), Anthropic, Gemini, Groq,
  Ollama, plus dependency-free `fake` and `echo` providers.
- **Embeddings** — HuggingFace `sentence-transformers` (with automatic E5/BGE/
  Nomic instruction prefixes), OpenAI, and a dependency-free hashed n-gram model.
- **Vector stores** — in-memory (exact search, JSON persistence), FAISS
  (flat/IVF/HNSW), ChromaDB, Pinecone.
- **Loaders** — PDF, DOCX, PPTX, XLSX, CSV, Markdown, HTML, JSON/JSONL, images
  (OCR), audio (Whisper), web pages, YouTube transcripts, with automatic format
  detection.
- **Preprocessors** — cleaning, PII detection and redaction, deduplication,
  language detection, OCR fallback, table extraction, LLM metadata extraction.
- **Chunkers** — recursive, token, semantic, Markdown (heading-path aware), code
  (13 languages), parent-child.
- **Retrievers** — BM25, dense vector with MMR, hybrid with Reciprocal Rank
  Fusion, contextual enrichment with HyDE and multi-query expansion.
- **Memory** — buffer, sliding window, LLM summarising, vector long-term,
  composite.
- **Guardrails** — rule-based (PII, prompt injection, leaked secrets, banned
  terms) and NVIDIA NeMo Guardrails.
- **Evaluation** — built-in lexical and LLM-judged metrics, RAGAS, DeepEval.
- **Observability** — console, in-memory, LangSmith, Langfuse.
- **MCP** — FastMCP over stdio/SSE/HTTP, an in-process client for tests, and a
  multi-server aggregator with automatic namespacing.

### RAG

- `Windlass.rag()` fluent builder with automatic cross-component wiring: the
  semantic chunker receives the embedder, the vector retriever receives the
  embedder and store, hybrid retrieval builds its own BM25 leg, FAISS and
  Pinecone receive the embedding dimensionality, parent-child chunking wires
  parent expansion.
- Idempotent ingestion — chunk ids are content hashes, so re-ingesting
  unchanged content upserts rather than duplicating.
- Metadata filtering with Mongo-style operators, pushed down to stores that
  support it and applied client-side by those that do not.
- Context assembly against a token budget, dropping the lowest-ranked chunks
  first so the best context always survives.
- `strict()` and `min_score()` for refusing to answer without relevant context.
- Streaming, persistence, and evaluation from the pipeline object.

### Agents

- `Windlass.agent()` fluent builder.
- A built-in reason/act runtime with no dependencies, plus a LangGraph runtime
  exposing a real `StateGraph` through `native_graph()`.
- Automatic JSON-schema generation from type hints and Google-style docstrings,
  rendered per provider dialect.
- Parallel tool execution, per-tool timeouts, and tool failures reported to the
  model rather than raised.
- Conversation and long-term memory keyed by `thread_id`.
- Checkpointing (in-memory and SQLite) enabling resume and time travel.
- Human-in-the-loop approval with reject-with-feedback and edited arguments.
- Multi-agent supervision with `run`, `broadcast` and `pipeline` coordination.

### Tooling

- `windlass` CLI: `doctor`, `info`, `list`, `config`, `ask`, `chat`.
- `windlass.testing` — scripted models, offline pipeline and agent factories,
  registry isolation, span capture, recording tools.
- Over 650 tests, running offline in under a minute with no API keys —
  including every docstring example, executed as part of the suite so a
  documented example that does not run fails the build.
- `mypy` clean, `ruff` clean, `black` formatted.
- MkDocs documentation site with a generated API reference.
- Eight runnable examples, all of which run on the core install alone.

[Unreleased]: https://github.com/Kukilbharadwaj/WINDLASS/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Kukilbharadwaj/WINDLASS/releases/tag/v0.1.0
