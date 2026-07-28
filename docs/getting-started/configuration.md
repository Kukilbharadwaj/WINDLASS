# Configuration

Windlass reads configuration from four places. Later sources override earlier ones:

1. **Built-in defaults** — chosen so the framework works offline with nothing installed.
2. **A config file** — `windlass.toml`, `windlass.json` or `windlass.yaml` in the working directory, or wherever `WINDLASS_CONFIG` points.
3. **Environment variables** — both `WINDLASS_*` and the conventional provider names.
4. **Explicit arguments** — anything you pass to a builder or component.

## In code

```python
from windlass import configure, settings

configure(
    default_llm="anthropic",
    default_embedding="huggingface",
    temperature=0.0,
    max_concurrency=16,
    request_timeout=120.0,
    retry={"attempts": 5, "max_delay": 60},
    cache={"enabled": True, "backend": "disk"},
)

print(settings().default_llm)   # 'anthropic'
```

`configure()` merges — partial updates leave everything else alone, including credentials already read from the environment.

## Environment variables

Every setting has a `WINDLASS_`-prefixed variable:

```bash
export WINDLASS_DEFAULT_LLM=openai
export WINDLASS_TEMPERATURE=0.0
export WINDLASS_MAX_CONCURRENCY=16
export WINDLASS_RETRY__ATTEMPTS=5      # nested settings use __
export WINDLASS_CACHE__ENABLED=true
```

Credentials additionally accept the vendor's conventional name, so an existing setup needs no changes:

| Setting | Accepted variables |
|---|---|
| `openai_api_key` | `WINDLASS_OPENAI_API_KEY`, `OPENAI_API_KEY` |
| `anthropic_api_key` | `WINDLASS_ANTHROPIC_API_KEY`, `ANTHROPIC_API_KEY` |
| `google_api_key` | `WINDLASS_GOOGLE_API_KEY`, `GOOGLE_API_KEY`, `GEMINI_API_KEY` |
| `groq_api_key` | `WINDLASS_GROQ_API_KEY`, `GROQ_API_KEY` |
| `huggingface_api_key` | `WINDLASS_HUGGINGFACE_API_KEY`, `HUGGINGFACE_API_KEY`, `HF_TOKEN` |
| `pinecone_api_key` | `WINDLASS_PINECONE_API_KEY`, `PINECONE_API_KEY` |
| `langsmith_api_key` | `WINDLASS_LANGSMITH_API_KEY`, `LANGCHAIN_API_KEY`, `LANGSMITH_API_KEY` |
| `langfuse_public_key` | `WINDLASS_LANGFUSE_PUBLIC_KEY`, `LANGFUSE_PUBLIC_KEY` |
| `langfuse_secret_key` | `WINDLASS_LANGFUSE_SECRET_KEY`, `LANGFUSE_SECRET_KEY` |
| `langfuse_host` | `WINDLASS_LANGFUSE_HOST`, `LANGFUSE_HOST`, `LANGFUSE_BASE_URL` |
| `openai_base_url` | `WINDLASS_OPENAI_BASE_URL`, `OPENAI_BASE_URL` |
| `ollama_base_url` | `WINDLASS_OLLAMA_BASE_URL`, `OLLAMA_BASE_URL` |

A `.env` file in the working directory is loaded automatically.

## Config files

=== "windlass.toml"

    ```toml
    default_llm = "openai"
    default_embedding = "huggingface"
    default_vectordb = "faiss"
    temperature = 0.1
    max_concurrency = 16

    [retry]
    attempts = 5
    max_delay = 60.0

    [cache]
    enabled = true
    backend = "disk"
    ttl = 86400
    ```

=== "windlass.json"

    ```json
    {
      "windlass": {
        "default_llm": "openai",
        "temperature": 0.1,
        "retry": { "attempts": 5 }
      }
    }
    ```

=== "windlass.yaml"

    ```yaml
    default_llm: openai
    temperature: 0.1
    retry:
      attempts: 5
    ```

    YAML requires `pip install pyyaml`; TOML and JSON use the standard library.

A top-level `windlass` table is unwrapped, so the settings can live inside a shared config file.

Point at an explicit path with `WINDLASS_CONFIG=/etc/myapp/windlass.toml`.

## Every setting

### Component defaults

| Setting | Default | Meaning |
|---|---|---|
| `default_llm` | `fake` | Provider used when a builder does not name one |
| `default_model` | `""` | Model id when a builder does not name one |
| `default_embedding` | `hash` | Embedding provider for RAG builders |
| `default_vectordb` | `memory` | Vector store for RAG builders |
| `default_chunker` | `recursive` | Chunking strategy |
| `default_retriever` | `vector` | Retrieval strategy |

The defaults are deliberately offline-capable. Set them once in production and every builder inherits them.

### Generation

| Setting | Default | Meaning |
|---|---|---|
| `temperature` | `0.2` | Sampling temperature |
| `max_tokens` | `None` | Completion ceiling; `None` defers to the provider |
| `request_timeout` | `60.0` | Per-request timeout in seconds |
| `max_concurrency` | `8` | Ceiling on in-flight provider calls for batch work |
| `batch_size` | `64` | Default batch size for embedding and indexing |

### Retry

| Setting | Default | Meaning |
|---|---|---|
| `retry.attempts` | `3` | Total tries including the first; `1` disables retrying |
| `retry.initial_delay` | `0.5` | Seconds before the first retry |
| `retry.max_delay` | `30.0` | Backoff ceiling |
| `retry.multiplier` | `2.0` | Backoff growth factor |
| `retry.jitter` | `0.2` | Random fraction added to each delay |

Only *transient* failures are retried — rate limits, timeouts, 5xx, connection resets. A 401 or a schema error fails immediately, because the next attempt would fail identically.

A provider's advertised `Retry-After` header wins over the computed backoff.

### Cache

| Setting | Default | Meaning |
|---|---|---|
| `cache.enabled` | `false` | Master switch |
| `cache.backend` | `memory` | `memory` (LRU per process) or `disk` (needs the `cache` extra) |
| `cache.ttl` | `3600.0` | Entry lifetime in seconds; `None` for forever |
| `cache.max_size` | `1024` | Entry ceiling for the memory backend |
| `cache.directory` | `~/.cache/windlass` | Where the disk backend stores data |

Caching is off by default: caching a non-deterministic model is a decision, not a default. Embeddings are always safe to cache — they are deterministic — and doing so makes re-ingesting a mostly-unchanged corpus nearly free.

### Behaviour

| Setting | Default | Meaning |
|---|---|---|
| `project` | `windlass` | Project/session name recorded by tracers |
| `strict_plugins` | `false` | Fail loudly when a third-party plugin cannot load |
| `telemetry` | `false` | Reserved. Windlass ships no telemetry backend |

## Per-component configuration

Global settings are defaults. Anything passed to a builder wins:

```python
configure(temperature=0.7)                     # global

rag = Windlass.rag().llm("openai", temperature=0.0)   # this pipeline is deterministic
agent = Windlass.agent().llm("openai")                # this one uses 0.7
```

And per-call overrides beat both:

```python
rag.ask("Be creative", temperature=1.0)
```

## Inspecting the effective configuration

```bash
windlass config          # human readable, secrets masked
windlass config --json   # machine readable
```

```python
from windlass import settings

print(settings().masked())   # every secret replaced with '***' — safe to log
```

!!! warning "Never log `settings()` directly"
    `settings()` holds live credentials. `settings().masked()` is the safe form, and it is what the CLI prints.

## Application-wide wiring

For anything that is not a scalar setting, bind it into the root container. Every builder created afterwards inherits it:

```python
from windlass import Windlass

Windlass.container().bind_instance("tracer", my_tracer)
Windlass.container().bind("llm", lambda: my_configured_model)
```

This is the seam for integrating Windlass into an existing dependency-injection setup. See [Architecture](../concepts/architecture.md#the-container).

## Testing

Settings are a process-wide singleton, so reset them between tests:

```python
import pytest
from windlass import reset_settings

@pytest.fixture(autouse=True)
def clean_settings():
    reset_settings()
    yield
    reset_settings()
```

The Windlass test suite does exactly this. See [Testing](../guides/testing.md).
