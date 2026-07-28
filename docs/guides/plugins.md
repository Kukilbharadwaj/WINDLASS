# Writing plugins

Every one of the fifteen component kinds is an extension point. A component you write is resolved, configured, traced and composed exactly like a built-in one — because there is no privileged path. Windlass resolves its own providers through the same registry your plugin registers into.

## The shape of a plugin

Three steps, always:

1. Subclass the interface for the kind.
2. Implement the one or two required methods.
3. Register a name.

```python
from windlass import Chunker, register


@register.chunker("by-sentence", description="Splits on sentence boundaries.")
class SentenceChunker(Chunker):
    """Splits text into one chunk per sentence.

    Args:
        max_sentences: Sentences to group into a single chunk.
        **config: Forwarded to Chunker.
    """

    def __init__(self, *, max_sentences: int = 3, **config):
        super().__init__(**config)
        self.max_sentences = max_sentences

    def split_text(self, text: str) -> list[str]:
        from windlass.core.text import split_sentences

        sentences = split_sentences(text)
        return [
            " ".join(sentences[i : i + self.max_sentences])
            for i in range(0, len(sentences), self.max_sentences)
        ]
```

```python
rag = Windlass.rag().chunker("by-sentence", max_sentences=5)
```

That is the whole story. Document iteration, metadata propagation, offset tracking, id assignment, under-sized chunk merging and concurrency are all handled by the base class.

## What each kind requires

| Kind | Interface | You implement |
|---|---|---|
| `llm` | `LLM` | `agenerate`; optionally `astream_generate` |
| `embedding` | `Embedder` | `aembed_texts` |
| `loader` | `Loader` | `aload_source`; set `extensions` |
| `preprocessor` | `Preprocessor` | `aprocess_one` |
| `chunker` | `Chunker` | `split_text` (or `asplit_text`) |
| `retriever` | `Retriever` | `aretrieve_chunks`; optionally `aindex` |
| `vectordb` | `VectorStore` | `aadd`, `asearch`, `adelete`, `acount` |
| `memory` | `Memory` | `aadd`, `aget` |
| `guardrail` | `Guardrail` | `acheck` |
| `evaluator` | `Evaluator` | `aevaluate_sample` |
| `tracer` | `Tracer` | `start_span`; optionally `end_span`, `flush` |
| `tool` | `Tool` | `acall` — or just use `@tool` |
| `mcp` | `MCPClient` | `aconnect`, `alist_tools`, `acall_tool` |

Implement the **async** method. The blocking API is derived from it, so the two cannot drift.

## Worked examples

### A retriever backed by your existing search service

The most common real plugin: you already have search, and you want the model to use it.

```python
from windlass import Chunk, Retriever, ScoredChunk, register


@register.retriever("company-search", description="Our internal search service.")
class CompanySearch(Retriever):
    """Queries the existing Elasticsearch-backed search API.

    Args:
        client: A configured search client.
        index: Which index to query.
        **config: Forwarded to Retriever.
    """

    def __init__(self, *, client=None, index: str = "docs", **config):
        super().__init__(**config)
        if client is None:
            raise ConfigurationError(
                "CompanySearch needs a client.",
                hint="Pass client=SearchClient(...).",
            )
        self.client = client
        self.index = index

    async def aretrieve_chunks(self, query, k, *, filters=None, **kwargs):
        hits = await self.client.search(self.index, query, size=k, filters=filters)
        return [
            ScoredChunk(
                chunk=Chunk(
                    content=hit["text"],
                    metadata={"source": hit["url"], **hit.get("meta", {})},
                    document_id=hit["doc_id"],
                ),
                score=hit["score"],
            )
            for hit in hits
        ]

    def native(self):
        """Return the search client, for callers who need it directly."""
        return self.client
```

It now composes into hybrid retrieval alongside dense search:

```python
rag = Windlass.rag().retriever(
    "hybrid",
    retrievers=[dense_retriever, CompanySearch(client=my_client)],
    weights=[0.5, 0.5],
)
```

### An LLM provider

```python
from windlass import LLM, Completion, register


@register.llm("acme", description="The Acme inference API.")
class AcmeLLM(LLM):
    """Chat completions from the Acme API.

    Args:
        model: Model id.
        api_key: Credential; falls back to ACME_API_KEY.
        **config: Forwarded to LLM.
    """

    provider_name = "acme"
    supports_tools = True
    supports_streaming = True

    def __init__(self, model: str = "", *, api_key: str | None = None, **config):
        super().__init__(model=model, **config)
        import os

        key = api_key or os.getenv("ACME_API_KEY")
        if not key:
            raise AuthenticationError(
                "No API key configured for Acme.",
                provider="acme",
                hint="Set ACME_API_KEY, or pass Windlass.llm('acme', api_key='...').",
            )
        self._client = AcmeClient(api_key=key, timeout=self.timeout)

    @classmethod
    def default_model(cls) -> str:
        return "acme-large"

    async def agenerate(self, messages, *, tools=None, **kwargs):
        response = await self._client.chat(
            model=self.model,
            messages=[{"role": m.role.value, "content": m.content} for m in messages],
            tools=tools,
            **self._merged(**kwargs),
        )
        return Completion(
            content=response.text,
            model=self.model,
            usage=self._usage(response.prompt_tokens, response.completion_tokens),
            raw=response,
        )

    async def astream_generate(self, messages, *, tools=None, **kwargs):
        from windlass import StreamEvent

        async for chunk in self._client.stream(model=self.model, messages=messages):
            yield StreamEvent(type="text", delta=chunk.text, raw=chunk)
        yield StreamEvent(type="done")

    def native(self):
        return self._client

    async def aclose(self):
        await self._client.close()
```

### A guardrail

```python
from windlass import Guardrail, GuardrailResult, register


@register.guardrail("no-competitors")
class CompetitorGuardrail(Guardrail):
    """Blocks mentions of named competitors in generated output.

    Args:
        names: Competitor names to catch.
        **config: Forwarded to Guardrail.
    """

    def __init__(self, *, names: list[str] | None = None, **config):
        config.setdefault("stages", ("output",))
        super().__init__(**config)
        self.names = [n.lower() for n in (names or [])]

    async def acheck(self, content, *, stage="input", context=None):
        found = [n for n in self.names if n in content.lower()]
        redacted = content
        for name in found:
            redacted = re.sub(re.escape(name), "[COMPETITOR]", redacted, flags=re.I)
        return GuardrailResult(
            allowed=not found,
            content=redacted,
            detections=[{"rule": "competitor", "name": n} for n in found],
            rule="competitor" if found else None,
            stage=stage,
        )
```

Report *what you found* and leave the policy to `on_violation`. That keeps one detector usable in both blocking and redacting configurations.

### A tracer

```python
from windlass import register
from windlass.interfaces.tracer import Span, Tracer


@register.tracer("otel")
class OpenTelemetryTracer(Tracer):
    """Exports Windlass spans to OpenTelemetry."""

    def __init__(self, **config):
        super().__init__(**config)
        from opentelemetry import trace

        self._tracer = trace.get_tracer("windlass")
        self._active: dict[str, object] = {}

    def start_span(self, span: Span) -> None:
        otel = self._tracer.start_span(span.name)
        otel.set_attribute("windlass.kind", span.kind)
        self._active[span.id] = otel
        span.attach_native(otel)

    def end_span(self, span: Span) -> None:
        otel = self._active.pop(span.id, None)
        if otel is None:
            return
        otel.set_attribute("windlass.duration_ms", span.duration_ms)
        if span.usage:
            otel.set_attribute("windlass.total_tokens", span.usage.total_tokens)
        if span.error:
            otel.set_attribute("windlass.error", span.error)
        otel.end()
```

!!! important "A tracer must never break the application"
    Windlass already wraps `start_span` and `end_span` in a try/except, but keep your own export path defensive too.

## Distributing a plugin

Registering by decorator only works once your module is imported. To make a component appear in *other people's* Windlass, publish an entry point.

```
windlass-acme/
├── pyproject.toml
├── src/windlass_acme/
│   ├── __init__.py
│   └── llm.py
└── tests/
```

```toml
# pyproject.toml
[project]
name = "windlass-acme"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["windlass>=0.1", "acme-sdk>=2.0"]

[project.entry-points."windlass.llm"]
acme = "windlass_acme.llm:AcmeLLM"
```

```bash
pip install windlass-acme
```

```python
Windlass.llm("acme")          # it is simply there
```

The value is a dotted path, so your module is imported only when somebody uses the component.

### Registering several components at once

```toml
[project.entry-points."windlass.plugins"]
acme = "windlass_acme:setup"
```

```python
# src/windlass_acme/__init__.py
def setup(registry):
    """Register every Acme component. Called by Windlass during discovery."""
    registry.register_lazy("llm", "acme", "windlass_acme.llm:AcmeLLM", extra="acme")
    registry.register_lazy("embedding", "acme", "windlass_acme.embed:AcmeEmbedder")
    registry.register_lazy("vectordb", "acme", "windlass_acme.store:AcmeStore")
```

Use `register_lazy` here so installing your plugin does not import your dependencies at `import windlass` time.

### Verifying discovery

```python
from windlass import Windlass

print(Windlass.plugins(strict=True))   # raises on the first plugin that fails
print(Windlass.list("llm"))
```

```bash
windlass list llm -v
```

By default a plugin that fails to load logs a warning and is skipped, so one broken third-party package cannot take down the framework. `configure(strict_plugins=True)` inverts that for development.

## A custom component kind

If you need an extension point Windlass does not have:

```python
from windlass import REGISTRY, register

REGISTRY.add_kind("reranker")


@register.component("reranker", "cohere")
class CohereReranker:
    """Reranks retrieved chunks with the Cohere rerank API."""

    async def arerank(self, query, hits, k):
        scored = await cohere.rerank(query=query, documents=[h.content for h in hits])
        return [hits[r.index] for r in scored[:k]]
```

```python
reranker = Windlass.create("reranker", "cohere")
rag = Windlass.rag().retriever("vector", rerank=reranker, fetch_k=50)
```

Any retriever accepts a `rerank=` object with an `arerank(query, hits, k)` coroutine.

## Testing a plugin

```python
import pytest

from windlass import Windlass
from windlass.testing import isolated_registry


def test_registers_and_constructs():
    with isolated_registry() as registry:
        import windlass_acme                      # noqa: F401 — triggers registration

        assert registry.has("llm", "acme")
        assert Windlass.llm("acme", api_key="test").model == "acme-large"


async def test_satisfies_the_contract():
    from windlass.core.types import Message

    llm = Windlass.llm("acme", api_key="test")
    completion = await llm.acomplete("hello")
    assert isinstance(completion.content, str)
    assert completion.usage.total_tokens >= 0


def test_composes_into_a_pipeline():
    rag = Windlass.rag().llm("acme", api_key="test")
    rag.ingest_text("Some content to retrieve later.")
    assert rag.ask("what content?").contexts
```

`isolated_registry()` snapshots and restores the registry, so registration in one test cannot leak into another.

## Checklist for a good plugin

- [ ] Subclasses the right interface; implements the async methods
- [ ] Raises `ConfigurationError` / `AuthenticationError` with a **hint** that says what to do
- [ ] Heavy imports use `windlass.core.lazy.require`, so a missing dependency is actionable
- [ ] `native()` returns the wrapped object
- [ ] Google-style docstrings with `Args`, `Returns`, `Raises` and an `Example`
- [ ] Registered with a description and any aliases
- [ ] Distributed via an entry point, registered with `register_lazy`
- [ ] Tested against the interface contract *and* composed into a pipeline
- [ ] Thread-safe if it holds mutable state
