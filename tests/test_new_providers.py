"""Tests for the hosted HuggingFace embedder and the fan-out tracer.

Both are new in this release, and both are the kind of adapter whose defects
hide in client construction and response translation rather than in any single
computation — so the HTTP layer is exercised with a mock transport, and the
translation cases that vary between models are pinned individually.
"""

from __future__ import annotations

import httpx
import pytest

from windlass import Windlass
from windlass.core.exceptions import (
    AuthenticationError,
    ConfigurationError,
    ProviderError,
    RateLimitError,
)
from windlass.interfaces.tracer import Span, Tracer
from windlass.providers.embeddings.hf_inference import HuggingFaceInferenceEmbedder
from windlass.providers.observability.console import MemoryTracer


def _embedder(handler, **kwargs) -> HuggingFaceInferenceEmbedder:
    """Build an embedder whose HTTP client is backed by a mock transport."""
    embedder = HuggingFaceInferenceEmbedder(api_key="hf_test", **kwargs)
    embedder._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return embedder


class TestHuggingFaceInferenceEmbedder:
    """The hosted embedding provider."""

    def test_registered_and_needs_no_extra(self) -> None:
        assert "hf_inference" in Windlass.list("embedding")

    def test_known_model_dimensions_without_a_call(self) -> None:
        """A vector index has to be provisioned before the first embed call."""
        embedder = HuggingFaceInferenceEmbedder(api_key="hf_test")
        assert embedder.dimension() == 768

    def test_missing_token_is_an_actionable_error(self, monkeypatch) -> None:
        monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("WINDLASS_HUGGINGFACE_API_KEY", "")
        from windlass.core.config import reset_settings

        reset_settings()
        with pytest.raises(AuthenticationError) as caught:
            HuggingFaceInferenceEmbedder()
        assert "HUGGINGFACE_API_KEY" in str(caught.value)
        reset_settings()

    def test_flat_vectors_pass_through(self) -> None:
        """One vector per input, L2-normalised by the base class for cosine."""
        import math

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        vectors = _embedder(handler).embed(["a", "b"])
        assert len(vectors) == 2
        for vector in vectors:
            assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-6)
        # Direction is preserved: the ratios between components are unchanged.
        assert math.isclose(vectors[0][1] / vectors[0][0], 2.0, rel_tol=1e-6)

    def test_token_level_output_is_mean_pooled(self) -> None:
        """Checkpoints without a pooling layer return [batch][tokens][dim].

        Mean-pooling the token axis is the standard recovery. Indexing the first
        token instead would silently degrade every vector.
        """

        import math

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[[[0.0, 2.0], [2.0, 4.0]]])

        # Mean over the token axis is [1.0, 3.0]; the base class then
        # normalises, so assert on the ratio rather than the raw values.
        # Indexing the first token instead would give a ratio of 2.0/0.0.
        vector = _embedder(handler).embed(["a"])[0]
        assert math.isclose(vector[1] / vector[0], 3.0, rel_tol=1e-6)
        assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-6)

    def test_query_prefix_applied_to_queries_only(self) -> None:
        """BGE wants an instruction prefix on queries but not on documents."""
        seen: list[list[str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            seen.append(json.loads(request.content)["inputs"])
            return httpx.Response(200, json=[[0.1] * 768])

        embedder = _embedder(handler)
        embedder.embed_query("what is rag")
        embedder.embed_one("a document", kind="document")

        assert seen[0][0].startswith("Represent this sentence for searching")
        assert seen[1][0] == "a document"

    def test_wrong_vector_count_is_rejected(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[[0.1, 0.2]])

        with pytest.raises(ProviderError, match="1 vectors for 2 inputs"):
            _embedder(handler).embed(["a", "b"])

    @pytest.mark.parametrize(
        ("status", "expected"),
        [(401, AuthenticationError), (403, AuthenticationError), (429, RateLimitError)],
    )
    def test_http_errors_map_to_the_hierarchy(self, status, expected) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status, text="nope")

        with pytest.raises(expected):
            _embedder(handler).embed(["a"])

    def test_server_error_carries_status_code_for_retry(self) -> None:
        """Retryability is duck-typed on ``exc.status_code``.

        Without it, a transient 5xx from the router fails a whole ingestion run
        that one retry would have fixed.
        """
        from windlass.core.retry import is_retryable

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="upstream boom")

        with pytest.raises(ProviderError) as caught:
            _embedder(handler).embed(["a"])
        assert caught.value.status_code == 500
        assert is_retryable(caught.value)

    def test_unknown_model_id_is_reported_clearly(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="not found")

        with pytest.raises(ProviderError, match="no inference endpoint"):
            _embedder(handler, model="nobody/nothing").embed(["a"])


class _ExplodingTracer(Tracer):
    """A backend that fails on every call, to prove isolation."""

    provider_name = "exploding"

    def start_span(self, span: Span) -> None:
        raise RuntimeError("exporter is down")

    def end_span(self, span: Span) -> None:
        raise RuntimeError("exporter is down")

    def flush(self) -> None:
        raise RuntimeError("exporter is down")


class TestMultiTracer:
    """Fan-out tracing."""

    def test_registered(self) -> None:
        assert "multi" in Windlass.list("tracer")

    def test_every_backend_receives_the_span(self) -> None:
        tracer = Windlass.tracer("multi", backends=["memory", "memory"])
        with tracer.span("work", kind="chain"):
            pass
        assert [len(b.spans) for b in tracer.backends] == [1, 1]

    def test_accepts_instances_as_well_as_names(self) -> None:
        collected = MemoryTracer()
        tracer = Windlass.tracer("multi", backends=[collected, "memory"])
        with tracer.span("work", kind="llm"):
            pass
        assert len(collected.spans) == 1

    def test_a_broken_backend_does_not_break_the_run(self) -> None:
        """A tracer is never allowed to break the application it observes."""
        healthy = MemoryTracer()
        tracer = Windlass.tracer("multi", backends=[_ExplodingTracer(), healthy])

        with tracer.span("work", kind="chain"):  # must not raise
            pass
        tracer.flush()

        assert len(healthy.spans) == 1, "a failing backend starved a healthy one"

    def test_a_broken_backend_is_skipped_after_the_first_failure(self) -> None:
        broken = _ExplodingTracer()
        tracer = Windlass.tracer("multi", backends=[broken, MemoryTracer()])
        for _ in range(3):
            with tracer.span("work", kind="chain"):
                pass
        assert tracer._broken == {0}

    def test_unknown_backend_name_fails_at_construction(self) -> None:
        """A misspelled backend is a config mistake, not a runtime blip."""
        with pytest.raises((ConfigurationError, Exception)):
            Windlass.tracer("multi", backends=["definitely-not-a-tracer"])

    def test_describe_names_the_backends(self) -> None:
        tracer = Windlass.tracer("multi", backends=["memory", "console"])
        assert tracer.describe()["backends"] == ["memory", "console"]

    def test_end_to_end_through_an_agent(self) -> None:
        one, two = MemoryTracer(), MemoryTracer()
        tracer = Windlass.tracer("multi", backends=[one, two])
        Windlass.agent().llm("fake", responses=["hi"]).observe(tracer).run("hello")
        assert one.count("agent") == 1
        assert two.count("agent") == 1
