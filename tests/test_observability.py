"""Tests for the hosted tracing backends.

The Langfuse SDK rewrote its span API between v2 and v3, and Windlass has to
work with both. These tests drive the adapter against fake clients shaped like
each generation, so the version-detection and translation logic is covered
without credentials or network.

They exist because the adapter shipped targeting v2 only, and on an installed
v3 client it exported *nothing at all* — every call failed and was swallowed by
the defensive `except Exception`. Silence is the hardest failure to notice, so
it is now asserted directly.
"""

from __future__ import annotations

import threading
import time

import pytest

from windlass.core.exceptions import ConfigurationError
from windlass.core.types import Usage
from windlass.providers.observability.platforms import (
    _LANGFUSE_TYPES,
    LangfuseTracer,
    _bounded_flush,
)


# ---------------------------------------------------------------------------
# Fake Langfuse clients
# ---------------------------------------------------------------------------
class _V3Observation:
    """Stands in for a LangfuseSpan / LangfuseGeneration."""

    def __init__(self, **payload):
        self.payload = payload
        self.children: list[_V3Observation] = []
        self.updates: list[dict] = []
        self.ended = False

    def start_observation(self, **payload):
        child = _V3Observation(**payload)
        self.children.append(child)
        return child

    def update(self, **payload):
        self.updates.append(payload)
        return self

    def end(self, **_):
        self.ended = True
        return self


class _V3Client:
    """A langfuse >= 3 client: start_observation, no trace()."""

    def __init__(self, **_):
        self.roots: list[_V3Observation] = []
        self.flushed = 0

    def start_observation(self, **payload):
        root = _V3Observation(**payload)
        self.roots.append(root)
        return root

    def flush(self):
        self.flushed += 1


class _V2Handle:
    def __init__(self, kind, **payload):
        self.kind = kind
        self.payload = payload
        self.children: list[_V2Handle] = []
        self.ended: dict | None = None

    def span(self, **payload):
        child = _V2Handle("span", **payload)
        self.children.append(child)
        return child

    def generation(self, **payload):
        child = _V2Handle("generation", **payload)
        self.children.append(child)
        return child

    def end(self, **payload):
        self.ended = payload
        return self


class _V2Client:
    """A langfuse 2.x client: trace(), no start_observation."""

    def __init__(self, **_):
        self.traces: list[_V2Handle] = []
        self.flushed = 0

    def trace(self, **payload):
        root = _V2Handle("trace", **payload)
        self.traces.append(root)
        return root

    def flush(self):
        self.flushed += 1


class _ApilessClient:
    """A client exposing neither generation's span API."""

    def __init__(self, **_):
        pass

    def flush(self):
        pass


def _install(monkeypatch, client_cls, version="4.0.0"):
    """Make `require("langfuse")` return a module exposing ``client_cls``."""
    import types

    module = types.SimpleNamespace(Langfuse=client_cls, __version__=version)
    monkeypatch.setattr(
        "windlass.providers.observability.platforms.require", lambda *a, **k: module
    )
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    from windlass.core.config import reset_settings

    reset_settings()


class TestLangfuseVersionDetection:
    def test_v3_client_is_detected(self, monkeypatch):
        _install(monkeypatch, _V3Client)
        assert LangfuseTracer().api == "v3"

    def test_v2_client_is_detected(self, monkeypatch):
        _install(monkeypatch, _V2Client, version="2.60.0")
        assert LangfuseTracer().api == "v2"

    def test_an_unrecognised_client_raises_instead_of_exporting_nothing(self, monkeypatch):
        """Failing loudly beats reporting healthy while sending nothing."""
        _install(monkeypatch, _ApilessClient, version="99.0.0")
        with pytest.raises(ConfigurationError, match="start_observation"):
            LangfuseTracer()

    def test_missing_credentials_raise(self, monkeypatch, tmp_path):
        _install(monkeypatch, _V3Client)
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY")
        # Settings also read a .env from the working directory, so the test has
        # to run somewhere without one.
        monkeypatch.chdir(tmp_path)
        from windlass.core.config import reset_settings

        reset_settings()
        with pytest.raises(ConfigurationError, match="public and a secret key"):
            LangfuseTracer(secret_key="sk-test")


class TestLangfuseV3Export:
    def test_a_span_is_actually_exported(self, monkeypatch):
        """The regression: on a v3 client the adapter used to export nothing."""
        _install(monkeypatch, _V3Client)
        tracer = LangfuseTracer()
        with tracer.span("adjudicate", kind="agent", inputs={"claim": "C-1"}) as span:
            span.set_output("approved")

        client = tracer.native()
        assert len(client.roots) == 1, "nothing reached the client"
        assert client.roots[0].payload["name"] == "adjudicate"
        assert span.native() is client.roots[0]

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            ("llm", "generation"),
            ("embedding", "embedding"),
            ("retriever", "retriever"),
            ("tool", "tool"),
            ("agent", "agent"),
            ("guardrail", "guardrail"),
            ("chain", "chain"),
            ("evaluation", "evaluator"),
            ("ingestion", "span"),
        ],
    )
    def test_span_kinds_map_onto_observation_types(self, monkeypatch, kind, expected):
        _install(monkeypatch, _V3Client)
        tracer = LangfuseTracer()
        with tracer.span("work", kind=kind):
            pass
        assert tracer.native().roots[0].payload["as_type"] == expected

    def test_every_windlass_kind_has_a_mapping(self):
        from windlass.interfaces.tracer import SPAN_KINDS

        assert set(SPAN_KINDS) <= set(_LANGFUSE_TYPES)

    def test_nesting_creates_children_from_the_parent_handle(self, monkeypatch):
        _install(monkeypatch, _V3Client)
        tracer = LangfuseTracer()
        with tracer.span("root", kind="agent"), tracer.span("child", kind="llm"):
            pass

        client = tracer.native()
        assert len(client.roots) == 1, "the child should not be a second root"
        assert [c.payload["name"] for c in client.roots[0].children] == ["child"]

    def test_output_and_usage_are_recorded_on_close(self, monkeypatch):
        _install(monkeypatch, _V3Client)
        tracer = LangfuseTracer()
        with tracer.span("generate", kind="llm", model="llama-3.3-70b") as span:
            span.set_output("an answer")
            span.set_usage(Usage(prompt_tokens=120, completion_tokens=18))

        root = tracer.native().roots[0]
        assert root.ended is True
        update = root.updates[-1]
        assert update["output"] == "an answer"
        assert update["usage_details"] == {"input": 120, "output": 18, "total": 138}

    def test_the_model_is_passed_for_generations(self, monkeypatch):
        _install(monkeypatch, _V3Client)
        tracer = LangfuseTracer()
        with tracer.span("generate", kind="llm", model="llama-3.3-70b"):
            pass
        assert tracer.native().roots[0].payload["model"] == "llama-3.3-70b"

    def test_an_error_is_recorded_as_a_level(self, monkeypatch):
        _install(monkeypatch, _V3Client)
        tracer = LangfuseTracer()
        with pytest.raises(ValueError), tracer.span("boom", kind="tool"):
            raise ValueError("tool exploded")

        update = tracer.native().roots[0].updates[-1]
        assert update["level"] == "ERROR"
        assert "tool exploded" in update["status_message"]


class TestLangfuseV2Export:
    def test_a_root_span_becomes_a_trace(self, monkeypatch):
        _install(monkeypatch, _V2Client, version="2.60.0")
        tracer = LangfuseTracer()
        with tracer.span("root", kind="chain"):
            pass
        assert tracer.native().traces[0].kind == "trace"

    def test_a_nested_llm_span_becomes_a_generation(self, monkeypatch):
        _install(monkeypatch, _V2Client, version="2.60.0")
        tracer = LangfuseTracer()
        with tracer.span("root", kind="chain"), tracer.span("call", kind="llm"):
            pass
        assert tracer.native().traces[0].children[0].kind == "generation"

    def test_usage_uses_the_v2_field_name(self, monkeypatch):
        _install(monkeypatch, _V2Client, version="2.60.0")
        tracer = LangfuseTracer()
        with tracer.span("root", kind="llm") as span:
            span.set_usage(Usage(prompt_tokens=10, completion_tokens=5))
        assert "usage" in tracer.native().traces[0].ended


class TestBoundedFlush:
    def test_a_normal_flush_completes(self):
        calls = []
        _bounded_flush(lambda: calls.append(1), label="test", timeout=5, log=_NullLog())
        assert calls == [1]

    def test_a_hanging_flush_is_abandoned_not_awaited(self):
        """The production hang: Langfuse's flush joins a queue that never drains."""
        release = threading.Event()
        warnings: list[str] = []

        started = time.perf_counter()
        _bounded_flush(
            lambda: release.wait(30),
            label="Wedged",
            timeout=0.3,
            log=_RecordingLog(warnings),
        )
        elapsed = time.perf_counter() - started
        release.set()

        assert elapsed < 5, "flush blocked past its deadline"
        assert warnings and "did not flush" in warnings[0]

    def test_a_raising_flush_is_swallowed(self):
        def explode():
            raise RuntimeError("vendor is down")

        _bounded_flush(explode, label="test", timeout=5, log=_NullLog())

    def test_tracer_flush_does_not_hang(self, monkeypatch):
        class _Wedged(_V3Client):
            def flush(self):
                threading.Event().wait(30)

        _install(monkeypatch, _Wedged)
        tracer = LangfuseTracer(flush_timeout=0.3)
        started = time.perf_counter()
        tracer.flush()
        assert time.perf_counter() - started < 5


class _NullLog:
    def warning(self, *a, **k): ...
    def debug(self, *a, **k): ...


class _RecordingLog:
    def __init__(self, sink):
        self.sink = sink

    def warning(self, message, *args):
        self.sink.append(message % args if args else message)

    def debug(self, *a, **k): ...
