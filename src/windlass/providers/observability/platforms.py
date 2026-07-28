"""LangSmith and Langfuse tracers.

Both give you a hosted trace viewer: nested spans, prompts, completions, token
counts and latency, searchable across runs. That is the difference between
"users say it got worse" and "the retriever started returning the wrong section
on Tuesday".

Install with::

    pip install "windlass[observability]"

Neither tracer is ever allowed to break your application. Every export path
swallows its own errors and logs a warning, and every ``flush`` runs under a
deadline — a vendor client that blocks on an internal queue must not be able to
hang the process it is observing, and an exception handler cannot catch a hang.

Example:
    >>> from windlass import Windlass                                  # doctest: +SKIP
    >>> rag = Windlass.rag().observe("langfuse")                      # doctest: +SKIP
"""

from __future__ import annotations

import threading
from typing import Any

from windlass.core.config import settings
from windlass.core.exceptions import ConfigurationError
from windlass.core.lazy import require
from windlass.core.registry import register
from windlass.interfaces.tracer import Span, Tracer

__all__ = ["LangSmithTracer", "LangfuseTracer"]

#: Seconds to wait for a backend to flush before giving up on it. Both vendor
#: clients can block indefinitely on an internal queue; a tracing backend must
#: never be able to hang the application it is observing.
FLUSH_TIMEOUT = 10.0

#: Windlass span kinds mapped onto the vocabulary each platform expects.
_LANGSMITH_KINDS = {
    "llm": "llm",
    "embedding": "embedding",
    "retriever": "retriever",
    "tool": "tool",
    "chain": "chain",
    "agent": "chain",
    "guardrail": "chain",
    "ingestion": "chain",
    "evaluation": "chain",
}

#: Windlass span kinds mapped onto Langfuse v3/v4 observation types, which are
#: close to a one-to-one match with Windlass's own taxonomy.
_LANGFUSE_TYPES = {
    "llm": "generation",
    "embedding": "embedding",
    "retriever": "retriever",
    "tool": "tool",
    "chain": "chain",
    "agent": "agent",
    "guardrail": "guardrail",
    "evaluation": "evaluator",
    "ingestion": "span",
}


def _bounded_flush(flusher: Any, *, label: str, timeout: float, log: Any) -> None:
    """Call a vendor ``flush()`` with a deadline.

    Langfuse's ``flush`` joins an internal queue, which blocks forever when a
    worker has stopped with items outstanding — a real hang seen in production,
    and one that ``except Exception`` cannot catch. Running it on a daemon
    thread bounds the damage to a warning.

    Args:
        flusher: Zero-argument callable performing the flush.
        label: Backend name for the log line.
        timeout: Seconds to wait.
        log: Logger to warn on.
    """
    error: list[BaseException] = []

    def _run() -> None:
        try:
            flusher()
        except BaseException as exc:
            error.append(exc)

    worker = threading.Thread(target=_run, name=f"windlass-flush-{label}", daemon=True)
    try:
        worker.start()
    except RuntimeError as exc:
        # Python refuses to start threads once interpreter shutdown has begun,
        # which is exactly when an atexit-driven flush runs. Flushing inline here
        # would risk the very hang this helper exists to bound, so the flush is
        # dropped instead. A tracer is never allowed to break the application —
        # least of all on the way out.
        log.debug("%s flush skipped during interpreter shutdown: %s", label, exc)
        return
    worker.join(timeout)
    if worker.is_alive():
        log.warning(
            "%s did not flush within %.0fs; abandoning the flush so the "
            "application is not blocked. Some traces may be lost.",
            label,
            timeout,
        )
    elif error:
        log.debug("%s flush failed: %s", label, error[0])


@register.tracer(
    "langsmith",
    aliases=("langchain",),
    description="Exports traces to LangSmith.",
)
class LangSmithTracer(Tracer):
    """Sends spans to LangSmith.

    Args:
        api_key: Credential. Falls back to ``LANGSMITH_API_KEY`` /
            ``LANGCHAIN_API_KEY``.
        project: LangSmith project name.
        api_url: Endpoint override for self-hosted deployments.
        tags: Tags applied to every run.
        enabled: Master switch.
        **config: Forwarded to :class:`~windlass.interfaces.tracer.Tracer`.

    Raises:
        MissingDependencyError: When ``langsmith`` is not installed.
        ConfigurationError: When no API key can be found.

    Performance:
        The LangSmith client batches and sends in the background, so tracing adds
        negligible latency to the request path. Call :meth:`flush` before a
        short-lived process exits or you will lose the tail of your traces.
    """

    provider_name = "langsmith"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        project: str | None = None,
        api_url: str | None = None,
        tags: tuple[str, ...] = (),
        enabled: bool = True,
        **config: Any,
    ) -> None:
        super().__init__(enabled=enabled, project=project, tags=tags, **config)
        langsmith = require("langsmith", extra="observability", feature="The LangSmith tracer")
        key = api_key or settings().secret("langsmith_api_key")
        if not key:
            raise ConfigurationError(
                "No API key configured for LangSmith.",
                hint="Set LANGSMITH_API_KEY, or pass "
                "Windlass.tracer('langsmith', api_key='...').",
            )
        self._client = langsmith.Client(api_key=key, api_url=api_url)
        self._runs: dict[str, Any] = {}

    def native(self) -> Any:
        """Return the underlying ``langsmith.Client`` (Level 3 access)."""
        return self._client

    def start_span(self, span: Span) -> None:
        """Create a LangSmith run for the span."""
        import uuid

        try:
            run_id = uuid.UUID(span.id.ljust(32, "0")[:32])
            parent_id = uuid.UUID(span.parent_id.ljust(32, "0")[:32]) if span.parent_id else None
            self._client.create_run(
                id=run_id,
                name=span.name,
                run_type=_LANGSMITH_KINDS.get(span.kind, "chain"),
                inputs=_as_dict(span.inputs),
                parent_run_id=parent_id,
                project_name=self.project,
                tags=list(self.tags),
                extra={"metadata": span.metadata},
            )
            self._runs[span.id] = run_id
            span.attach_native(run_id)
        except Exception as exc:
            self._log.debug("LangSmith create_run failed: %s", exc)

    def end_span(self, span: Span) -> None:
        """Close the LangSmith run."""
        run_id = self._runs.pop(span.id, None)
        if run_id is None:
            return
        try:
            self._client.update_run(
                run_id,
                outputs=_as_dict(span.outputs),
                error=span.error,
                end_time=None,
                extra={
                    "metadata": {
                        **span.metadata,
                        "duration_ms": round(span.duration_ms, 2),
                        **(span.usage.model_dump() if span.usage else {}),
                    }
                },
            )
        except Exception as exc:
            self._log.debug("LangSmith update_run failed: %s", exc)

    def flush(self) -> None:
        """Wait for queued runs to be sent, giving up after ``FLUSH_TIMEOUT``."""
        flusher = getattr(self._client, "flush", None)
        if callable(flusher):
            _bounded_flush(flusher, label="LangSmith", timeout=FLUSH_TIMEOUT, log=self._log)

    async def aclose(self) -> None:
        """Flush, then shut the client's background worker down.

        Without the shutdown, the LangSmith SDK's worker thread outlives the
        application and tries to spawn a *new* thread from its own interpreter
        shutdown hook — which Python refuses once teardown has begun. The result
        is a traceback printed from inside ``langsmith`` after the program has
        already finished successfully, which looks like a crash and is not one.
        """
        self.flush()
        for method in ("cleanup", "shutdown", "close"):
            closer = getattr(self._client, method, None)
            if callable(closer):
                _bounded_flush(
                    closer, label=f"LangSmith {method}", timeout=FLUSH_TIMEOUT, log=self._log
                )
                return


@register.tracer(
    "langfuse",
    description="Exports traces to Langfuse (cloud or self-hosted).",
)
class LangfuseTracer(Tracer):
    """Sends spans to Langfuse.

    Args:
        public_key: Langfuse public key. Falls back to ``LANGFUSE_PUBLIC_KEY``.
        secret_key: Langfuse secret key. Falls back to ``LANGFUSE_SECRET_KEY``.
        host: Endpoint. Falls back to ``LANGFUSE_HOST``, defaulting to the cloud.
        project: Session / project name recorded on each trace.
        tags: Tags applied to every trace.
        enabled: Master switch.
        **config: Forwarded to :class:`~windlass.interfaces.tracer.Tracer`.

    Raises:
        MissingDependencyError: When ``langfuse`` is not installed.
        ConfigurationError: When the key pair is incomplete.

    Note:
        Langfuse's SDK is versioned aggressively and rewrote its span API
        between v2 and v3. This adapter detects which surface the installed
        version exposes and uses it:

        * **v3 / v4** — ``client.start_observation(as_type=...)``, whose
          observation types map almost one-to-one onto Windlass span kinds.
        * **v2** — ``client.trace()`` / ``parent.span()`` / ``parent.generation()``.

        If neither is present the constructor raises. Silently exporting
        nothing while reporting healthy is a worse failure than refusing to
        start: you only discover it when you go looking for traces that were
        never sent.

    Attributes:
        api: ``"v3"`` or ``"v2"`` — which surface was detected.
    """

    provider_name = "langfuse"

    def __init__(
        self,
        *,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
        project: str | None = None,
        tags: tuple[str, ...] = (),
        enabled: bool = True,
        flush_timeout: float = FLUSH_TIMEOUT,
        **config: Any,
    ) -> None:
        super().__init__(enabled=enabled, project=project, tags=tags, **config)
        langfuse = require("langfuse", extra="observability", feature="The Langfuse tracer")
        cfg = settings()
        public = public_key or cfg.secret("langfuse_public_key")
        secret = secret_key or cfg.secret("langfuse_secret_key")
        if not public or not secret:
            raise ConfigurationError(
                "Langfuse needs both a public and a secret key.",
                hint="Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY, or pass them "
                "to Windlass.tracer('langfuse', ...).",
            )
        self.flush_timeout = flush_timeout
        self._client = langfuse.Langfuse(
            public_key=public, secret_key=secret, host=host or cfg.langfuse_host
        )

        if hasattr(self._client, "start_observation"):
            self.api = "v3"
        elif hasattr(self._client, "trace"):
            self.api = "v2"
        else:
            raise ConfigurationError(
                "The installed langfuse version exposes neither "
                "start_observation() (v3/v4) nor trace() (v2).",
                hint="Install a supported release: pip install 'langfuse>=2.50'",
                context={"version": getattr(langfuse, "__version__", "unknown")},
            )
        self._handles: dict[str, Any] = {}

    def native(self) -> Any:
        """Return the underlying ``Langfuse`` client (Level 3 access)."""
        return self._client

    def start_span(self, span: Span) -> None:
        """Open a Langfuse observation, nested under its parent when there is one."""
        try:
            parent = self._handles.get(span.parent_id or "")
            handle = (
                self._start_v3(span, parent) if self.api == "v3" else self._start_v2(span, parent)
            )
            if handle is not None:
                self._handles[span.id] = handle
                span.attach_native(handle)
        except Exception as exc:
            self._log.debug("Langfuse span creation failed: %s", exc)

    def _start_v3(self, span: Span, parent: Any) -> Any:
        """Create an observation with the v3/v4 API."""
        payload: dict[str, Any] = {
            "name": span.name,
            "as_type": _LANGFUSE_TYPES.get(span.kind, "span"),
            "input": span.inputs,
            "metadata": {**span.metadata, "session": self.project, "tags": list(self.tags)},
        }
        if span.kind == "llm" and span.metadata.get("model"):
            payload["model"] = str(span.metadata["model"])
        # Nesting is expressed by creating the child from the parent handle, so
        # the trace tree matches Windlass's own span stack exactly.
        source = parent if parent is not None else self._client
        return source.start_observation(**payload)

    def _start_v2(self, span: Span, parent: Any) -> Any:
        """Create a trace, span or generation with the v2 API."""
        payload: dict[str, Any] = {
            "name": span.name,
            "input": span.inputs,
            "metadata": span.metadata,
        }
        if parent is None:
            return self._client.trace(**payload, session_id=self.project, tags=list(self.tags))
        if span.kind == "llm":
            return parent.generation(**payload, model=span.metadata.get("model"))
        return parent.span(**payload)

    def end_span(self, span: Span) -> None:
        """Record the outcome and close the Langfuse handle."""
        handle = self._handles.pop(span.id, None)
        if handle is None:
            return
        try:
            payload: dict[str, Any] = {
                "output": span.outputs,
                "metadata": {**span.metadata, "duration_ms": round(span.duration_ms, 2)},
            }
            if span.error:
                payload["level"] = "ERROR"
                payload["status_message"] = span.error
            if span.usage:
                usage = {
                    "input": span.usage.prompt_tokens,
                    "output": span.usage.completion_tokens,
                    "total": span.usage.total_tokens,
                }
                # v3 renamed the field and made it observation-level.
                payload["usage_details" if self.api == "v3" else "usage"] = usage

            if self.api == "v3":
                handle.update(**payload)
                handle.end()
            else:
                updater = getattr(handle, "end", None) or getattr(handle, "update", None)
                if callable(updater):
                    updater(**payload)
        except Exception as exc:
            self._log.debug("Langfuse span close failed: %s", exc)

    def flush(self) -> None:
        """Send any buffered events, giving up after :attr:`flush_timeout`.

        Langfuse's own ``flush`` joins an internal queue and can block forever
        when a worker thread has already stopped with items outstanding.
        """
        _bounded_flush(
            self._client.flush,
            label="Langfuse",
            timeout=self.flush_timeout,
            log=self._log,
        )

    async def aclose(self) -> None:
        """Flush and shut the client down, both under a deadline."""
        self.flush()
        shutdown = getattr(self._client, "shutdown", None)
        if callable(shutdown):
            _bounded_flush(
                shutdown, label="Langfuse shutdown", timeout=self.flush_timeout, log=self._log
            )


def _as_dict(value: Any) -> dict[str, Any]:
    """Coerce a span payload into the dict shape both platforms expect."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return {"value": value}
