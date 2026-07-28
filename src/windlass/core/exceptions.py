"""Exception hierarchy for Windlass.

Every error raised by the framework derives from :class:`WindlassError`, so
applications can wrap a whole Windlass call in a single ``except`` block::

    from windlass import WindlassError

    try:
        rag.ask("...")
    except WindlassError as exc:
        log.error("windlass failed: %s", exc)

Errors carry structured context in :attr:`WindlassError.context` and render a
human-readable remediation hint whenever one is available. Windlass never lets a
raw :class:`ImportError` from an optional dependency reach user code — those are
translated into :class:`MissingDependencyError` with the exact ``pip install``
command to run.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "FIX_MALFORMED_TOOL_CALL",
    "AgentError",
    "AuthenticationError",
    "ComponentNotFoundError",
    "ConfigurationError",
    "DuplicateComponentError",
    "EvaluationError",
    "GuardrailViolation",
    "IngestionError",
    "InterruptedError_",
    "MCPError",
    "MalformedToolCallError",
    "MaxIterationsExceeded",
    "MemoryError_",
    "MissingDependencyError",
    "PipelineError",
    "PluginError",
    "ProviderError",
    "ProviderTimeoutError",
    "RateLimitError",
    "RegistryError",
    "ResponseError",
    "RetrievalError",
    "SerializationError",
    "ToolError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ValidationError",
    "WindlassError",
]


class WindlassError(Exception):
    """Base class for every error raised by Windlass.

    Args:
        message: Human readable description of what went wrong.
        hint: Optional actionable remediation shown after the message.
        context: Arbitrary structured detail (provider name, model, ids, ...).
            Useful for logging and for tests that assert on specifics.

    Attributes:
        message: The raw message without the hint appended.
        hint: The remediation hint, if any.
        context: Structured detail about the failure.

    Example:
        >>> raise WindlassError("boom", hint="try again", context={"attempt": 2})
        Traceback (most recent call last):
        windlass.core.exceptions.WindlassError: boom
        <BLANKLINE>
        Hint: try again
    """

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.hint = hint
        self.context: dict[str, Any] = dict(context or {})
        super().__init__(self._render())

    def _render(self) -> str:
        parts = [self.message]
        if self.hint:
            parts.append(f"\n\nHint: {self.hint}")
        return "".join(parts)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}({self.message!r}, context={self.context!r})"


# --------------------------------------------------------------------------
# Configuration & wiring
# --------------------------------------------------------------------------
class ConfigurationError(WindlassError):
    """A component was configured with invalid or incomplete settings."""


class MissingDependencyError(WindlassError):
    """An optional dependency required by the requested feature is not installed.

    Windlass keeps its core install tiny; every integration lives behind an
    optional dependency group. When a feature is used without its group being
    installed this error explains exactly which command fixes it.

    Args:
        feature: Human readable feature name, e.g. ``"The Pinecone vector store"``.
        extra: The extras group to install, e.g. ``"vectordb"``.
        package: The distribution that failed to import, for the context dict.
        original: The underlying :class:`ImportError`, kept for debugging.

    Example:
        >>> raise MissingDependencyError("Evaluation", extra="evaluation")
        Traceback (most recent call last):
        windlass.core.exceptions.MissingDependencyError: Evaluation is not installed.
        <BLANKLINE>
        Hint: Run:
        <BLANKLINE>
            pip install "windlass[evaluation]"
    """

    def __init__(
        self,
        feature: str,
        *,
        extra: str,
        package: str | None = None,
        original: BaseException | None = None,
    ) -> None:
        self.feature = feature
        self.extra = extra
        self.package = package
        self.original = original
        super().__init__(
            f"{feature} is not installed.",
            hint=f'Run:\n\n    pip install "windlass[{extra}]"',
            context={"feature": feature, "extra": extra, "package": package},
        )


class RegistryError(WindlassError):
    """Base class for component-registry problems."""


class ComponentNotFoundError(RegistryError):
    """No component is registered under the requested name.

    The message lists the available names so typos are obvious.
    """

    def __init__(self, kind: str, name: str, available: list[str] | None = None) -> None:
        self.kind = kind
        self.name = name
        self.available = sorted(available or [])
        options = ", ".join(self.available) if self.available else "(none registered)"
        super().__init__(
            f"No {kind} named {name!r} is registered.",
            hint=(
                f"Available {kind}s: {options}\n"
                f"Register your own with @windlass.register.{kind}('{name}')."
            ),
            context={"kind": kind, "name": name, "available": self.available},
        )


class DuplicateComponentError(RegistryError):
    """A component name is already taken and ``override=False``."""

    def __init__(self, kind: str, name: str) -> None:
        super().__init__(
            f"A {kind} named {name!r} is already registered.",
            hint="Pass override=True to replace the existing component deliberately.",
            context={"kind": kind, "name": name},
        )


class PluginError(WindlassError):
    """A third-party plugin failed to load or misbehaved during discovery."""


# --------------------------------------------------------------------------
# Provider / transport
# --------------------------------------------------------------------------
class ProviderError(WindlassError):
    """A backing provider (LLM, vector store, ...) returned an error.

    Args:
        message: What went wrong.
        provider: Which provider raised it.
        status_code: HTTP status, when the failure came from an HTTP call.
            **Pass this whenever you have it.** Retry classification reads it
            (see :func:`windlass.core.retry.is_retryable`), so an adapter that
            omits it turns a transient 429 or 503 — which one retry would have
            fixed — into a hard failure of the whole run.
        hint: Actionable remediation.
        context: Structured detail.
        original: The underlying exception, kept for debugging.

    Attributes:
        provider: The provider name.
        status_code: The HTTP status, when known.
        original: The wrapped exception, when there was one.

    Example:
        >>> err = ProviderError("upstream boom", provider="acme", status_code=503)
        >>> from windlass.core.retry import is_retryable
        >>> is_retryable(err)
        True
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int | None = None,
        hint: str | None = None,
        context: dict[str, Any] | None = None,
        original: BaseException | None = None,
    ) -> None:
        self.provider = provider
        self.status_code = status_code
        self.original = original
        ctx = {"provider": provider, **(context or {})}
        if status_code is not None:
            ctx.setdefault("status_code", status_code)
        super().__init__(message, hint=hint, context=ctx)


class AuthenticationError(ProviderError):
    """Credentials are missing, expired, or rejected by the provider."""


class RateLimitError(ProviderError):
    """The provider applied rate limiting or quota exhaustion.

    Attributes:
        retry_after: Seconds the provider asked us to wait, when advertised.
    """

    def __init__(self, message: str, *, retry_after: float | None = None, **kwargs: Any) -> None:
        self.retry_after = retry_after
        kwargs.setdefault("context", {})["retry_after"] = retry_after
        super().__init__(message, **kwargs)


class ProviderTimeoutError(ProviderError):
    """The provider did not answer within the configured timeout."""


class ResponseError(ProviderError):
    """The provider replied, but the payload could not be interpreted."""


class MalformedToolCallError(ResponseError):
    """The model emitted a tool call the provider could not parse.

    This is a *model mistake*, not a provider failure, and it is recoverable in
    exactly the way a hallucinated tool name is: show the model what went wrong
    and let it try again. :class:`~windlass.agent.runtime.AgentRuntime` catches
    it and feeds :attr:`feedback` back into the conversation rather than ending
    the run.

    The commonest cause is a model trying to express a data dependency the
    tool-calling protocol cannot represent — nesting one call inside another's
    arguments, as in ``settle(amount=<function=compute>{...}</function>)``,
    because the value it needs does not exist until the first call has run.
    The protocol requires that to happen across two turns.

    Attributes:
        raw: The provider's rejected generation, when it reports one. Useful in
            logs; it is the only place the model's actual intent survives.

    Example:
        >>> err = MalformedToolCallError("bad call", provider="groq", raw="<function=x>{")
        >>> "one tool call at a time" in err.feedback
        True
    """

    def __init__(self, message: str, *, raw: str | None = None, **kwargs: Any) -> None:
        self.raw = raw
        kwargs.setdefault("hint", FIX_MALFORMED_TOOL_CALL)
        ctx = kwargs.setdefault("context", {})
        if raw:
            ctx["failed_generation"] = raw[:500]
        super().__init__(message, **kwargs)

    @property
    def feedback(self) -> str:
        """The correction shown to the model so it can retry.

        Written as an instruction rather than a stack trace, because the reader
        is a language model deciding what to emit next.
        """
        detail = f"\n\nWhat you sent was:\n{self.raw[:400]}" if self.raw else ""
        return (
            "Your previous tool call could not be parsed and was rejected. "
            "Make one tool call at a time, and pass only literal JSON values as "
            "arguments — never place a tool call inside another tool's "
            "arguments. If one tool needs another's result, call the first tool "
            "now, wait for its result, then use that value in the next call."
            f"{detail}"
        )


#: Guidance attached to :class:`MalformedToolCallError` for the human reading logs.
FIX_MALFORMED_TOOL_CALL = (
    "The model nested or mis-formatted a tool call. Windlass feeds this back to "
    "the model automatically. If it recurs, tell the model in its system prompt "
    "to make one tool call per turn and never nest calls, or reduce the number "
    "of bound tools."
)


# --------------------------------------------------------------------------
# Validation & safety
# --------------------------------------------------------------------------
class ValidationError(WindlassError):
    """User supplied data failed validation before reaching a provider."""


class GuardrailViolation(WindlassError):
    """A guardrail blocked an input or an output.

    Args:
        message: What the guardrail objected to.
        stage: ``"input"`` or ``"output"``.
        rule: Identifier of the rule that fired.
        detections: Structured detail about each detection.
    """

    def __init__(
        self,
        message: str,
        *,
        stage: str = "input",
        rule: str | None = None,
        detections: list[dict[str, Any]] | None = None,
    ) -> None:
        self.stage = stage
        self.rule = rule
        self.detections = detections or []
        super().__init__(
            message,
            hint="Relax the policy with .guardrails(on_violation='redact') "
            "to sanitise instead of block.",
            context={"stage": stage, "rule": rule, "detections": self.detections},
        )


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------
class PipelineError(WindlassError):
    """A RAG or ingestion pipeline step failed."""


class IngestionError(PipelineError):
    """A document could not be loaded, preprocessed, chunked or indexed."""


class RetrievalError(PipelineError):
    """Retrieval failed or returned an unusable result."""


# --------------------------------------------------------------------------
# Tools & agents
# --------------------------------------------------------------------------
class ToolError(WindlassError):
    """Base class for tool related failures."""


class ToolNotFoundError(ToolError):
    """The model asked to call a tool that is not registered."""

    def __init__(self, name: str, available: list[str] | None = None) -> None:
        options = ", ".join(sorted(available or [])) or "(no tools bound)"
        super().__init__(
            f"The model requested an unknown tool {name!r}.",
            hint=f"Tools available to this agent: {options}",
            context={"tool": name, "available": sorted(available or [])},
        )


class ToolExecutionError(ToolError):
    """A tool raised while executing.

    The original exception is preserved on :attr:`original` and as the
    ``__cause__`` of this error.
    """

    def __init__(self, name: str, original: BaseException) -> None:
        self.tool = name
        self.original = original
        super().__init__(
            f"Tool {name!r} raised {type(original).__name__}: {original}",
            context={"tool": name, "error_type": type(original).__name__},
        )


class AgentError(WindlassError):
    """The agent runtime could not complete the request."""


class MaxIterationsExceeded(AgentError):
    """The agent hit its reasoning-step budget without producing an answer."""

    def __init__(self, limit: int) -> None:
        super().__init__(
            f"Agent exceeded its budget of {limit} reasoning steps without finishing.",
            hint="Raise the ceiling with .max_iterations(n) or simplify the task.",
            context={"limit": limit},
        )


class InterruptedError_(AgentError):
    """Execution paused for human review (human-in-the-loop interrupt).

    Named with a trailing underscore to avoid shadowing the builtin
    ``InterruptedError``. Exported as ``AgentInterrupt`` from :mod:`windlass`.

    Attributes:
        payload: What the agent wants a human to approve or edit.
        thread_id: Checkpoint thread that can be resumed.
    """

    def __init__(
        self,
        message: str = "Execution paused for human approval.",
        *,
        payload: Any = None,
        thread_id: str | None = None,
    ) -> None:
        self.payload = payload
        self.thread_id = thread_id
        super().__init__(
            message,
            hint="Resume with agent.resume(thread_id=..., approved=True).",
            context={"thread_id": thread_id},
        )


class MCPError(WindlassError):
    """An MCP server could not be reached or returned an error."""


class EvaluationError(WindlassError):
    """An evaluation run or metric computation failed."""


class MemoryError_(WindlassError):
    """A memory backend failed. Exported as ``WindlassMemoryError``."""


class SerializationError(WindlassError):
    """State could not be serialised to or from a checkpoint."""
