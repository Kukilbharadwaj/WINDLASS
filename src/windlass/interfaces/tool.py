"""The tool interface.

A tool is a capability an agent can invoke: a Python function, an HTTP call, a
database query, a remote MCP endpoint. Windlass represents all of them with this
one class, so an agent binding a local function and a remote MCP tool cannot
tell the difference.

Most users never subclass this — the :func:`~windlass.tools.tool` decorator wraps
a plain function, derives the JSON schema from its type hints and docstring, and
registers it. Subclass :class:`Tool` when the capability is not a function:
stateful clients, dynamically generated tools, or remote proxies.

Implementers override one coroutine, :meth:`Tool.acall`.

Example:
    >>> from windlass.tools import tool
    >>> @tool
    ... def add(a: int, b: int) -> int:
    ...     '''Add two numbers.'''
    ...     return a + b
    >>> add.schema()["function"]["name"]
    'add'
    >>> add.run(a=1, b=2).content
    '3'
"""

from __future__ import annotations

import abc
import json
import time
from typing import Any

from windlass.core.concurrency import run_sync
from windlass.core.exceptions import ToolExecutionError
from windlass.core.types import ToolCall, ToolResult
from windlass.interfaces.base import Component

__all__ = ["Tool"]

#: How long a tool may run before the agent gives up on it, unless overridden.
DEFAULT_TOOL_TIMEOUT = 60.0


class Tool(Component):
    """Abstract agent-callable capability.

    Args:
        name: Tool name the model sees. Must match ``^[a-zA-Z0-9_-]{1,64}$``,
            which is what the major providers accept.
        description: What the tool does. This is *prompt text* — the model
            chooses tools based on it, so vague descriptions cause bad calls.
        parameters: JSON Schema for the arguments. Derived automatically by the
            decorator; supply it manually when subclassing.
        timeout: Seconds before the call is abandoned.
        requires_approval: When True the agent pauses for human approval before
            invoking it. Use this for anything that spends money or mutates
            state.
        **config: Tool-specific options.

    Attributes:
        description: The model-facing description.
        parameters: The JSON Schema for arguments.
        requires_approval: Whether human-in-the-loop approval is required.

    Example:
        Implementing a stateful tool::

            class SearchTool(Tool):
                def __init__(self, client):
                    super().__init__(
                        name="search",
                        description="Search the product catalogue.",
                        parameters={
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                            "required": ["q"],
                        },
                    )
                    self.client = client

                async def acall(self, **kwargs):
                    return await self.client.search(kwargs["q"])
    """

    kind = "tool"
    provider_name: str = "tool"

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        timeout: float | None = None,
        requires_approval: bool = False,
        **config: Any,
    ) -> None:
        _validate_name(name)
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            timeout=timeout,
            requires_approval=requires_approval,
            **config,
        )
        self.description = description or (self.__doc__ or "").strip().split("\n")[0]
        self.parameters: dict[str, Any] = parameters or {
            "type": "object",
            "properties": {},
        }
        self.timeout = timeout if timeout is not None else DEFAULT_TOOL_TIMEOUT
        self.requires_approval = requires_approval

    # -- provider hooks ---------------------------------------------------
    @abc.abstractmethod
    async def acall(self, **kwargs: Any) -> Any:
        """Execute the tool.

        Args:
            **kwargs: Arguments matching :attr:`parameters`.

        Returns:
            Any JSON-serialisable value. It is rendered to a string for the
            model and kept intact on :attr:`~windlass.core.types.ToolResult.data`.

        Raises:
            Exception: Anything. The agent catches it and reports the failure
                back to the model rather than crashing the run.
        """

    # -- public API -------------------------------------------------------
    def schema(self, *, style: str = "openai") -> dict[str, Any]:
        """Return the tool definition in a provider's format.

        Args:
            style: ``"openai"`` (the default, also used by Groq and Ollama),
                ``"anthropic"``, or ``"gemini"``.

        Returns:
            A dict ready to drop into that provider's request.

        Raises:
            ValueError: For an unknown style.

        Example:
            >>> from windlass.tools import tool
            >>> @tool
            ... def ping() -> str:
            ...     '''Ping.'''
            ...     return "pong"
            >>> ping.schema(style="anthropic")["name"]
            'ping'
        """
        match style:
            case "openai":
                return {
                    "type": "function",
                    "function": {
                        "name": self.name,
                        "description": self.description,
                        "parameters": self.parameters,
                    },
                }
            case "anthropic":
                return {
                    "name": self.name,
                    "description": self.description,
                    "input_schema": self.parameters,
                }
            case "gemini":
                return {
                    "name": self.name,
                    "description": self.description,
                    "parameters": _strip_unsupported(self.parameters),
                }
            case _:
                raise ValueError(
                    f"Unknown tool schema style {style!r}; use openai, anthropic or gemini."
                )

    async def ainvoke(self, call: ToolCall) -> ToolResult:
        """Execute a model-issued tool call and wrap the outcome.

        Failures are captured rather than raised: the agent needs to hand the
        error text back to the model so it can recover, which is impossible if
        the exception unwinds the loop.

        Args:
            call: The call requested by the model.

        Returns:
            A :class:`~windlass.core.types.ToolResult`, with ``is_error`` set
            when the tool raised or timed out.

        Performance:
            Enforces :attr:`timeout`. A hung tool fails the call, not the run.

        Example:
            >>> import asyncio
            >>> from windlass.core.types import ToolCall
            >>> from windlass.tools import tool
            >>> @tool
            ... def echo(text: str) -> str:
            ...     '''Echo text.'''
            ...     return text
            >>> asyncio.run(echo.ainvoke(ToolCall(name="echo", arguments={"text": "hi"}))).content
            'hi'
        """
        import asyncio

        started = time.perf_counter()
        try:
            value = await asyncio.wait_for(self.acall(**call.arguments), timeout=self.timeout)
        except TimeoutError:
            return ToolResult(
                call_id=call.id,
                name=self.name,
                content=f"Tool {self.name!r} timed out after {self.timeout}s.",
                is_error=True,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            self._log.warning("Tool %s failed: %s", self.name, exc)
            return ToolResult(
                call_id=call.id,
                name=self.name,
                content=f"{type(exc).__name__}: {exc}",
                data=None,
                is_error=True,
                duration_ms=(time.perf_counter() - started) * 1000,
            )

        return ToolResult(
            call_id=call.id,
            name=self.name,
            content=render_result(value),
            data=value,
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    def invoke(self, call: ToolCall) -> ToolResult:
        """Blocking :meth:`ainvoke`."""
        return run_sync(self.ainvoke(call))

    async def arun(self, **kwargs: Any) -> ToolResult:
        """Execute the tool directly, outside an agent.

        Args:
            **kwargs: Tool arguments.

        Returns:
            The result.

        Raises:
            ToolExecutionError: When the tool fails. Unlike :meth:`ainvoke`,
                direct invocation raises — there is no model to recover.
        """
        result = await self.ainvoke(ToolCall(name=self.name, arguments=kwargs))
        if result.is_error:
            raise ToolExecutionError(self.name, RuntimeError(result.content))
        return result

    def run(self, **kwargs: Any) -> ToolResult:
        """Blocking :meth:`arun`."""
        return run_sync(self.arun(**kwargs))

    def describe(self) -> dict[str, Any]:
        """Return a JSON-safe summary including the schema."""
        return {
            **super().describe(),
            "description": self.description,
            "parameters": self.parameters,
            "requires_approval": self.requires_approval,
        }

    def __repr__(self) -> str:
        params = ", ".join(self.parameters.get("properties", {}))
        return f"Tool({self.name}({params}))"


def render_result(value: Any) -> str:
    """Render a tool's return value as text for the model.

    Strings pass through; everything else is JSON-encoded when possible and
    ``repr``-ed otherwise. Keeping this in one place means every tool's output
    looks the same to the model regardless of who wrote it.

    Args:
        value: The tool's return value.

    Returns:
        A string safe to place in a ``tool`` message.

    Example:
        >>> render_result({"a": 1})
        '{"a": 1}'
        >>> render_result("plain")
        'plain'
    """
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):  # pragma: no cover - exotic objects
        return repr(value)


def _validate_name(name: str) -> None:
    """Reject tool names the major providers would refuse."""
    import re

    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name or ""):
        from windlass.core.exceptions import ValidationError

        raise ValidationError(
            f"Invalid tool name {name!r}.",
            hint="Use 1-64 characters from [a-zA-Z0-9_-]; providers reject anything else.",
            context={"name": name},
        )


def _strip_unsupported(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove JSON Schema keywords Gemini's function declarations reject."""
    unsupported = {"additionalProperties", "$schema", "default", "examples", "title"}
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key in unsupported:
            continue
        if isinstance(value, dict):
            out[key] = _strip_unsupported(value)
        elif isinstance(value, list):
            out[key] = [_strip_unsupported(v) if isinstance(v, dict) else v for v in value]
        else:
            out[key] = value
    return out
