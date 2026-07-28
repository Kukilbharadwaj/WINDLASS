"""Tools — turning Python functions into agent capabilities.

The whole surface is one decorator::

    from windlass import tool

    @tool
    def get_weather(city: str, units: str = "celsius") -> dict:
        # Docstring summary becomes the model-facing description, and a
        # Google-style argument section becomes the per-parameter descriptions.
        # See :func:`tool` for a fully documented example.
        return {"city": city, "temp": 18, "units": units}

The schema comes from the type hints, the description from the docstring, and
the result is a :class:`~windlass.interfaces.tool.Tool` that any Windlass agent can
bind. Async functions work identically; sync functions are run on a worker thread
so a slow tool never blocks the event loop.

This module also provides :class:`ToolRegistry`, which agents use to hold their
bound tools and execute calls — including parallel execution when a model
requests several at once.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Callable, Sequence
from typing import Any, TypeVar, overload

from windlass.core.concurrency import gather_bounded, to_thread
from windlass.core.exceptions import ToolNotFoundError, ValidationError
from windlass.core.logging import get_logger
from windlass.core.registry import REGISTRY
from windlass.core.types import ToolCall, ToolResult
from windlass.interfaces.tool import Tool, render_result
from windlass.tools.schema import function_schema, parse_docstring, python_type_to_schema

__all__ = [
    "FunctionTool",
    "ToolRegistry",
    "function_schema",
    "parse_docstring",
    "python_type_to_schema",
    "render_result",
    "tool",
]

_F = TypeVar("_F", bound=Callable[..., Any])
_log = get_logger(__name__)


class FunctionTool(Tool):
    """A :class:`~windlass.interfaces.tool.Tool` backed by a Python function.

    Normally produced by the :func:`tool` decorator rather than constructed
    directly.

    Args:
        fn: The function to wrap. May be sync or async.
        name: Tool name. Defaults to the function's name.
        description: Model-facing description. Defaults to the docstring summary.
        parameters: JSON Schema override. Derived from the signature by default.
        timeout: Seconds before the call is abandoned.
        requires_approval: Pause for human approval before running.
        **config: Forwarded to :class:`~windlass.interfaces.tool.Tool`.

    Attributes:
        fn: The wrapped function, still callable directly.
        is_async: Whether the function is a coroutine function.

    Example:
        >>> def add(a: int, b: int) -> int:
        ...     '''Add two integers.'''
        ...     return a + b
        >>> t = FunctionTool(add)
        >>> t.run(a=2, b=3).data
        5
    """

    provider_name = "function"

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        timeout: float | None = None,
        requires_approval: bool = False,
        **config: Any,
    ) -> None:
        summary, _ = parse_docstring(inspect.getdoc(fn))
        super().__init__(
            name=name or fn.__name__,
            description=description or summary or f"Call {fn.__name__}.",
            parameters=parameters or function_schema(fn),
            timeout=timeout,
            requires_approval=requires_approval,
            **config,
        )
        self.fn = fn
        self.is_async = asyncio.iscoroutinefunction(fn)
        self.__doc__ = fn.__doc__
        self.__name__ = getattr(fn, "__name__", self.name)

    async def acall(self, **kwargs: Any) -> Any:
        """Invoke the wrapped function.

        Sync functions run on a worker thread, so a blocking HTTP call inside a
        tool cannot stall the agent's event loop.

        Args:
            **kwargs: Arguments matching the schema.

        Returns:
            The function's return value.

        Raises:
            ValidationError: When a required argument is missing.
            Exception: Anything the function raises.
        """
        self._validate(kwargs)
        if self.is_async:
            return await self.fn(**kwargs)
        return await to_thread(self.fn, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the underlying function directly, bypassing the tool machinery.

        This keeps a decorated function usable as an ordinary function, which
        matters for testing and for reuse outside an agent.
        """
        return self.fn(*args, **kwargs)

    def _validate(self, arguments: dict[str, Any]) -> None:
        """Check required arguments before invoking the function.

        Models occasionally omit a required argument. Catching it here produces
        a clear error the agent can hand back, instead of an opaque ``TypeError``.
        """
        required = self.parameters.get("required") or []
        missing = [key for key in required if key not in arguments]
        if missing:
            raise ValidationError(
                f"Tool {self.name!r} was called without required argument(s): "
                f"{', '.join(missing)}.",
                context={"tool": self.name, "missing": missing},
            )

    def native(self) -> Any:
        """Return the wrapped function."""
        return self.fn


@overload
def tool(fn: _F) -> FunctionTool: ...


@overload
def tool(
    *,
    name: str | None = ...,
    description: str | None = ...,
    parameters: dict[str, Any] | None = ...,
    timeout: float | None = ...,
    requires_approval: bool = ...,
    register: bool = ...,
) -> Callable[[_F], FunctionTool]: ...


def tool(
    fn: _F | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
    timeout: float | None = None,
    requires_approval: bool = False,
    register: bool = False,
) -> Any:
    """Turn a Python function into an agent-callable tool.

    Works bare (``@tool``) or with arguments (``@tool(timeout=5)``).

    Args:
        fn: The function, when used bare.
        name: Tool name shown to the model. Defaults to the function name.
        description: Model-facing description. Defaults to the docstring summary.
            **Write a good one** — it is how the model decides when to call this.
        parameters: JSON Schema override. Derived from type hints by default.
        timeout: Seconds before the call is abandoned.
        requires_approval: Pause the agent for human approval before running.
            Use this for anything that spends money or changes state.
        register: Also add the tool to the global registry under ``name``, so it
            can be referenced as a string.

    Returns:
        A :class:`FunctionTool`, or a decorator producing one.

    Raises:
        ValidationError: If the resulting name is not provider-legal.

    Note:
        The docstring is not decoration. Its summary becomes the model-facing
        description, and its Google-style argument section becomes the
        per-parameter descriptions in the JSON schema. Both are prompt text — the
        model decides *when* to call the tool, and *what to pass*, from them.

    Example:
        Bare::

            @tool
            def search(query: str) -> list[str]:
                # Docstring: "Search the product catalogue." plus an argument
                # section describing `query`.
                return db.search(query)

        Configured::

            @tool(name="charge_card", requires_approval=True, timeout=30)
            async def charge(amount_cents: int, customer_id: str) -> dict:
                # Approval-gated: the agent pauses before this ever runs.
                return await payments.charge(customer_id, amount_cents)
    """

    def wrap(target: _F) -> FunctionTool:
        built = FunctionTool(
            target,
            name=name,
            description=description,
            parameters=parameters,
            timeout=timeout,
            requires_approval=requires_approval,
        )
        if register:
            REGISTRY.register(
                "tool",
                built.name,
                lambda **_: built,
                description=built.description,
                origin="user",
                override=True,
            )
        return built

    if fn is not None:
        return wrap(fn)
    return wrap


class ToolRegistry:
    """The set of tools bound to one agent.

    Holds tools by name, renders provider-specific schemas, and executes calls —
    in parallel when the model requests several at once, which is where a lot of
    agent latency quietly hides.

    Args:
        tools: Initial tools to bind.

    Attributes:
        tools: Bound tools, keyed by name.

    Example:
        >>> @tool
        ... def double(x: int) -> int:
        ...     '''Double a number.'''
        ...     return x * 2
        >>> registry = ToolRegistry([double])
        >>> registry.execute(ToolCall(name="double", arguments={"x": 4})).data
        8
    """

    def __init__(self, tools: Sequence[Tool] | None = None) -> None:
        self.tools: dict[str, Tool] = {}
        self._lock = threading.RLock()
        for item in tools or ():
            self.add(item)

    # -- membership -------------------------------------------------------
    def add(self, item: Tool | Callable[..., Any]) -> Tool:
        """Bind a tool.

        Args:
            item: A :class:`~windlass.interfaces.tool.Tool`, or a plain function
                which is wrapped automatically.

        Returns:
            The bound tool.

        Raises:
            TypeError: If ``item`` is neither a tool nor callable.
        """
        if isinstance(item, Tool):
            built = item
        elif callable(item):
            built = FunctionTool(item)
        else:
            raise TypeError(f"Expected a Tool or a callable, got {type(item).__name__}.")
        with self._lock:
            if built.name in self.tools and self.tools[built.name] is not built:
                _log.debug("Replacing already-bound tool %r.", built.name)
            self.tools[built.name] = built
        return built

    def extend(self, items: Sequence[Tool | Callable[..., Any]]) -> ToolRegistry:
        """Bind several tools and return ``self``."""
        for item in items:
            self.add(item)
        return self

    def remove(self, name: str) -> None:
        """Unbind a tool. No-op when it is not bound."""
        with self._lock:
            self.tools.pop(name, None)

    def get(self, name: str) -> Tool:
        """Return a bound tool.

        Args:
            name: Tool name.

        Returns:
            The tool.

        Raises:
            ToolNotFoundError: When no tool of that name is bound. The message
                lists what is available.
        """
        with self._lock:
            found = self.tools.get(name)
            if found is None:
                raise ToolNotFoundError(name, list(self.tools))
            return found

    def names(self) -> list[str]:
        """Return the bound tool names, sorted."""
        with self._lock:
            return sorted(self.tools)

    # -- schemas ----------------------------------------------------------
    def schemas(self, style: str = "openai") -> list[dict[str, Any]]:
        """Return every bound tool's definition in a provider's format.

        Args:
            style: ``"openai"``, ``"anthropic"`` or ``"gemini"``.

        Returns:
            Tool definitions, or ``[]`` when nothing is bound — which is what
            lets an agent pass ``tools=None`` cleanly.
        """
        with self._lock:
            return [t.schema(style=style) for t in self.tools.values()]

    # -- execution --------------------------------------------------------
    async def aexecute(self, call: ToolCall) -> ToolResult:
        """Execute one tool call.

        An unknown tool produces an error :class:`~windlass.core.types.ToolResult`
        rather than an exception, so the agent can tell the model it hallucinated
        a tool and let it recover.

        Args:
            call: The call to execute.

        Returns:
            The result.
        """
        try:
            target = self.get(call.name)
        except ToolNotFoundError as exc:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=str(exc),
                is_error=True,
            )
        return await target.ainvoke(call)

    def execute(self, call: ToolCall) -> ToolResult:
        """Blocking :meth:`aexecute`."""
        from windlass.core.concurrency import run_sync

        return run_sync(self.aexecute(call))

    async def aexecute_many(
        self, calls: Sequence[ToolCall], *, parallel: bool = True, limit: int = 8
    ) -> list[ToolResult]:
        """Execute several tool calls.

        Args:
            calls: The calls to execute.
            parallel: Run them concurrently. Models routinely request three or
                four independent lookups in one turn; running those serially
                triples the user's wait for no reason.
            limit: Maximum concurrent executions.

        Returns:
            Results in the same order as ``calls``.

        Example:
            >>> import asyncio
            >>> @tool
            ... def echo(text: str) -> str:
            ...     '''Echo text.'''
            ...     return text
            >>> registry = ToolRegistry([echo])
            >>> calls = [ToolCall(name="echo", arguments={"text": t}) for t in "ab"]
            >>> [r.content for r in asyncio.run(registry.aexecute_many(calls))]
            ['a', 'b']
        """
        if not calls:
            return []
        if not parallel or len(calls) == 1:
            return [await self.aexecute(call) for call in calls]
        return await gather_bounded([self.aexecute(c) for c in calls], limit=limit)

    def execute_many(self, calls: Sequence[ToolCall], *, parallel: bool = True) -> list[ToolResult]:
        """Blocking :meth:`aexecute_many`."""
        from windlass.core.concurrency import run_sync

        return run_sync(self.aexecute_many(calls, parallel=parallel))

    def needs_approval(self, calls: Sequence[ToolCall]) -> list[ToolCall]:
        """Return the calls whose tools require human approval.

        Args:
            calls: Calls the model requested.

        Returns:
            The subset that must be approved before running.
        """
        approvals: list[ToolCall] = []
        for call in calls:
            found = self.tools.get(call.name)
            if found is not None and found.requires_approval:
                approvals.append(call)
        return approvals

    def describe(self) -> list[dict[str, Any]]:
        """Return a JSON-safe summary of every bound tool."""
        with self._lock:
            return [t.describe() for t in self.tools.values()]

    def __len__(self) -> int:
        return len(self.tools)

    def __contains__(self, name: str) -> bool:
        return name in self.tools

    def __iter__(self) -> Any:
        return iter(self.tools.values())

    def __repr__(self) -> str:
        return f"ToolRegistry({', '.join(sorted(self.tools))})"
