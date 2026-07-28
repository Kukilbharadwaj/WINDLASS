"""The fluent agent builder.

::

    agent = (
        Windlass.agent()
        .llm("gpt-4o")
        .tool(weather_tool)
        .tool(search_tool)
        .mcp(command="npx",
             args=["-y", "@modelcontextprotocol/server-filesystem", "."])
        .memory()
        .guardrails()
    )

    response = agent.run("Summarise my PDF documents")

Like the RAG builder, nothing is constructed until first use, every argument
accepts a name, an instance or a factory, and the builder forwards the runtime's
methods so there is no mandatory ``.build()`` step.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Self

from windlass.agent.runtime import AgentRuntime
from windlass.core.container import Container, root_container
from windlass.core.exceptions import ConfigurationError
from windlass.core.logging import get_logger
from windlass.core.types import AgentResponse, StreamEvent
from windlass.tools import ToolRegistry

__all__ = ["AgentBuilder"]

_log = get_logger(__name__)


class AgentBuilder:
    """Fluent builder for an agent.

    Args:
        container: Dependency container. Defaults to a child of the process root.

    Example:
        Level 1 — a working agent in two lines, no dependencies::

            agent = Windlass.agent().llm("fake", responses=["Hello!"])
            print(agent.run("Say hello"))

        Level 2 — a real agent::

            agent = (Windlass.agent()
                     .llm("anthropic", model="claude-sonnet-4-5")
                     .tool(search).tool(calculator)
                     .system("You are a research assistant. Cite your sources.")
                     .memory("summary")
                     .guardrails(pii=True, on_violation="redact")
                     .checkpoint("sqlite", path="./state.db")
                     .max_iterations(15)
                     .observe("langfuse"))

        Level 3 — a graph you control::

            agent = Windlass.agent().llm("openai").tool(search).graph()
            graph = agent.native_graph()
            graph.add_node("critic", critic_fn)
            graph.add_edge("tools", "critic")
            agent.recompile()
    """

    def __init__(self, container: Container | None = None) -> None:
        self.container = container or root_container().scope()
        self._specs: dict[str, tuple[Any, dict[str, Any]]] = {}
        self._tools: list[Any] = []
        self._mcp_specs: list[tuple[Any, dict[str, Any]]] = []
        self._sub_agents: dict[str, Any] = {}
        self._descriptions: dict[str, str] = {}
        self._options: dict[str, Any] = {
            "system_prompt": None,
            "max_iterations": 10,
            "parallel_tools": True,
            "require_approval": False,
            "max_tool_call_retries": 2,
            "name": "agent",
        }
        self._use_graph = False
        self._runtime: AgentRuntime | None = None

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------
    def llm(self, spec: Any = None, /, **config: Any) -> Self:
        """Choose the reasoning model.

        Args:
            spec: Registry name (``"openai"``, ``"anthropic"``, ``"gemini"``,
                ``"groq"``, ``"ollama"``, ``"fake"``), an
                :class:`~windlass.interfaces.llm.LLM`, or a factory. A bare model
                id like ``"gpt-4o"`` is also accepted and mapped to its provider.
            **config: Provider options — ``model``, ``temperature``, ``api_key``, ...

        Returns:
            ``self``.

        Example:
            >>> from windlass import Windlass
            >>> _ = Windlass.agent().llm("gpt-4o-mini", temperature=0)
        """
        if isinstance(spec, str):
            provider, model = _split_model(spec)
            if model and "model" not in config:
                config = {**config, "model": model}
            spec = provider
        return self._set("llm", spec, config)

    def tool(self, *tools: Any) -> Self:
        """Bind one or more tools.

        Args:
            *tools: :class:`~windlass.interfaces.tool.Tool` instances, functions
                decorated with :func:`~windlass.tools.tool`, plain functions
                (wrapped automatically), or registry names.

        Returns:
            ``self``.

        Raises:
            ConfigurationError: When an argument cannot be interpreted as a tool.

        Example:
            >>> from windlass import Windlass, tool
            >>> @tool
            ... def now() -> str:
            ...     '''Return the current time.'''
            ...     return "12:00"
            >>> _ = Windlass.agent().tool(now)
        """
        from windlass.interfaces.tool import Tool

        for item in tools:
            if item is None:
                continue
            # A Tool subclass is not necessarily callable — only FunctionTool
            # forwards __call__ — so the isinstance check has to come first.
            if not (isinstance(item, Tool | str) or callable(item)):
                raise ConfigurationError(
                    f"Cannot use {type(item).__name__} as a tool.",
                    hint="Pass a @tool-decorated function, a Tool instance, a plain "
                    "callable, or a registered tool name.",
                )
            self._tools.append(item)
        return self._invalidate()

    def tools(self, tools: Sequence[Any]) -> Self:
        """Bind a sequence of tools. See :meth:`tool`."""
        return self.tool(*tools)

    def mcp(self, spec: Any = None, /, **config: Any) -> Self:
        """Connect an MCP server and bind its tools.

        Call it repeatedly to connect several servers; their tools are namespaced
        by server name so identically-named tools do not collide.

        Args:
            spec: Registry name (``"fastmcp"``, ``"static"``), an
                :class:`~windlass.interfaces.mcp.MCPClient`, or a factory. Omit it
                for the FastMCP client.
            **config: Transport options — ``command``, ``args``, ``url``, ``env``,
                ``server``.

        Returns:
            ``self``.

        Example:
            >>> from windlass import Windlass
            >>> _ = Windlass.agent().mcp(server="fs", command="npx",
            ...     args=["-y", "@modelcontextprotocol/server-filesystem", "."])
        """
        self._mcp_specs.append((spec if spec is not None else "fastmcp", config))
        return self._invalidate()

    def memory(self, spec: Any = None, /, **config: Any) -> Self:
        """Attach conversation memory.

        Args:
            spec: Registry name (``"window"``, ``"buffer"``, ``"summary"``,
                ``"vector"``, ``"composite"``), a
                :class:`~windlass.interfaces.memory.Memory`, or a factory. Omit it
                for a sliding window.
            **config: Memory options.

        Returns:
            ``self``.
        """
        return self._set("memory", spec if spec is not None else "window", config)

    def guardrails(self, spec: Any = None, /, **config: Any) -> Self:
        """Enable input and output guardrails.

        Args:
            spec: Registry name (``"rules"``, ``"nemo"``), a
                :class:`~windlass.interfaces.guardrail.Guardrail`, or a factory.
                Omit it for the dependency-free rule guardrail.
            **config: Policy options.

        Returns:
            ``self``.
        """
        return self._set("guardrail", spec if spec is not None else "rules", config)

    def observe(self, spec: Any = None, /, **config: Any) -> Self:
        """Enable tracing.

        Args:
            spec: Registry name (``"console"``, ``"langfuse"``, ``"langsmith"``,
                ``"memory"``), a :class:`~windlass.interfaces.tracer.Tracer`, or a
                factory.
            **config: Tracer options.

        Returns:
            ``self``.
        """
        return self._set("tracer", spec if spec is not None else "console", config)

    def checkpoint(self, spec: Any = None, /, **config: Any) -> Self:
        """Enable durable state, resume and human-in-the-loop approval.

        Args:
            spec: Registry name (``"memory"``, ``"sqlite"``), a
                :class:`~windlass.agent.checkpoint.Checkpointer`, or a factory.
                Omit it for the in-process store.
            **config: Checkpointer options — ``path``, ``max_history``.

        Returns:
            ``self``.

        Note:
            Approval interrupts require a checkpointer, since a paused run has to
            be stored somewhere before it can be resumed.
        """
        return self._set("checkpointer", spec if spec is not None else "memory", config)

    def agent(self, name: str, worker: Any, description: str = "") -> Self:
        """Add a specialist, turning this into a supervisor.

        Args:
            name: The name the supervisor uses to delegate.
            worker: An agent, builder, or anything with ``arun``.
            description: What the specialist is good at. The supervisor routes on
                this text.

        Returns:
            ``self``.

        Example:
            >>> from windlass import Windlass
            >>> _ = (Windlass.agent().llm("fake")
            ...      .agent("researcher", Windlass.agent().llm("fake"),
            ...             "Finds and summarises source material."))
        """
        self._sub_agents[name] = worker
        if description:
            self._descriptions[name] = description
        return self._invalidate()

    # ------------------------------------------------------------------
    # Behaviour
    # ------------------------------------------------------------------
    def system(self, prompt: str) -> Self:
        """Set the system prompt.

        Args:
            prompt: Instructions prepended to every run.

        Returns:
            ``self``.
        """
        self._options["system_prompt"] = prompt
        return self._invalidate()

    def name(self, value: str) -> Self:
        """Name the agent, for traces and multi-agent routing."""
        self._options["name"] = value
        return self._invalidate()

    def max_iterations(self, limit: int) -> Self:
        """Set the reason/act step budget.

        Args:
            limit: Maximum cycles before
                :class:`~windlass.core.exceptions.MaxIterationsExceeded`.

        Returns:
            ``self``.

        Raises:
            ValueError: If ``limit`` is not positive.
        """
        if limit <= 0:
            raise ValueError("max_iterations must be positive")
        self._options["max_iterations"] = limit
        return self._invalidate()

    def tool_call_retries(self, limit: int) -> Self:
        """Set how often a run may recover from an unparseable tool call.

        Models occasionally emit a tool call the provider cannot parse — most
        often by nesting one call inside another's arguments, which the
        tool-calling protocol has no way to express. Windlass shows the model the
        problem and lets it retry, the same way it handles an invented tool name.

        Args:
            limit: Maximum corrections per run. Each one costs an iteration from
                :meth:`max_iterations`, so a model that never recovers still
                terminates. ``0`` fails on the first malformed call.

        Returns:
            ``self``.

        Raises:
            ValueError: If ``limit`` is negative.

        Example:
            >>> from windlass import Windlass
            >>> _ = Windlass.agent().llm("fake").tool_call_retries(3)
        """
        if limit < 0:
            raise ValueError("tool_call_retries must not be negative")
        self._options["max_tool_call_retries"] = limit
        return self._invalidate()

    def parallel_tools(self, enabled: bool = True) -> Self:
        """Control whether simultaneous tool calls run concurrently.

        Args:
            enabled: False forces serial execution — occasionally necessary when
                tools share mutable state.

        Returns:
            ``self``.
        """
        self._options["parallel_tools"] = enabled
        return self._invalidate()

    def human_in_the_loop(self, enabled: bool = True) -> Self:
        """Require human approval before *every* tool call.

        Individual tools can require approval on their own with
        ``@tool(requires_approval=True)``; this turns it on globally.

        Args:
            enabled: True to pause before every tool call.

        Returns:
            ``self``.

        Note:
            Implies a checkpointer, since a paused run has to be stored. One is
            enabled automatically if you have not chosen one.
        """
        self._options["require_approval"] = enabled
        if enabled and "checkpointer" not in self._specs:
            self.checkpoint()
        return self._invalidate()

    def graph(self, enabled: bool = True) -> Self:
        """Use the LangGraph runtime instead of the built-in loop.

        Args:
            enabled: True to compile the agent into a LangGraph state machine.

        Returns:
            ``self``.

        Raises:
            MissingDependencyError: At build time, when ``langgraph`` is missing.
        """
        self._use_graph = enabled
        return self._invalidate()

    def bind(self, key: str, factory: Any) -> Self:
        """Bind an arbitrary dependency into this builder's container."""
        if callable(factory):
            self.container.bind(key, factory)
        else:
            self.container.bind_instance(key, factory)
        return self._invalidate()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self) -> AgentRuntime:
        """Resolve every component and construct the runtime.

        Called automatically on first use.

        Returns:
            An :class:`~windlass.agent.runtime.AgentRuntime`, a
            :class:`~windlass.agent.graph.LangGraphRuntime`, or a
            :class:`~windlass.agent.supervisor.Supervisor` — whichever the
            configuration describes.

        Raises:
            ConfigurationError: When a component cannot be constructed.
            MissingDependencyError: When a chosen provider's extra is missing.
        """
        if self._runtime is not None:
            return self._runtime

        from windlass.core.config import settings

        cfg = settings()
        llm = self._resolve("llm", cfg.default_llm)
        registry = ToolRegistry()

        for item in self._tools:
            registry.add(self.container.component("tool", item) if isinstance(item, str) else item)

        for spec, config in self._mcp_specs:
            for remote in self._connect_mcp(spec, config):
                registry.add(remote)

        common: dict[str, Any] = {
            "llm": llm,
            "memory": self._resolve_memory(),
            "guardrail": self._resolve_optional("guardrail"),
            "checkpointer": self._resolve_optional("checkpointer"),
            "tracer": self._resolve_optional("tracer"),
            **self._options,
        }

        if self._sub_agents:
            from windlass.agent.supervisor import Supervisor

            self._runtime = Supervisor(
                agents={k: _built(v) for k, v in self._sub_agents.items()},
                descriptions=self._descriptions,
                tools=list(registry),
                **common,
            )
        elif self._use_graph:
            from windlass.agent.graph import LangGraphRuntime

            self._runtime = LangGraphRuntime(tools=registry, **common)
        else:
            self._runtime = AgentRuntime(tools=registry, **common)

        return self._runtime

    @property
    def runtime(self) -> AgentRuntime:
        """The built runtime, constructing it on first access."""
        return self.build()

    # ------------------------------------------------------------------
    # Runtime delegation
    # ------------------------------------------------------------------
    def run(self, prompt: Any, **kwargs: Any) -> AgentResponse:
        """Run the agent. See :meth:`~windlass.agent.runtime.AgentRuntime.run`."""
        return self.build().run(prompt, **kwargs)

    async def arun(self, prompt: Any, **kwargs: Any) -> AgentResponse:
        """Async :meth:`run`."""
        return await self.build().arun(prompt, **kwargs)

    def stream(self, prompt: Any, **kwargs: Any) -> Iterator[StreamEvent]:
        """Stream the run. See :meth:`~windlass.agent.runtime.AgentRuntime.stream`."""
        return self.build().stream(prompt, **kwargs)

    def astream(self, prompt: Any, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        """Async :meth:`stream`."""
        return self.build().astream(prompt, **kwargs)

    def astream_text(self, prompt: Any, **kwargs: Any) -> AsyncIterator[str]:
        """Stream only the assistant's text."""
        return self.build().astream_text(prompt, **kwargs)

    def resume(self, thread_id: str, **kwargs: Any) -> AgentResponse:
        """Resume an interrupted run."""
        return self.build().resume(thread_id, **kwargs)

    async def aresume(self, thread_id: str, **kwargs: Any) -> AgentResponse:
        """Async :meth:`resume`."""
        return await self.build().aresume(thread_id, **kwargs)

    def pending_approvals(self, thread_id: str) -> list[Any]:
        """Return the tool calls awaiting approval on a thread."""
        return self.build().pending_approvals(thread_id)

    def state(self, thread_id: str) -> dict[str, Any] | None:
        """Return the latest checkpoint for a thread."""
        return self.build().state(thread_id)

    def history(self, thread_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent checkpoints for a thread."""
        return self.build().history(thread_id, limit)

    def broadcast(self, task: str) -> dict[str, AgentResponse]:
        """Run every specialist on the same task (supervisor only).

        Raises:
            ConfigurationError: When no specialists are configured.
        """
        runtime = self.build()
        caller = getattr(runtime, "broadcast", None)
        if caller is None:
            raise ConfigurationError(
                "broadcast() needs specialist agents.",
                hint="Add them with .agent('name', worker).",
            )
        return caller(task)

    def pipeline(self, task: str, order: list[str]) -> AgentResponse:
        """Chain specialists in sequence (supervisor only).

        Raises:
            ConfigurationError: When no specialists are configured.
        """
        runtime = self.build()
        caller = getattr(runtime, "pipeline", None)
        if caller is None:
            raise ConfigurationError(
                "pipeline() needs specialist agents.",
                hint="Add them with .agent('name', worker).",
            )
        return caller(task, order)

    # -- Level 3 ----------------------------------------------------------
    def native_llm(self) -> Any:
        """Return the model provider's own client."""
        return self.build().native_llm()

    def native_graph(self) -> Any:
        """Return the LangGraph ``StateGraph``.

        Raises:
            AgentError: When the agent was not built with :meth:`graph`.
        """
        return self.build().native_graph()

    def recompile(self) -> Any:
        """Recompile the graph after editing it.

        Raises:
            ConfigurationError: When the agent is not graph-backed.
        """
        runtime = self.build()
        recompiler = getattr(runtime, "recompile", None)
        if recompiler is None:
            raise ConfigurationError(
                "recompile() only applies to graph-backed agents.",
                hint="Build the agent with .graph().",
            )
        return recompiler()

    def describe(self) -> dict[str, Any]:
        """Return a JSON-safe description of the agent."""
        return self.build().describe()

    def close(self) -> None:
        """Release the model's and MCP clients' resources."""
        if self._runtime is not None:
            self._runtime.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _set(self, kind: str, spec: Any, config: dict[str, Any]) -> Self:
        self._specs[kind] = (spec, config)
        return self._invalidate()

    def _invalidate(self) -> Self:
        self._runtime = None
        return self

    def _resolve(self, kind: str, default: str) -> Any:
        """Resolve a required component.

        Precedence is explicit builder call, then a container binding, then the
        configured default — see the matching note in
        :meth:`~windlass.rag.builder.RAGBuilder._resolve`.
        """
        spec, config = self._specs.get(kind, (None, {}))
        if spec is None and not config and self.container.has(kind):
            return self.container.component(kind)
        return self.container.component(kind, spec if spec is not None else default, **config)

    def _resolve_optional(self, kind: str) -> Any:
        """Resolve a component the user asked for, or one the container supplies."""
        if kind not in self._specs:
            return self.container.component(kind) if self.container.has(kind) else None
        spec, config = self._specs[kind]
        return self.container.component(kind, spec, **config)

    def _resolve_memory(self) -> Any:
        """Resolve memory, injecting an embedder for the semantic backends."""
        if "memory" not in self._specs:
            return None
        spec, config = self._specs["memory"]
        options = dict(config)
        if spec in {"vector", "long-term", "longterm", "semantic"} and "embedder" not in options:
            from windlass.core.config import settings

            options["embedder"] = self.container.component(
                "embedding", settings().default_embedding
            )
        return self.container.component("memory", spec, **options)

    def _connect_mcp(self, spec: Any, config: dict[str, Any]) -> list[Any]:
        """Connect one MCP server and return its tools.

        A server that is unreachable logs a warning and contributes no tools,
        rather than preventing the agent from starting. An agent that works with
        four of its five tool sources is more useful than one that refuses to run.
        """
        from windlass.core.concurrency import run_sync
        from windlass.interfaces.mcp import MCPClient

        options = dict(config)
        namespace = len(self._mcp_specs) > 1

        if isinstance(spec, MCPClient):
            # An already-constructed client cannot take construction config, so
            # the namespacing decision is applied to the live object instead.
            client = spec
            if namespace:
                client.namespace = True
        else:
            if namespace:
                options.setdefault("namespace", True)
            client = self.container.component("mcp", spec, **options)

        try:
            return list(run_sync(client.alist_tools()))
        except Exception as exc:
            _log.warning(
                "MCP server %r is unavailable; its tools are not bound: %s",
                getattr(client, "server", spec),
                exc,
            )
            return []

    def __repr__(self) -> str:
        runtime = "graph" if self._use_graph else ("supervisor" if self._sub_agents else "builtin")
        return (
            f"AgentBuilder(runtime={runtime}, tools={len(self._tools)}, "
            f"mcp={len(self._mcp_specs)})"
        )


#: Bare model ids mapped to the provider that serves them, so ``.llm("gpt-4o")``
#: works as well as ``.llm("openai", model="gpt-4o")``.
_MODEL_PREFIXES: tuple[tuple[str, str], ...] = (
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("claude", "anthropic"),
    ("gemini", "gemini"),
    ("llama", "groq"),
    ("mixtral", "groq"),
    ("gemma", "groq"),
    ("qwen", "ollama"),
    ("mistral", "ollama"),
    ("phi", "ollama"),
)


def _split_model(spec: str) -> tuple[str, str]:
    """Split a model spec into ``(provider, model)``.

    Args:
        spec: A provider name (``"openai"``), a bare model id (``"gpt-4o"``), or
            a qualified ``provider/model`` string.

    Returns:
        A ``(provider, model)`` pair; ``model`` is ``""`` when only a provider
        was given.

    Example:
        >>> _split_model("openai")
        ('openai', '')
        >>> _split_model("gpt-4o")
        ('openai', 'gpt-4o')
        >>> _split_model("ollama/llama3.2")
        ('ollama', 'llama3.2')
    """
    if "/" in spec:
        provider, _, model = spec.partition("/")
        return provider, model

    from windlass.core.registry import REGISTRY

    if REGISTRY.has("llm", spec):
        return spec, ""

    lowered = spec.lower()
    for prefix, provider in _MODEL_PREFIXES:
        if lowered.startswith(prefix):
            return provider, spec
    return spec, ""


def _built(candidate: Any) -> Any:
    """Return a runnable agent from a builder or a runtime."""
    builder = getattr(candidate, "build", None)
    return builder() if callable(builder) else candidate
