"""LangGraph-backed agent runtime.

When you need more than a loop — conditional routing, subgraphs, explicit state
machines, LangGraph's own checkpointers — build the agent with ``.graph()``.
This runtime compiles the same configuration into a real ``StateGraph`` and
hands it to you via :meth:`LangGraphRuntime.native_graph`.

Install with::

    pip install "windlass[agent]"

The Level 3 promise in full::

    agent = Windlass.agent().llm("openai").tool(search).graph()

    graph = agent.native_graph()          # the uncompiled StateGraph
    graph.add_node("review", review_fn)
    graph.add_edge("tools", "review")
    graph.add_conditional_edges("review", route_fn, {"retry": "agent", "done": END})
    agent.recompile()                     # pick up the changes

Example:
    >>> from windlass import Windlass                                # doctest: +SKIP
    >>> agent = Windlass.agent().llm("openai").graph()              # doctest: +SKIP
    >>> agent.run("hello").output                                  # doctest: +SKIP
"""

from __future__ import annotations

import contextlib
import operator
import time
from collections.abc import AsyncIterator, Sequence
from typing import Annotated, Any, TypedDict

from windlass.agent.runtime import AgentRuntime
from windlass.core.exceptions import AgentError
from windlass.core.lazy import require
from windlass.core.types import (
    AgentResponse,
    AgentStep,
    Message,
    Role,
    StreamEvent,
    ToolCall,
    Usage,
    new_id,
)

__all__ = ["LangGraphRuntime"]


class _AgentState(TypedDict):
    """Graph state carried between nodes.

    Defined at module scope on purpose. This module uses ``from __future__
    import annotations``, so these annotations are strings, and LangGraph
    resolves them with ``get_type_hints(State)`` — which evaluates them against
    the *defining module's* globals. A ``State`` declared inside the builder
    method, with ``Annotated`` imported into that method's local scope, raises
    ``NameError: name 'Annotated' is not defined`` at graph construction.
    """

    messages: Annotated[list[Any], operator.add]
    steps: Annotated[list[Any], operator.add]


class LangGraphRuntime(AgentRuntime):
    """An agent whose execution is a compiled LangGraph state machine.

    Behaves exactly like :class:`~windlass.agent.runtime.AgentRuntime` — same
    constructor, same ``run``/``stream``/``resume`` — but the loop is a graph you
    can inspect, extend and visualise.

    Args:
        **kwargs: Everything :class:`~windlass.agent.runtime.AgentRuntime` accepts,
            including ``llm``, ``tools``, ``memory``, ``guardrail``,
            ``checkpointer`` and ``tracer``.

    Raises:
        MissingDependencyError: When ``langgraph`` is not installed.

    Note:
        Passing ``checkpointer`` enables both savers: LangGraph's own keeps the
        graph state, and the Windlass one keeps the snapshot that makes
        :meth:`~windlass.agent.runtime.AgentRuntime.resume` behave identically
        across the two runtimes.

    Note:
        The graph is compiled lazily on first run. After mutating the graph
        returned by :meth:`native_graph`, call :meth:`recompile` so the change
        takes effect.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._langgraph = require("langgraph.graph", extra="agent", feature="The LangGraph runtime")
        self._graph: Any = None
        self._compiled: Any = None
        self._saver: Any = None

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def native_graph(self) -> Any:
        """Return the uncompiled ``StateGraph`` for direct manipulation.

        Returns:
            The LangGraph ``StateGraph``, with the ``agent`` and ``tools`` nodes
            already wired.

        Example:
            >>> graph = agent.native_graph()             # doctest: +SKIP
            >>> graph.add_node("audit", audit_fn)        # doctest: +SKIP
            >>> agent.recompile()                        # doctest: +SKIP
        """
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def compiled_graph(self) -> Any:
        """Return the compiled graph, compiling it if necessary."""
        if self._compiled is None:
            self._compiled = self._compile()
        return self._compiled

    def recompile(self) -> Any:
        """Recompile after mutating the graph.

        Returns:
            The freshly compiled graph.
        """
        self._compiled = self._compile()
        return self._compiled

    def _build_graph(self) -> Any:
        """Assemble the standard agent graph: think → act → think."""
        state_graph = self._langgraph.StateGraph
        end = self._langgraph.END

        async def think(state: _AgentState) -> dict[str, Any]:
            """Call the model with the current transcript."""
            messages = [_to_windlass(m) for m in state["messages"]]
            completion = await self.llm.acomplete(messages, tools=self._schemas())
            step = AgentStep(
                index=len(state.get("steps", [])),
                thought=completion.content,
                tool_calls=completion.tool_calls,
                usage=completion.usage,
                node="agent",
            )
            return {"messages": [completion.to_message()], "steps": [step]}

        async def act(state: _AgentState) -> dict[str, Any]:
            """Execute whatever tools the model asked for."""
            last = _to_windlass(state["messages"][-1])
            results = await self._act(last.tool_calls)
            step = AgentStep(
                index=len(state.get("steps", [])),
                tool_calls=last.tool_calls,
                tool_results=results,
                node="tools",
            )
            return {
                "messages": [Message.tool(r) for r in results],
                "steps": [step],
            }

        def route(state: _AgentState) -> str:
            """Continue to the tools node when the model requested a call."""
            last = _to_windlass(state["messages"][-1])
            if last.tool_calls:
                return "tools"
            return end

        graph = state_graph(_AgentState)
        graph.add_node("agent", think)
        graph.add_node("tools", act)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", route, {"tools": "tools", end: end})
        graph.add_edge("tools", "agent")
        return graph

    def _compile(self) -> Any:
        """Compile the graph, attaching a LangGraph checkpointer when enabled."""
        graph = self.native_graph()
        kwargs: dict[str, Any] = {}
        if self.checkpointer is not None:
            kwargs["checkpointer"] = self._langgraph_saver()
        if self.require_approval or self.tools.needs_approval(
            [ToolCall(name=n) for n in self.tools.names()]
        ):
            kwargs["interrupt_before"] = ["tools"]
        try:
            return graph.compile(**kwargs)
        except Exception as exc:
            raise AgentError(
                f"Could not compile the agent graph: {exc}",
                hint="If you edited the graph, check every node name referenced by "
                "an edge exists.",
            ) from exc

    def _langgraph_saver(self) -> Any:
        """Return a LangGraph checkpoint saver."""
        if self._saver is None:
            memory_module = require(
                "langgraph.checkpoint.memory", extra="agent", feature="Agent checkpointing"
            )
            self._saver = memory_module.MemorySaver()
        return self._saver

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    async def arun(
        self,
        prompt: Any,
        *,
        thread_id: str | None = None,
        max_iterations: int | None = None,
        **llm_kwargs: Any,
    ) -> AgentResponse:
        """Run the compiled graph to completion.

        Args:
            prompt: A string, a message, or a full transcript.
            thread_id: Conversation thread, also used as LangGraph's thread key.
            max_iterations: Override for the recursion limit.
            **llm_kwargs: Accepted for signature parity; graph nodes use the
                model's configured defaults.

        Returns:
            The answer, transcript, per-step trace and aggregate usage.

        Raises:
            MaxIterationsExceeded: When the recursion limit is hit.
            AgentInterrupt: When execution paused before a tool node.
            AgentError: When graph execution fails.
        """
        from windlass.core.exceptions import InterruptedError_ as AgentInterrupt
        from windlass.core.exceptions import MaxIterationsExceeded

        thread = thread_id or new_id("thread")
        started = time.perf_counter()
        messages = await self._prepare(prompt, thread)
        graph = self.compiled_graph()

        config = {
            "configurable": {"thread_id": thread},
            "recursion_limit": (max_iterations or self.max_iterations) * 2 + 2,
        }

        with self.tracer.span(f"agent.{self.name}", kind="agent", inputs=str(prompt)) as span:
            try:
                final = await graph.ainvoke({"messages": messages, "steps": []}, config=config)
            except Exception as exc:
                if "recursion" in str(exc).lower():
                    raise MaxIterationsExceeded(max_iterations or self.max_iterations) from exc
                raise AgentError(f"Graph execution failed: {exc}") from exc

            state = graph.get_state(config) if self.checkpointer is not None else None
            if state is not None and getattr(state, "next", None):
                pending = _pending_calls(final.get("messages", []))
                self._checkpoint(
                    thread,
                    [_to_windlass(m) for m in final.get("messages", [])],
                    list(final.get("steps", [])),
                    pending=pending,
                )
                raise AgentInterrupt(
                    f"{len(pending)} tool call(s) need approval.",
                    payload=[c.model_dump() for c in pending],
                    thread_id=thread,
                )

            transcript = [_to_windlass(m) for m in final.get("messages", [])]
            steps = [
                s if isinstance(s, AgentStep) else AgentStep.model_validate(s)
                for s in final.get("steps", [])
            ]
            output = transcript[-1].content if transcript else ""
            output = await self._finish(output, thread, prompt, transcript)

            usage = Usage(calls=0)
            for step in steps:
                usage = usage + step.usage

            span.set_output(output)
            span.set_usage(usage)

        return AgentResponse(
            output=output,
            messages=transcript,
            steps=steps,
            usage=usage,
            latency_ms=(time.perf_counter() - started) * 1000,
            thread_id=thread,
            metadata={"runtime": "langgraph"},
        )

    async def astream(
        self, prompt: Any, *, thread_id: str | None = None, **llm_kwargs: Any
    ) -> AsyncIterator[StreamEvent]:
        """Stream graph execution.

        LangGraph streams at node granularity rather than token granularity, so
        you receive one text event per completed assistant turn plus tool-call
        events. For token-level streaming use the built-in runtime.

        Args:
            prompt: A string, a message, or a full transcript.
            thread_id: Conversation thread.
            **llm_kwargs: Accepted for signature parity.

        Yields:
            :class:`~windlass.core.types.StreamEvent` values.
        """
        thread = thread_id or new_id("thread")
        messages = await self._prepare(prompt, thread)
        graph = self.compiled_graph()
        config = {
            "configurable": {"thread_id": thread},
            "recursion_limit": self.max_iterations * 2 + 2,
        }

        async for update in graph.astream({"messages": messages, "steps": []}, config=config):
            for node, payload in (update or {}).items():
                for raw in (payload or {}).get("messages", []):
                    message = _to_windlass(raw)
                    if message.content:
                        yield StreamEvent(type="text", delta=message.content, raw={"node": node})
                    for call in message.tool_calls:
                        yield StreamEvent(type="tool_call", tool_call=call)
        yield StreamEvent(type="done")

    def describe(self) -> dict[str, Any]:
        """Return a JSON-safe description, including the graph's node names."""
        parts = super().describe()
        parts["runtime"] = "langgraph"
        with contextlib.suppress(Exception):  # description must never fail
            parts["nodes"] = sorted(self.native_graph().nodes)
        return parts

    def draw(self) -> str:
        """Return a Mermaid diagram of the compiled graph.

        Returns:
            Mermaid source, ready to paste into documentation.

        Raises:
            AgentError: When the installed LangGraph cannot render diagrams.
        """
        try:
            return str(self.compiled_graph().get_graph().draw_mermaid())
        except Exception as exc:
            raise AgentError(f"Could not render the graph: {exc}") from exc


def _to_windlass(message: Any) -> Message:
    """Coerce whatever LangGraph put in state back into a Windlass message.

    LangGraph state is untyped by design; nodes may return Windlass messages,
    LangChain messages or plain dicts. Normalising here keeps that variability
    out of the rest of the runtime.

    Args:
        message: A message in any of those shapes.

    Returns:
        A Windlass :class:`~windlass.core.types.Message`.
    """
    if isinstance(message, Message):
        return message
    if isinstance(message, dict):
        return Message.model_validate(message)

    role_attr = getattr(message, "type", None) or getattr(message, "role", "assistant")
    role = {
        "human": Role.USER,
        "ai": Role.ASSISTANT,
        "system": Role.SYSTEM,
        "tool": Role.TOOL,
    }.get(str(role_attr), Role.ASSISTANT)

    calls: list[ToolCall] = []
    for raw in getattr(message, "tool_calls", None) or []:
        if isinstance(raw, dict):
            calls.append(
                ToolCall(
                    id=raw.get("id") or new_id("call"),
                    name=raw.get("name", ""),
                    arguments=raw.get("args") or raw.get("arguments") or {},
                )
            )

    return Message(
        role=role,
        content=str(getattr(message, "content", "") or ""),
        tool_calls=calls,
        tool_call_id=getattr(message, "tool_call_id", None),
    )


def _pending_calls(messages: Sequence[Any]) -> list[ToolCall]:
    """Return the tool calls from the newest assistant message."""
    for raw in reversed(list(messages)):
        message = _to_windlass(raw)
        if message.tool_calls:
            return message.tool_calls
    return []
