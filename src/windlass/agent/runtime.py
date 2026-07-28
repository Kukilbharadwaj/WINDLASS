"""The built-in agent runtime.

A reason/act loop: call the model, execute whatever tools it asks for, feed the
results back, repeat until it answers or runs out of budget. That is what an
agent *is*, and Windlass implements it directly rather than requiring a graph
library — so agents work on a bare ``pip install windlass``.

The runtime supports everything the LangGraph path does except graph topology:
streaming, memory, guardrails, checkpointing, human-in-the-loop approval,
parallel tool execution and full tracing. Reach for
:class:`~windlass.agent.graph.LangGraphRuntime` (``.graph()`` on the builder)
when you need conditional routing, subgraphs or explicit state machines.

Example:
    >>> from windlass import Windlass, tool
    >>> @tool
    ... def add(a: int, b: int) -> int:
    ...     '''Add two numbers.'''
    ...     return a + b
    >>> agent = Windlass.agent().llm("fake", responses=["4"]).tool(add)
    >>> agent.run("What is 2 + 2?").output
    '4'
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from windlass.agent.checkpoint import Checkpointer
from windlass.core.concurrency import iter_sync, run_sync
from windlass.core.exceptions import AgentError, MalformedToolCallError, MaxIterationsExceeded
from windlass.core.exceptions import InterruptedError_ as AgentInterrupt
from windlass.core.logging import get_logger
from windlass.core.types import (
    AgentResponse,
    AgentStep,
    Message,
    Role,
    StreamEvent,
    ToolCall,
    Usage,
    new_id,
    normalize_messages,
)
from windlass.interfaces.guardrail import Guardrail
from windlass.interfaces.llm import LLM
from windlass.interfaces.memory import Memory
from windlass.interfaces.tracer import NullTracer, Tracer
from windlass.tools import ToolRegistry

__all__ = ["DEFAULT_SYSTEM_PROMPT", "AgentRuntime"]

_log = get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = """\
You are a capable assistant with access to tools.

Use a tool whenever it would give you information you do not already have, or
perform an action the user asked for. Call several tools in one turn when they
are independent. When you have what you need, answer directly and concisely.

If a tool fails, say what went wrong rather than inventing a result."""


class AgentRuntime:
    """A tool-calling agent.

    Args:
        llm: The reasoning model. Must support tool calling if tools are bound.
        tools: Tools the agent may call.
        system_prompt: Instructions prepended to every run.
        max_iterations: Ceiling on reason/act cycles. Prevents a confused agent
            from looping until your budget runs out.
        memory: Optional conversation memory, keyed by ``thread_id``.
        guardrail: Optional input/output policy.
        checkpointer: Optional durable state, enabling resume and time travel.
        tracer: Observability backend.
        parallel_tools: Execute simultaneous tool calls concurrently.
        require_approval: Pause before *every* tool call, not just those whose
            tools declare ``requires_approval``.
        max_tool_call_retries: How many times a run may recover from the model
            emitting an unparseable tool call before giving up. Each recovery
            costs one iteration from ``max_iterations``, so a model that cannot
            get it right still terminates. ``0`` disables recovery and restores
            the previous behaviour of failing immediately.
        name: Agent name, used in traces and multi-agent routing.

    Attributes:
        tools: The bound :class:`~windlass.tools.ToolRegistry`.
        name: The agent's name.

    Raises:
        AgentError: When tools are bound to a model that cannot call them.
    """

    def __init__(
        self,
        *,
        llm: LLM,
        tools: ToolRegistry | Sequence[Any] | None = None,
        system_prompt: str | None = None,
        max_iterations: int = 10,
        memory: Memory | None = None,
        guardrail: Guardrail | None = None,
        checkpointer: Checkpointer | None = None,
        tracer: Tracer | None = None,
        parallel_tools: bool = True,
        require_approval: bool = False,
        max_tool_call_retries: int = 2,
        name: str = "agent",
    ) -> None:
        self.llm = llm
        self.tools = tools if isinstance(tools, ToolRegistry) else ToolRegistry(list(tools or []))
        self.system_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        self.max_iterations = max(1, max_iterations)
        self.memory = memory
        self.guardrail = guardrail
        self.checkpointer = checkpointer
        self.tracer = tracer or NullTracer()
        self.parallel_tools = parallel_tools
        self.require_approval = require_approval
        self.max_tool_call_retries = max(0, max_tool_call_retries)
        self.name = name

        if len(self.tools) and not llm.supports_tools:
            raise AgentError(
                f"The {llm.name!r} provider does not support tool calling, but "
                f"{len(self.tools)} tool(s) are bound.",
                hint="Use a tool-capable model, or drop the tools.",
                context={"provider": llm.name, "tools": self.tools.names()},
            )

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------
    async def arun(
        self,
        prompt: Any,
        *,
        thread_id: str | None = None,
        max_iterations: int | None = None,
        **llm_kwargs: Any,
    ) -> AgentResponse:
        """Run the agent to completion.

        Args:
            prompt: A string, a message, or a full transcript.
            thread_id: Conversation thread. Required to use memory or
                checkpointing across calls; generated when omitted.
            max_iterations: Override for the step budget.
            **llm_kwargs: Per-call model overrides.

        Returns:
            The answer, transcript, per-step trace and aggregate usage.

        Raises:
            MaxIterationsExceeded: When the budget runs out before an answer.
            AgentInterrupt: When a tool needs human approval. Resume with
                :meth:`aresume`.
            GuardrailViolation: When a guardrail blocks the input or output.
            AgentError: When the run cannot proceed.

        Performance:
            One model call per iteration, plus tool time. Independent tool calls
            in the same turn run concurrently, which is usually where the wall
            clock is won.

        Example:
            >>> import asyncio
            >>> from windlass import Windlass
            >>> agent = Windlass.agent().llm("fake", responses=["Done."])
            >>> asyncio.run(agent.arun("Say done.")).output
            'Done.'
        """
        thread = thread_id or new_id("thread")
        budget = max_iterations or self.max_iterations
        started = time.perf_counter()

        with self.tracer.span(f"agent.{self.name}", kind="agent", inputs=str(prompt)) as span:
            messages = await self._prepare(prompt, thread)
            steps: list[AgentStep] = []
            usage = Usage(calls=0)

            corrections = 0
            for index in range(budget):
                try:
                    completion = await self._think(messages, index, **llm_kwargs)
                except MalformedToolCallError as exc:
                    # The model emitted a tool call the provider could not
                    # parse — usually by nesting one call inside another's
                    # arguments, which the protocol cannot express. That is the
                    # same class of mistake as inventing a tool name, so it gets
                    # the same treatment: show the model what went wrong and let
                    # it try again, rather than ending the run.
                    corrections += 1
                    if corrections > self.max_tool_call_retries:
                        raise
                    _log.warning(
                        "Model produced an unparseable tool call (correction %d/%d): %s",
                        corrections,
                        self.max_tool_call_retries,
                        exc,
                    )
                    messages.append(Message.user(exc.feedback))
                    steps.append(AgentStep(index=index, thought=f"[recovered] {exc}"))
                    continue

                usage = usage + completion.usage
                assistant = completion.to_message()
                messages.append(assistant)

                if not completion.tool_calls:
                    output = await self._finish(completion.content, thread, prompt, messages)
                    steps.append(
                        AgentStep(index=index, thought=completion.content, usage=completion.usage)
                    )
                    span.set_output(output)
                    span.set_usage(usage)
                    self._checkpoint(thread, messages, steps, done=True)
                    return AgentResponse(
                        output=output,
                        messages=messages,
                        steps=steps,
                        usage=usage,
                        latency_ms=(time.perf_counter() - started) * 1000,
                        thread_id=thread,
                    )

                pending = self._approvals(completion.tool_calls)
                if pending:
                    self._checkpoint(thread, messages, steps, pending=pending)
                    raise AgentInterrupt(
                        f"{len(pending)} tool call(s) need approval: "
                        + ", ".join(c.name for c in pending),
                        payload=[c.model_dump() for c in pending],
                        thread_id=thread,
                    )

                results = await self._act(completion.tool_calls)
                messages.extend(Message.tool(r) for r in results)
                steps.append(
                    AgentStep(
                        index=index,
                        thought=completion.content,
                        tool_calls=completion.tool_calls,
                        tool_results=results,
                        usage=completion.usage,
                    )
                )
                self._checkpoint(thread, messages, steps)

            raise MaxIterationsExceeded(budget)

    def run(
        self,
        prompt: Any,
        *,
        thread_id: str | None = None,
        max_iterations: int | None = None,
        **llm_kwargs: Any,
    ) -> AgentResponse:
        """Blocking :meth:`arun`."""
        return run_sync(
            self.arun(prompt, thread_id=thread_id, max_iterations=max_iterations, **llm_kwargs)
        )

    async def astream(
        self,
        prompt: Any,
        *,
        thread_id: str | None = None,
        **llm_kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream the agent's work as it happens.

        You get text deltas as the model produces them, ``tool_call`` events
        when it decides to act, and a final ``done``. That is what a live agent
        UI needs: users tolerate latency far better when they can see progress.

        Args:
            prompt: A string, a message, or a full transcript.
            thread_id: Conversation thread.
            **llm_kwargs: Per-call model overrides.

        Yields:
            :class:`~windlass.core.types.StreamEvent` values.

        Raises:
            MaxIterationsExceeded: When the budget runs out.
        """
        thread = thread_id or new_id("thread")
        messages = await self._prepare(prompt, thread)
        schemas = self._schemas()

        for _ in range(self.max_iterations):
            collected: list[str] = []
            calls: list[ToolCall] = []

            async for event in self.llm.astream(messages, tools=schemas, **llm_kwargs):
                if event.type == "text" and event.delta:
                    collected.append(event.delta)
                    yield event
                elif event.type == "tool_call" and event.tool_call:
                    calls.append(event.tool_call)
                    yield event

            text = "".join(collected)
            messages.append(Message(role=Role.ASSISTANT, content=text, tool_calls=calls))

            if not calls:
                await self._finish(text, thread, prompt, messages)
                yield StreamEvent(type="done")
                return

            results = await self._act(calls)
            messages.extend(Message.tool(r) for r in results)

        raise MaxIterationsExceeded(self.max_iterations)

    def stream(self, prompt: Any, **kwargs: Any) -> Iterator[StreamEvent]:
        """Blocking :meth:`astream`."""
        return iter_sync(self.astream(prompt, **kwargs))

    async def astream_text(self, prompt: Any, **kwargs: Any) -> AsyncIterator[str]:
        """Stream only the assistant's text — the common case for a chat UI."""
        async for event in self.astream(prompt, **kwargs):
            if event.type == "text" and event.delta:
                yield event.delta

    # ------------------------------------------------------------------
    # Human in the loop
    # ------------------------------------------------------------------
    async def aresume(
        self,
        thread_id: str,
        *,
        approved: bool = True,
        feedback: str = "",
        edited_arguments: dict[str, dict[str, Any]] | None = None,
        **llm_kwargs: Any,
    ) -> AgentResponse:
        """Resume a run that paused for human approval.

        Args:
            thread_id: The thread reported on the :class:`AgentInterrupt`.
            approved: True to execute the pending calls, False to reject them.
            feedback: Message shown to the model when rejecting, so it can try a
                different approach instead of simply repeating itself.
            edited_arguments: Replacement arguments keyed by tool-call id. This
                is the "human edits the action" case — approve the intent, fix
                the parameters.
            **llm_kwargs: Per-call model overrides.

        Returns:
            The completed run.

        Raises:
            AgentError: When there is no checkpoint, or nothing was pending.

        Example:
            >>> from windlass import Windlass, tool, AgentInterrupt
            >>> @tool(requires_approval=True)
            ... def deploy(env: str) -> str:
            ...     '''Deploy to an environment.'''
            ...     return f"deployed to {env}"
            >>> agent = (Windlass.agent()
            ...          .llm("fake", responses=["", "Deployed."],
            ...               tool_calls=[[ToolCall(name="deploy",
            ...                                     arguments={"env": "prod"})], []])
            ...          .tool(deploy).checkpoint())
            >>> try:
            ...     agent.run("Deploy to prod", thread_id="t1")
            ... except AgentInterrupt as pause:
            ...     resumed = agent.resume("t1", approved=True)
            >>> resumed.output
            'Deployed.'
        """
        if self.checkpointer is None:
            raise AgentError(
                "Resuming needs a checkpointer.",
                hint="Enable one with .checkpoint() on the agent builder.",
            )
        snapshot = self.checkpointer.get(thread_id)
        if snapshot is None:
            raise AgentError(
                f"No checkpoint found for thread {thread_id!r}.",
                context={"threads": self.checkpointer.threads()[:20]},
            )
        pending_raw = snapshot.get("pending") or []
        if not pending_raw:
            raise AgentError(
                f"Thread {thread_id!r} is not waiting for approval.",
                hint="Only interrupted runs can be resumed.",
            )

        messages = [Message.model_validate(m) for m in snapshot.get("messages", [])]
        steps = [AgentStep.model_validate(s) for s in snapshot.get("steps", [])]
        pending = [ToolCall.model_validate(c) for c in pending_raw]

        if approved:
            if edited_arguments:
                pending = [
                    call.model_copy(
                        update={"arguments": edited_arguments.get(call.id, call.arguments)}
                    )
                    for call in pending
                ]
            results = await self._act(pending)
        else:
            from windlass.core.types import ToolResult

            reason = feedback or "A human rejected this action."
            results = [
                ToolResult(call_id=c.id, name=c.name, content=reason, is_error=True)
                for c in pending
            ]

        messages.extend(Message.tool(r) for r in results)
        steps.append(AgentStep(index=len(steps), tool_calls=pending, tool_results=results))
        self._checkpoint(thread_id, messages, steps)

        response = await self.arun(
            messages,
            thread_id=thread_id,
            max_iterations=max(1, self.max_iterations - len(steps)),
            **llm_kwargs,
        )

        # ``arun`` starts a fresh step list, so without this the approved call —
        # the one a human explicitly authorised — is absent from ``steps`` while
        # still present in ``messages``. Anything reading ``steps`` as the audit
        # trail would show a payment that no step accounts for.
        if steps:
            combined = [*steps, *response.steps]
            for index, step in enumerate(combined):
                step.index = index
            response.steps = combined
        return response

    def resume(self, thread_id: str, **kwargs: Any) -> AgentResponse:
        """Blocking :meth:`aresume`."""
        return run_sync(self.aresume(thread_id, **kwargs))

    def pending_approvals(self, thread_id: str) -> list[ToolCall]:
        """Return the tool calls awaiting approval on a thread.

        Args:
            thread_id: The thread to inspect.

        Returns:
            The pending calls, or ``[]``.
        """
        if self.checkpointer is None:
            return []
        snapshot = self.checkpointer.get(thread_id) or {}
        return [ToolCall.model_validate(c) for c in snapshot.get("pending") or []]

    def state(self, thread_id: str) -> dict[str, Any] | None:
        """Return the latest checkpoint for a thread, or ``None``."""
        return self.checkpointer.get(thread_id) if self.checkpointer else None

    def history(self, thread_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent checkpoints for a thread, newest first."""
        return self.checkpointer.history(thread_id, limit) if self.checkpointer else []

    # ------------------------------------------------------------------
    # Level 3
    # ------------------------------------------------------------------
    def native_llm(self) -> Any:
        """Return the model provider's own client."""
        return self.llm.native()

    def native_graph(self) -> Any:
        """Return the underlying execution graph.

        The built-in runtime has no graph — it is a loop. Use ``.graph()`` on the
        agent builder to get a LangGraph-backed runtime whose ``native_graph()``
        returns a real ``StateGraph`` you can add nodes and edges to.

        Raises:
            AgentError: Always, explaining how to get a real graph.
        """
        raise AgentError(
            "The built-in runtime is a loop, not a graph.",
            hint="Build the agent with .graph() to use LangGraph, then call "
            "native_graph() to add nodes, edges and conditional edges.",
        )

    def describe(self) -> dict[str, Any]:
        """Return a JSON-safe description of the agent's configuration."""
        parts: dict[str, Any] = {
            "name": self.name,
            "runtime": "builtin",
            "llm": self.llm.describe(),
            "tools": self.tools.describe(),
            "max_iterations": self.max_iterations,
            "parallel_tools": self.parallel_tools,
        }
        for label, component in (
            ("memory", self.memory),
            ("guardrail", self.guardrail),
            ("tracer", self.tracer),
        ):
            if component is not None:
                parts[label] = component.describe()
        if self.checkpointer is not None:
            parts["checkpointer"] = type(self.checkpointer).__name__
        return parts

    async def aclose(self) -> None:
        """Release resources held by the model and any MCP connections."""
        for component in (self.llm, self.tracer):
            close = getattr(component, "aclose", None)
            if close is not None:
                try:
                    await close()
                except Exception as exc:
                    _log.debug("Closing %s failed: %s", component, exc)

    def close(self) -> None:
        """Blocking :meth:`aclose`."""
        run_sync(self.aclose())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    async def _prepare(self, prompt: Any, thread_id: str) -> list[Message]:
        """Build the initial transcript: system prompt, memory, then the input."""
        incoming = normalize_messages(prompt)
        latest = next((m.content for m in reversed(incoming) if m.role == Role.USER), "")

        if self.guardrail is not None and latest:
            safe = await self.guardrail.avalidate(latest, stage="input")
            if safe != latest:
                incoming = [
                    (
                        m.model_copy(update={"content": safe})
                        if m.role == Role.USER and m.content == latest
                        else m
                    )
                    for m in incoming
                ]

        messages: list[Message] = []
        if self.system_prompt and not any(m.role == Role.SYSTEM for m in incoming):
            messages.append(Message.system(self.system_prompt))

        if self.memory is not None:
            recalled = await self.memory.aget(thread_id=thread_id, query=latest)
            messages.extend(recalled)

        messages.extend(incoming)
        return messages

    async def _think(self, messages: list[Message], index: int, **llm_kwargs: Any) -> Any:
        """Make one model call."""
        with self.tracer.span(f"think[{index}]", kind="llm") as span:
            completion = await self.llm.acomplete(messages, tools=self._schemas(), **llm_kwargs)
            span.set_output(completion.content or [c.name for c in completion.tool_calls])
            span.set_usage(completion.usage)
            span.set_metadata(model=completion.model, tool_calls=len(completion.tool_calls))
        return completion

    async def _act(self, calls: Sequence[ToolCall]) -> list[Any]:
        """Execute tool calls, concurrently when there is more than one."""
        with self.tracer.span("tools", kind="tool", inputs=[c.name for c in calls]) as span:
            results = await self.tools.aexecute_many(calls, parallel=self.parallel_tools)
            span.set_output([r.content[:200] for r in results])
            span.set_metadata(count=len(results), errors=sum(1 for r in results if r.is_error))
        return results

    async def _finish(self, text: str, thread_id: str, prompt: Any, messages: list[Message]) -> str:
        """Apply the output guardrail and record the exchange in memory."""
        output = text
        if self.guardrail is not None and output:
            output = await self.guardrail.avalidate(output, stage="output")
        if self.memory is not None:
            user_text = next(
                (m.content for m in reversed(messages) if m.role == Role.USER), str(prompt)
            )
            await self.memory.aadd(
                [Message.user(user_text), Message.assistant(output)], thread_id=thread_id
            )
        return output

    def _schemas(self) -> list[dict[str, Any]] | None:
        """Return tool schemas in this provider's dialect, or ``None``."""
        if not len(self.tools):
            return None
        style = {"anthropic": "anthropic", "gemini": "gemini"}.get(self.llm.provider_name, "openai")
        return self.tools.schemas(style=style)

    def _approvals(self, calls: Sequence[ToolCall]) -> list[ToolCall]:
        """Return the calls that must be approved before running."""
        if self.checkpointer is None:
            return []
        if self.require_approval:
            return list(calls)
        return self.tools.needs_approval(calls)

    def _checkpoint(
        self,
        thread_id: str,
        messages: list[Message],
        steps: list[AgentStep],
        *,
        pending: Sequence[ToolCall] | None = None,
        done: bool = False,
    ) -> None:
        """Save a state snapshot, never failing the run if the save fails."""
        if self.checkpointer is None:
            return
        try:
            self.checkpointer.put(
                thread_id,
                {
                    "messages": [m.model_dump(mode="json") for m in messages],
                    "steps": [s.model_dump(mode="json") for s in steps],
                    "pending": [c.model_dump(mode="json") for c in (pending or [])],
                    "done": done,
                    "agent": self.name,
                },
            )
        except Exception as exc:
            _log.warning("Checkpoint save failed for thread %s: %s", thread_id, exc)

    def __repr__(self) -> str:
        return (
            f"AgentRuntime(name={self.name!r}, llm={self.llm.name!r}, "
            f"tools={self.tools.names()})"
        )
