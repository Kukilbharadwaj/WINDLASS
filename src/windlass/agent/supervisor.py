"""Multi-agent supervision.

One agent with thirty tools reasons badly: the tool list alone crowds the
context, and the model has to hold every domain in its head at once. The fix is
specialisation — a researcher, a coder, a writer, each with a handful of
relevant tools — plus a supervisor that routes work between them.

:class:`Supervisor` implements the standard pattern: a coordinating model reads
the request and each worker's description, picks one, runs it, reads the result,
and either delegates again or answers.

Example:
    >>> from windlass import Windlass
    >>> from windlass.core.types import ToolCall
    >>> researcher = Windlass.agent().llm("fake", responses=["Paris is in France."])
    >>> writer_llm = Windlass.llm("fake", responses=["Paris, France."])
    >>> writer = Windlass.agent().llm(writer_llm)
    >>> boss = Supervisor(
    ...     llm=Windlass.llm("fake", responses=["", "Paris, France."],
    ...                     tool_calls=[[ToolCall(name="researcher",
    ...                                           arguments={"task": "where is Paris"})], []]),
    ...     agents={"researcher": researcher, "writer": writer},
    ... )
    >>> boss.run("Where is Paris?").output
    'Paris, France.'
"""

from __future__ import annotations

import time
from typing import Any

from windlass.agent.runtime import AgentRuntime
from windlass.core.concurrency import gather_bounded, run_sync
from windlass.core.exceptions import AgentError
from windlass.core.logging import get_logger
from windlass.core.types import AgentResponse, AgentStep, Message, Usage, new_id
from windlass.interfaces.llm import LLM
from windlass.interfaces.tool import Tool
from windlass.interfaces.tracer import NullTracer, Tracer

__all__ = ["SUPERVISOR_PROMPT", "AgentTool", "Supervisor"]

_log = get_logger(__name__)

SUPERVISOR_PROMPT = """\
You coordinate a team of specialist agents. Your job is to route work, not to do
it yourself.

Available specialists:
{roster}

Delegate by calling the specialist as a tool, passing a clear, self-contained
task description — the specialist cannot see this conversation. Delegate to
several specialists at once when their work is independent.

When you have everything you need, write the final answer yourself. Do not
delegate work you have already received an answer for."""


class AgentTool(Tool):
    """Exposes an agent as a tool the supervisor can call.

    This is what makes delegation work with ordinary tool calling: the
    supervisor does not need a special routing mechanism, because a worker
    *is* a tool from its point of view.

    Args:
        agent: The worker agent, or anything with an ``arun``/``run`` method.
        name: Tool name the supervisor sees. Should say what the worker is for.
        description: What this specialist is good at. The supervisor routes on
            this text, so it deserves the same care as a prompt.
        **config: Forwarded to :class:`~windlass.interfaces.tool.Tool`.

    Example:
        >>> from windlass import Windlass
        >>> worker = Windlass.agent().llm("fake", responses=["done"])
        >>> t = AgentTool(worker, name="helper", description="Does small jobs.")
        >>> t.run(task="tidy up").content
        'done'
    """

    provider_name = "agent"

    def __init__(
        self,
        agent: Any,
        *,
        name: str,
        description: str = "",
        **config: Any,
    ) -> None:
        super().__init__(
            name=name,
            description=description or f"Delegate a task to the {name!r} specialist.",
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "A complete, self-contained description of "
                        "the work. The specialist cannot see the wider conversation.",
                    }
                },
                "required": ["task"],
            },
            **config,
        )
        self.agent = agent

    async def acall(self, **kwargs: Any) -> Any:
        """Run the worker agent on the delegated task.

        Args:
            **kwargs: Must contain ``task``.

        Returns:
            The worker's output text.

        Raises:
            AgentError: When the worker fails.
        """
        task = str(kwargs.get("task", "")).strip()
        if not task:
            return "No task was provided."
        runner = getattr(self.agent, "arun", None)
        if runner is None:
            raise AgentError(f"{self.name!r} is not runnable — it has no arun method.")
        response = await runner(task)
        return getattr(response, "output", str(response))

    def native(self) -> Any:
        """Return the wrapped agent."""
        return self.agent


class Supervisor(AgentRuntime):
    """An agent that delegates to specialist agents.

    Args:
        llm: The coordinating model. It only routes and summarises, so a
            mid-sized model is usually the right call.
        agents: Workers, keyed by the name the supervisor will use. Values may
            be :class:`~windlass.agent.runtime.AgentRuntime` instances, builders,
            or anything with ``arun``.
        descriptions: What each worker is for, keyed by the same names. Falls
            back to the worker's own description.
        system_prompt: Override for the routing instructions.
        max_iterations: Ceiling on routing rounds.
        tools: Tools the supervisor may call itself, alongside its workers.
        tracer: Observability backend.
        **kwargs: Forwarded to :class:`~windlass.agent.runtime.AgentRuntime`.

    Raises:
        AgentError: When no workers are supplied.

    Performance:
        Workers requested in the same turn run concurrently, so a supervisor
        that fans out to three specialists waits for the slowest, not the sum.
    """

    def __init__(
        self,
        *,
        llm: LLM,
        agents: dict[str, Any] | None = None,
        descriptions: dict[str, str] | None = None,
        system_prompt: str | None = None,
        max_iterations: int = 10,
        tools: list[Any] | None = None,
        tracer: Tracer | None = None,
        **kwargs: Any,
    ) -> None:
        workers = dict(agents or {})
        if not workers:
            raise AgentError(
                "A supervisor needs at least one specialist agent.",
                hint="Supervisor(llm=..., agents={'researcher': agent, ...})",
            )

        notes = dict(descriptions or {})
        agent_tools: list[Tool] = [
            AgentTool(worker, name=name, description=notes.get(name, _describe(worker, name)))
            for name, worker in workers.items()
        ]

        roster = "\n".join(f"  - {t.name}: {t.description}" for t in agent_tools)
        super().__init__(
            llm=llm,
            tools=[*agent_tools, *(tools or [])],
            system_prompt=system_prompt or SUPERVISOR_PROMPT.format(roster=roster),
            max_iterations=max_iterations,
            tracer=tracer or NullTracer(),
            name=kwargs.pop("name", "supervisor"),
            **kwargs,
        )
        self.agents = workers

    def add_agent(self, name: str, agent: Any, description: str = "") -> Supervisor:
        """Add a specialist after construction.

        Args:
            name: The name the supervisor will use.
            agent: The worker.
            description: What it is good at.

        Returns:
            ``self``.
        """
        self.agents[name] = agent
        self.tools.add(
            AgentTool(agent, name=name, description=description or _describe(agent, name))
        )
        return self

    async def abroadcast(self, task: str) -> dict[str, AgentResponse]:
        """Run every specialist on the same task, concurrently.

        Useful for ensembling — ask three specialists the same question and
        compare — and for fan-out work where every worker has something to
        contribute.

        Args:
            task: The task to send to every worker.

        Returns:
            A ``{worker_name: response}`` mapping. Workers that fail are
            reported as an error response rather than raising, so one failure
            does not lose the others' work.

        Example:
            >>> import asyncio
            >>> from windlass import Windlass
            >>> boss = Supervisor(
            ...     llm=Windlass.llm("fake"),
            ...     agents={"a": Windlass.agent().llm("fake", responses=["A"])},
            ... )
            >>> asyncio.run(boss.abroadcast("go"))["a"].output
            'A'
        """
        names = list(self.agents)

        async def _run(name: str) -> AgentResponse:
            worker = self.agents[name]
            runner = getattr(worker, "arun", None)
            if runner is None:
                raise AgentError(f"{name!r} is not runnable.")
            return await runner(task)

        started = time.perf_counter()
        outcomes = await gather_bounded(
            [_run(n) for n in names], limit=len(names), return_exceptions=True
        )

        results: dict[str, AgentResponse] = {}
        for name, outcome in zip(names, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                _log.warning("Specialist %s failed: %s", name, outcome)
                results[name] = AgentResponse(
                    output=f"{name} failed: {outcome}",
                    metadata={"error": True, "agent": name},
                )
            else:
                results[name] = outcome
        _log.debug(
            "Broadcast to %d specialists in %.2fs", len(names), time.perf_counter() - started
        )
        return results

    def broadcast(self, task: str) -> dict[str, AgentResponse]:
        """Blocking :meth:`abroadcast`."""
        return run_sync(self.abroadcast(task))

    async def apipeline(self, task: str, order: list[str]) -> AgentResponse:
        """Chain specialists in sequence, each seeing the previous output.

        The other multi-agent shape: research → draft → edit, where each stage
        genuinely depends on the last.

        Args:
            task: The initial task.
            order: Worker names, in execution order.

        Returns:
            The final worker's response, with every stage recorded in
            :attr:`~windlass.core.types.AgentResponse.steps`.

        Raises:
            AgentError: When a named worker does not exist.

        Example:
            >>> import asyncio
            >>> from windlass import Windlass
            >>> boss = Supervisor(
            ...     llm=Windlass.llm("fake"),
            ...     agents={
            ...         "draft": Windlass.agent().llm("fake", responses=["a draft"]),
            ...         "edit": Windlass.agent().llm("fake", responses=["an edit"]),
            ...     },
            ... )
            >>> asyncio.run(boss.apipeline("write", ["draft", "edit"])).output
            'an edit'
        """
        missing = [n for n in order if n not in self.agents]
        if missing:
            raise AgentError(
                f"Unknown specialist(s): {', '.join(missing)}.",
                context={"available": sorted(self.agents)},
            )

        started = time.perf_counter()
        thread = new_id("pipeline")
        current = task
        steps: list[AgentStep] = []
        transcript: list[Message] = [Message.user(task)]
        usage = Usage(calls=0)

        for index, name in enumerate(order):
            worker = self.agents[name]
            response = await worker.arun(current)
            current = response.output
            usage = usage + response.usage
            transcript.append(Message.assistant(current, name=name))
            steps.append(AgentStep(index=index, thought=current, node=name, usage=response.usage))

        return AgentResponse(
            output=current,
            messages=transcript,
            steps=steps,
            usage=usage,
            latency_ms=(time.perf_counter() - started) * 1000,
            thread_id=thread,
            metadata={"pattern": "pipeline", "order": order},
        )

    def pipeline(self, task: str, order: list[str]) -> AgentResponse:
        """Blocking :meth:`apipeline`."""
        return run_sync(self.apipeline(task, order))

    def describe(self) -> dict[str, Any]:
        """Return a JSON-safe description including the roster."""
        return {
            **super().describe(),
            "pattern": "supervisor",
            "specialists": sorted(self.agents),
        }

    def __repr__(self) -> str:
        return f"Supervisor(agents={sorted(self.agents)})"


def _describe(agent: Any, fallback: str) -> str:
    """Derive a worker description from its configuration.

    Args:
        agent: The worker.
        fallback: Name used when nothing better is available.

    Returns:
        A one-line description for the supervisor's roster.
    """
    prompt = getattr(agent, "system_prompt", None)
    if isinstance(prompt, str) and prompt.strip():
        first = prompt.strip().split("\n")[0]
        if first and not first.startswith("You are a capable assistant"):
            return first[:200]
    tools = getattr(agent, "tools", None)
    names = getattr(tools, "names", None)
    if callable(names):
        bound = names()
        if bound:
            return f"Specialist with access to: {', '.join(bound[:8])}."
    return f"The {fallback} specialist."
