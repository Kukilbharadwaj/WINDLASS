"""Agents — models that use tools to accomplish goals.

Start with :func:`windlass.Windlass.agent`, which returns an
:class:`~windlass.agent.builder.AgentBuilder`::

    from windlass import Windlass, tool

    @tool
    def search(query: str) -> list[str]:
        '''Search the knowledge base.'''
        return kb.search(query)

    agent = Windlass.agent().llm("openai").tool(search).memory()
    print(agent.run("What changed in the API last quarter?"))

The pieces:

* :class:`~windlass.agent.builder.AgentBuilder` — the fluent API.
* :class:`~windlass.agent.runtime.AgentRuntime` — the built-in reason/act loop,
  with no dependencies.
* :class:`~windlass.agent.graph.LangGraphRuntime` — LangGraph-backed execution,
  for conditional routing and subgraphs.
* :class:`~windlass.agent.supervisor.Supervisor` — multi-agent delegation.
* :mod:`~windlass.agent.checkpoint` — durable state for resume and
  human-in-the-loop.
"""

from __future__ import annotations

from windlass.agent.builder import AgentBuilder
from windlass.agent.checkpoint import Checkpointer, MemoryCheckpointer, SQLiteCheckpointer
from windlass.agent.runtime import DEFAULT_SYSTEM_PROMPT, AgentRuntime
from windlass.agent.supervisor import AgentTool, Supervisor

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "AgentBuilder",
    "AgentRuntime",
    "AgentTool",
    "Checkpointer",
    "MemoryCheckpointer",
    "SQLiteCheckpointer",
    "Supervisor",
]
