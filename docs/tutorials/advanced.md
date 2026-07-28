# Advanced: a multi-agent research system with human approval

Build a system where specialists collaborate, a human approves anything consequential, and the execution graph is yours to shape.

**You need:** the [intermediate tutorial](intermediate.md), and:

```bash
pip install "windlass[openai,agent,rag,mcp,observability]"
```

---

## The problem with one big agent

An agent with thirty tools reasons badly. The tool list alone crowds the context, and the model has to hold every domain simultaneously. Symptoms: it picks the wrong tool, it loops, and `max_iterations` starts firing.

The fix is specialisation — several agents with a handful of relevant tools each, plus something that routes between them.

---

## 1. Build the specialists

```python
# research/agents.py
from windlass import Windlass, tool


@tool
def web_search(query: str, limit: int = 5) -> list[dict]:
    """Search the web.

    Args:
        query: What to search for.
        limit: Maximum results.
    """
    return search_api.query(query, limit=limit)


@tool
def read_page(url: str) -> str:
    """Fetch a web page and return its readable text.

    Args:
        url: The page to read.
    """
    return Windlass.loader("web").load(url)[0].content


def researcher():
    return (
        Windlass.agent()
        .llm("gpt-4o-mini")
        .tool(web_search, read_page)
        .system(
            "You find source material. Search, read the promising results, and "
            "report findings with the URL you took each one from. Do not "
            "speculate — if you did not read it, do not report it."
        )
        .max_iterations(12)
        .name("researcher")
    )


def analyst(rag):
    return (
        Windlass.agent()
        .llm("claude-sonnet-4-5")
        .tool(internal_docs_tool(rag))
        .system(
            "You interpret findings against our internal documentation. "
            "State clearly when internal policy and external sources disagree."
        )
        .name("analyst")
    )


def writer():
    return (
        Windlass.agent()
        .llm("claude-sonnet-4-5")
        .system(
            "You write briefings: a two-sentence summary, then the detail, then "
            "the sources. Plain language. No filler."
        )
        .name("writer")
    )
```

Note that each agent has its **own model**. Research is high-volume and benefits from a cheap fast model; writing benefits from a stronger one. That mix is only possible because the model is a per-agent choice.

### A RAG pipeline as a tool

```python
def internal_docs_tool(rag):
    @tool(name="search_internal_docs")
    def search_internal_docs(query: str) -> list[dict]:
        """Search internal company documentation.

        Args:
            query: What to look for.
        """
        return [
            {"content": hit.chunk.content, "source": hit.chunk.metadata.get("source")}
            for hit in rag.search(query, k=5)
        ]

    return search_internal_docs
```

This is worth internalising: **a RAG pipeline is just a tool an agent can call.** Nothing special is required to combine the two.

---

## 2. Coordinate them

```python
# research/team.py
from windlass import Windlass

from research.agents import analyst, researcher, writer


def team(rag):
    return Windlass.supervisor(
        {
            "researcher": researcher(),
            "analyst": analyst(rag),
            "writer": writer(),
        },
        llm=Windlass.llm("gpt-4o"),
        descriptions={
            "researcher": "Searches the web and reads pages. Use for anything "
                          "requiring external, current information.",
            "analyst":    "Searches internal documentation and interprets findings "
                          "against company policy.",
            "writer":     "Turns notes into a finished briefing. Use last.",
        },
        max_iterations=15,
        tracer=Windlass.tracer("langfuse"),
    )
```

The supervisor sees each specialist as a tool, so delegation is ordinary tool calling.

!!! tip "The descriptions *are* the routing logic"
    The supervisor chooses purely on this text. "Use for anything requiring external, current information" routes correctly; "does research" does not. Write them like prompts, because they are.

### Three coordination shapes

```python
boss = team(rag)

# 1. Let the supervisor decide.
print(boss.run("How does the EU AI Act affect our product roadmap?"))

# 2. Fixed sequence, when each stage genuinely depends on the last.
print(boss.pipeline(
    "Brief us on the EU AI Act timeline",
    ["researcher", "analyst", "writer"],
))

# 3. Everyone at once, for independent work or ensembling.
opinions = boss.broadcast("Assess the risk in this proposal")
for name, response in opinions.items():
    print(f"--- {name} ---\n{response.output}\n")
```

Broadcast runs the specialists concurrently, so it costs the slowest one, not the sum.

---

## 3. Put a human in front of consequential actions

```python
@tool(requires_approval=True)
def publish_briefing(title: str, body: str, audience: str) -> dict:
    """Publish a briefing to the company wiki.

    Args:
        title: Briefing title.
        body: Markdown body.
        audience: Which group can see it.
    """
    return wiki.publish(title=title, body=body, audience=audience)


boss = (
    Windlass.agent()
    .llm("gpt-4o")
    .agent("researcher", researcher())
    .agent("writer", writer())
    .tool(publish_briefing)
    .checkpoint("sqlite", path="./state.db")
)
```

The run pauses and is checkpointed:

```python
from windlass import AgentInterrupt

try:
    response = boss.run("Research and publish a briefing on X", thread_id="job-42")
except AgentInterrupt as pause:
    for call in pause.payload:
        print(f"Wants to call {call['name']} with {call['arguments']}")
```

Whenever a human decides — minutes or days later, in a different process:

```python
boss.resume("job-42", approved=True)

boss.resume("job-42", approved=False,
            feedback="Too speculative. Cite primary sources and try again.")

boss.resume("job-42", approved=True,
            edited_arguments={"call_abc": {"audience": "leadership-only"}})
```

Because state lives in SQLite, an approval queue is just a table of thread ids:

```python
for thread in checkpointer.threads():
    pending = boss.pending_approvals(thread)
    if pending:
        queue.add(thread, [c.model_dump() for c in pending])
```

Rejection feedback reaches the model as the tool result, so it changes approach rather than repeating itself.

---

## 4. Shape the execution graph

When the loop is not enough — you want a critic that can send work back — build on LangGraph and edit the real graph.

```python
from langgraph.graph import END

from windlass import Windlass

agent = Windlass.agent().llm("gpt-4o").tool(web_search, read_page).graph()

graph = agent.native_graph()


async def critique(state):
    """Check the draft for unsupported claims."""
    draft = state["messages"][-1].content
    verdict = await critic_llm.acomplete(
        f"Does this draft contain claims not supported by its sources? "
        f"Answer PASS or REVISE, then explain.\n\n{draft}"
    )
    return {
        "messages": [Message.assistant(verdict.content)],
        "needs_revision": verdict.content.strip().startswith("REVISE"),
    }


graph.add_node("critic", critique)
graph.add_edge("tools", "critic")
graph.add_conditional_edges(
    "critic",
    lambda state: "revise" if state.get("needs_revision") else "done",
    {"revise": "agent", "done": END},
)

agent.recompile()

print(agent.draw())          # Mermaid source for the compiled graph
```

You changed the topology. Tools, memory, guardrails, checkpointing and tracing all still work — that is the Level 3 promise.

---

## 5. Add MCP capabilities

```python
agent = (
    Windlass.agent()
    .llm("gpt-4o")
    .tool(local_analysis)
    .mcp(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/data"],
         server="files")
    .mcp(url="https://mcp.internal/sse", server="internal")
)
```

Remote tools arrive namespaced (`files_read`, `internal_search`), and the model cannot distinguish them from local ones. A server that is down logs a warning and contributes nothing — the agent still runs with what it has.

---

## 6. Test it without the internet

The whole system is testable offline, because every model is replaceable.

```python
# tests/test_team.py
from windlass import Windlass
from windlass.testing import RecordingTool, call, capture_spans


def test_the_supervisor_delegates_before_answering():
    recorder = RecordingTool("researcher_stub", result="Findings: X is true.")

    boss = Windlass.supervisor(
        {"researcher": Windlass.agent().llm("fake", responses=["Findings: X is true."])},
        llm=Windlass.llm(
            "fake",
            responses=["", "X is true, according to the researcher."],
            tool_calls=[[call("researcher", task="investigate X")], []],
        ),
    )

    with capture_spans() as spans:
        response = boss.run("Is X true?")

    assert response.output == "X is true, according to the researcher."
    assert response.tool_calls[0].name == "researcher"
    assert spans.count("tool") >= 1


def test_publishing_pauses_for_approval():
    from windlass import AgentInterrupt, tool

    published = []

    @tool(requires_approval=True)
    def publish(title: str) -> str:
        """Publish a briefing."""
        published.append(title)
        return "published"

    agent = (
        Windlass.agent()
        .llm("fake", responses=["", "Published."],
             tool_calls=[[call("publish", title="Draft")], []])
        .tool(publish)
        .checkpoint()
    )

    try:
        agent.run("publish it", thread_id="t1")
        raise AssertionError("should have paused")
    except AgentInterrupt:
        pass

    assert published == []              # nothing happened without approval
    assert agent.resume("t1", approved=True).output == "Published."
    assert published == ["Draft"]
```

No network, no keys, no flakiness — and the approval test asserts the property that actually matters: **nothing executed before a human said yes.**

---

## 7. Operate it

**Budget every agent.** `max_iterations` per specialist, `timeout` per tool. A confused agent is expensive.

**Trace everything.** Multi-agent runs are impossible to debug from logs. `.observe("langfuse")` on the supervisor covers the whole tree.

**Alert on the step count.** A creeping average means a tool description has stopped working, or the task has outgrown the team's design.

**Match model to job.** Research is high-volume — use a cheap model. Writing and final synthesis benefit from a stronger one.

**Watch the delegation pattern.** If the supervisor sends everything to one specialist, the descriptions need work. If it delegates the same thing twice, its context is too short.

---

## What you learned

- Specialise, then coordinate. One agent with thirty tools reasons worse than three with ten.
- Specialist descriptions are the routing logic.
- A RAG pipeline is just a tool.
- Human approval is checkpointed, so the queue can be asynchronous and durable.
- Level 3 lets you rewrite the graph without giving up anything else.
- The whole system remains testable offline, because every model is replaceable.

## Next

- [Writing plugins](../guides/plugins.md) — turn a repeated Level 3 pattern into a component.
- [Performance](../guides/performance.md).
- [Best practices](../guides/best-practices.md).
