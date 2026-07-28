# 06 — Multi-agent research

Three specialists coordinated by a supervisor, with a RAG pipeline exposed as a tool.

## Run

```bash
python examples/06_multi_agent/main.py                    # offline, scripted routing
OPENAI_API_KEY=sk-... python examples/06_multi_agent/main.py   # real routing
```

Offline, the routing is scripted so the delegation path is deterministic and readable. With a key, the models decide for themselves.

## Why split an agent up

One agent with thirty tools reasons badly. The tool list alone crowds the context, and the model has to hold every domain simultaneously. The symptoms are familiar: it picks the wrong tool, it loops, `max_iterations` starts firing.

Specialisation fixes it — a handful of relevant tools each, plus something that routes between them.

## What it shows

1. **Supervisor routing** — delegation is ordinary tool calling, because each specialist *is* a tool from the supervisor's point of view.
2. **Broadcast** — everyone on the same task, concurrently. Costs the slowest specialist, not the sum.
3. **Pipeline** — a fixed sequence where each stage depends on the last.
4. **A RAG pipeline as a tool** — nothing special is required to combine RAG and agents.
5. **Per-agent models** — research is high-volume and uses a cheap model; synthesis uses a stronger one.

## The routing logic is prose

```python
descriptions={
    "researcher": "Searches the web. Use for anything requiring external, current information.",
    "analyst":    "Searches internal policy documentation and interprets findings against it.",
    "writer":     "Turns notes into a finished briefing. Use last.",
}
```

The supervisor chooses purely on this text. "Use for anything requiring external, current information" routes correctly. "Does research" does not.

Treat these like prompts, because they are prompts.

## A RAG pipeline is just a tool

```python
@tool(name="search_internal_docs")
def search_internal_docs(query: str) -> list[dict]:
    """Search internal policy documentation.

    Args:
        query: What to look for.
    """
    return [{"content": hit.chunk.content} for hit in rag.search(query, k=2)]
```

This is worth internalising. There is no "RAG agent" abstraction in Windlass because there does not need to be one.

## Three coordination shapes

| Shape | When | Cost |
|---|---|---|
| `run()` | The supervisor should decide | One model call per routing decision |
| `broadcast()` | Independent work, or ensembling | The slowest specialist |
| `pipeline()` | Each stage depends on the last | The sum, unavoidably |

## Per-agent models

```python
researcher = Windlass.agent().llm("gpt-4o-mini")      # high volume
writer     = Windlass.agent().llm("claude-sonnet-4-5")  # quality matters
```

This mix is only possible because the model is a per-agent choice rather than a global setting. On a research task that makes twenty search calls and one synthesis call, it is most of the cost difference.

## Operating it

- **Budget every agent.** `max_iterations` per specialist, `timeout` per tool.
- **Trace everything.** Multi-agent runs are impossible to debug from logs. `.observe("langfuse")` on the supervisor covers the whole tree.
- **Watch the delegation pattern.** Everything going to one specialist means the descriptions need work. The same thing delegated twice means the supervisor's context is too short.
- **Alert on step count.** A creeping average means a tool description has stopped working.

## Next

The [advanced tutorial](../../docs/tutorials/advanced.md) adds human approval, custom graph topology and MCP capabilities to this system.
