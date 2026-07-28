# 08 — MCP integration

Binding Model Context Protocol servers to an agent, with no subprocess required.

## Run

```bash
pip install windlass
python examples/08_mcp/main.py
```

This example uses `StaticMCPClient`, an in-process implementation of the MCP client interface. Every code path — discovery, namespacing, tool proxying, agent binding, failure handling — is the same one a real server takes. Only the transport differs.

For real servers:

```bash
pip install "windlass[mcp]"
```

## What it shows

1. **Discovery** — tools arriving with their schemas.
2. **Resources and prompts** — the other two things MCP servers expose.
3. **Remote and local tools together** — the model cannot tell them apart.
4. **Namespacing** — two servers both offering `search`, kept apart automatically.
5. **Graceful degradation** — a dead server logs a warning and contributes nothing.

## The point

After connecting, an MCP tool *is* a Windlass tool:

```python
agent = (
    Windlass.agent()
    .tool(my_local_function)       # local
    .mcp(command="npx", args=[...])  # remote
)
```

```
bound tools: ['list_files', 'read_file', 'summarise']
```

No distinction in the schema the model sees, no distinction in how results come back, no distinction in tracing.

## Namespacing

Two servers offering `search` is the normal case, not the exception:

```
discovered: ['internal_search', 'web_search']
```

With more than one server, Windlass prefixes tool names with the server label. The prefix is stripped again before the call is forwarded, so the server sees its own name.

## Degradation over failure

```
bound tools: ['list_files', 'read_file']
answer:      Still working with the tools I have.
```

An unreachable server logs a warning and contributes no tools. An agent that works with four of its five tool sources is more useful than one that refuses to start — and MCP servers are subprocesses and network endpoints, which means they *will* be unavailable sometimes.

## Testing MCP

`StaticMCPClient` is how the Windlass MCP tests run:

```python
from windlass.providers.mcp.fastmcp import StaticMCPClient

def test_the_agent_reads_the_file():
    client = StaticMCPClient(tools={"read_file": lambda path: "contents"})
    agent = fake_agent(["", "The file says: contents"],
                       tool_calls=[[call("read_file", path="/x")], []]).mcp(client)
    assert "contents" in agent.run("read /x").output
```

No subprocess, no network, no flakiness.

## Real servers

```python
agent = (
    Windlass.agent()
    .llm("gpt-4o")
    # stdio — Windlass starts and owns the subprocess
    .mcp(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/data"],
         server="filesystem")
    # HTTP / SSE
    .mcp(url="https://mcp.internal/sse", server="internal")
)
```

A stdio server is a subprocess Windlass starts. Use the async context manager, or call `disconnect()`, so it is reaped:

```python
async with Windlass.mcp(command="npx", args=[...]) as client:
    tools = await client.alist_tools()
```

## Security

MCP tools are code an LLM can trigger, on a server you may not control.

- **Scope the server.** The filesystem server takes a root directory — give it the narrowest one that works.
- **Gate consequential remote tools.** Wrap them, or set `requires_approval` on the proxy.
- **Treat results as untrusted.** Tool output can contain injected instructions: `agent.guardrails(injection=True)`.
- **Audit.** With tracing on, every remote call and result is recorded.
