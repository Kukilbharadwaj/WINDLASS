"""MCP — binding Model Context Protocol servers to an agent.

Uses the in-process StaticMCPClient so the whole MCP path — discovery,
namespacing, proxying, agent binding — runs with no subprocess and no extras.
The real FastMCP client has an identical interface; see the notes at the end.

    python examples/08_mcp/main.py
"""

from __future__ import annotations

from windlass import Windlass, ToolCall, tool
from windlass.providers.mcp.fastmcp import MultiMCPClient, StaticMCPClient

# ---------------------------------------------------------------------------
# Two stand-in "servers". A real one is a subprocess or an HTTP endpoint;
# from the agent's point of view there is no difference.
# ---------------------------------------------------------------------------

FILES = {
    "/data/notes.md": "Launch is scheduled for March. Owner: Ada.",
    "/data/budget.csv": "quarter,spend\nQ1,12000\nQ2,15400",
}


def filesystem_server() -> StaticMCPClient:
    """A stand-in for @modelcontextprotocol/server-filesystem."""

    def read_file(path: str) -> str:
        """Read a file from the data directory.

        Args:
            path: Absolute path to read.
        """
        return FILES.get(path, f"No such file: {path}")

    def list_files() -> list[str]:
        """List the files available in the data directory."""
        return sorted(FILES)

    return StaticMCPClient(
        tools={"read_file": read_file, "list_files": list_files},
        resources={f"file://{p}": c for p, c in FILES.items()},
        prompts={"summarise": "Summarise {path} in {style} style."},
        server="filesystem",
    )


def search_server() -> StaticMCPClient:
    """A stand-in for an internal search server — note the clashing tool name."""

    def search(query: str) -> list[str]:
        """Search the internal knowledge base.

        Args:
            query: What to search for.
        """
        return [f"internal result for {query!r}"]

    return StaticMCPClient(tools={"search": search}, server="internal")


def web_server() -> StaticMCPClient:
    """Another server offering a tool called `search`. Collisions are normal."""

    def search(query: str) -> list[str]:
        """Search the public web.

        Args:
            query: What to search for.
        """
        return [f"web result for {query!r}"]

    return StaticMCPClient(tools={"search": search}, server="web")


@tool
def summarise(text: str) -> str:
    """Summarise a passage of text locally.

    Args:
        text: The text to summarise.
    """
    return text.split(".")[0] + "."


def main() -> None:
    # -------------------------------------------------------------------
    # 1. Discovery. Tools arrive with schemas derived from the server.
    # -------------------------------------------------------------------
    print("=== Discovery ===")
    files = filesystem_server()
    files.connect()

    for remote in files.list_tools():
        params = ", ".join(remote.parameters.get("properties", {}))
        print(f"  {remote.name}({params})")
        print(f"      {remote.description}")

    # -------------------------------------------------------------------
    # 2. Resources and prompts, the other two things MCP servers expose.
    # -------------------------------------------------------------------
    print("\n=== Resources ===")
    for resource in files.list_resources():
        print(f"  {resource.uri}")
    print(f"\n  read: {files.read_resource('file:///data/notes.md')}")

    print("\n=== Prompts ===")
    for prompt in files.list_prompts():
        print(f"  {prompt.name}")
    print(f"  rendered: {files.get_prompt('summarise', {'path': 'notes.md', 'style': 'brief'})}")

    # -------------------------------------------------------------------
    # 3. Bind to an agent. Remote and local tools sit side by side, and
    #    the model cannot tell them apart.
    # -------------------------------------------------------------------
    print("\n=== Remote and local tools together ===")
    agent = (
        Windlass.agent()
        .llm(
            "fake",
            responses=["", "The launch is scheduled for March, owned by Ada."],
            tool_calls=[
                [ToolCall(name="read_file", arguments={"path": "/data/notes.md"})],
                [],
            ],
        )
        .tool(summarise)  # local
        .mcp(filesystem_server())  # remote
    )
    print(f"  bound tools: {agent.build().tools.names()}")

    response = agent.run("What does notes.md say about the launch?")
    print(f"\n  answer: {response.output}")
    for step in response.steps:
        for call, result in zip(step.tool_calls, step.tool_results, strict=True):
            print(f"  called: {call.name}({call.arguments}) -> {result.content}")

    # -------------------------------------------------------------------
    # 4. Two servers, both offering `search`. Namespacing keeps them apart.
    # -------------------------------------------------------------------
    print("\n=== Two servers, one tool name ===")
    multi = MultiMCPClient(clients=[search_server(), web_server()])
    print(f"  discovered: {sorted(t.name for t in multi.list_tools())}")
    print(f"  internal_search: {multi.call_tool('internal_search', {'query': 'launch'})}")
    print(f"  web_search:      {multi.call_tool('web_search', {'query': 'launch'})}")

    # -------------------------------------------------------------------
    # 5. A dead server degrades; it does not stop the agent.
    # -------------------------------------------------------------------
    print("\n=== A server that is down ===")
    from windlass.interfaces.mcp import MCPClient

    class DeadServer(MCPClient):
        provider_name = "dead"

        async def aconnect(self) -> None:
            raise RuntimeError("connection refused")

        async def alist_tools(self):
            raise RuntimeError("connection refused")

        async def acall_tool(self, name, arguments):
            raise RuntimeError("connection refused")

    resilient = (
        Windlass.agent()
        .llm("fake", responses=["Still working with the tools I have."])
        .mcp(filesystem_server())
        .mcp(DeadServer(server="offline"))
    )
    print(f"  bound tools: {resilient.build().tools.names()}")
    print(f"  answer:      {resilient.run('carry on').output}")
    print("  (the dead server logged a warning and contributed nothing)")

    # -------------------------------------------------------------------
    print("\n=== Connecting to a real server ===")
    print("""  pip install "windlass[mcp]"

  agent = (
      Windlass.agent()
      .llm("gpt-4o")
      .mcp(command="npx",
           args=["-y", "@modelcontextprotocol/server-filesystem", "/data"],
           server="filesystem")
      .mcp(url="https://mcp.internal/sse", server="internal")
  )

  Identical interface — only the transport changes.""")


if __name__ == "__main__":
    main()
