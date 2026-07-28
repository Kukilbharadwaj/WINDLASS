# Tools and MCP

## The decorator

```python
from windlass import tool

@tool
def get_weather(city: str, units: str = "celsius") -> dict:
    """Look up the current weather.

    Args:
        city: City name, e.g. "Paris".
        units: "celsius" or "fahrenheit".
    """
    return weather_api.current(city, units)
```

That is the whole API. The name comes from the function, the description from the docstring summary, and the JSON schema from the type hints and the `Args:` section. There is no second source of truth to keep in sync — which is exactly why schemas drift in hand-written tool definitions.

## Schema generation

| Annotation | Schema |
|---|---|
| `str` `int` `float` `bool` | `string` `integer` `number` `boolean` |
| `list[str]` | `{"type": "array", "items": {"type": "string"}}` |
| `dict[str, int]` | `{"type": "object", "additionalProperties": {"type": "integer"}}` |
| `set[str]` | array with `uniqueItems` |
| `tuple[int, str]` | array with `prefixItems` |
| `Literal["a", "b"]` | `{"enum": ["a", "b"], "type": "string"}` |
| `MyEnum` | enum of the member values |
| `str \| None` | `string`, and the parameter is optional |
| A Pydantic model | a nested object schema, with `$ref`s inlined |
| A dataclass | a nested object schema |
| `Annotated[int, "how many"]` | `integer` with that description |

Parameters without a default are `required`. `*args` and `**kwargs` are skipped — providers cannot express them, and a tool that needs them almost always wants an explicit list or dict parameter instead.

!!! note "One awkward annotation costs one parameter"
    If an annotation names a symbol that is not visible at module scope, Windlass resolves the remaining parameters individually rather than falling back to an empty schema for the whole tool.

### Provider dialects

```python
get_weather.schema()                      # OpenAI (also Groq, Ollama)
get_weather.schema(style="anthropic")     # input_schema
get_weather.schema(style="gemini")        # unsupported keywords stripped
```

The agent picks the right dialect for its model. You never have to.

## Async tools

```python
@tool
async def fetch_user(user_id: str) -> dict:
    """Fetch a user record.

    Args:
        user_id: The user's id.
    """
    async with httpx.AsyncClient() as client:
        return (await client.get(f"/users/{user_id}")).json()
```

Sync tools run on a worker thread, so a blocking HTTP call inside a tool cannot stall the agent's event loop.

## Options

```python
@tool(
    name="charge_card",             # override the function name
    description="Charge a saved card.",
    timeout=30.0,                   # abandon the call after this long
    requires_approval=True,         # pause for a human first
    register=True,                  # also add to the registry, usable by name
)
async def charge(amount_cents: int, customer_id: str) -> dict:
    """Charge a customer's saved card."""
    return await payments.charge(customer_id, amount_cents)
```

## Still an ordinary function

```python
@tool
def double(x: int) -> int:
    """Double a number."""
    return x * 2

double(21)                # 42 — plain call
double.run(x=21).data     # 42 — through the tool machinery
```

Which means tools are testable without an agent.

## Stateful tools

When a capability needs a client, a connection or configuration, subclass `Tool`:

```python
from windlass import Tool

class CatalogSearch(Tool):
    """Searches the product catalogue."""

    def __init__(self, client):
        super().__init__(
            name="search_catalog",
            description="Search the product catalogue by keyword.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for."},
                    "limit": {"type": "integer", "description": "Max results.", "default": 10},
                },
                "required": ["query"],
            },
        )
        self.client = client

    async def acall(self, **kwargs):
        return await self.client.search(kwargs["query"], limit=kwargs.get("limit", 10))

agent = Windlass.agent().tool(CatalogSearch(my_client))
```

## Errors are data

An exception inside a tool becomes an error result, not a crash:

```python
result = await registry.aexecute(ToolCall(name="explode"))
result.is_error      # True
result.content       # "RuntimeError: upstream is down"
```

The agent hands that text to the model, which can then explain the problem or try something else. A tool exception that unwound the agent loop would remove the model's chance to recover — and recovering is the whole point of an agent.

Direct invocation raises instead, because there is no model to recover:

```python
tool.run(bad="args")     # raises ToolExecutionError
```

## MCP

Model Context Protocol servers expose **tools**, **resources** and **prompts** over stdio or HTTP. Windlass treats a server as one more source of tools.

```bash
pip install "windlass[mcp]"
```

### Connecting

```python
agent = (
    Windlass.agent()
    .llm("gpt-4o")
    .mcp(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/data"])
    .mcp(url="https://mcp.internal/sse", server="internal")
)
```

After connecting, remote tools sit alongside your local ones and the model cannot tell them apart.

### Namespacing

With more than one server, tools are prefixed with the server name — `filesystem_read`, `internal_search` — because two servers offering `search` is the normal case, not the exception.

### Resources and prompts

```python
client = Windlass.mcp(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "."])
client.connect()

for resource in client.list_resources():
    print(resource.uri, resource.name)

content = client.read_resource("file:///data/report.md")

for prompt in client.list_prompts():
    print(prompt.name, prompt.arguments)

rendered = client.get_prompt("summarize", {"style": "brief"})
```

### Lifecycle

A stdio server is a subprocess Windlass starts and owns. Use the async context manager so it is reaped:

```python
async with Windlass.mcp(command="npx", args=[...]) as client:
    tools = await client.alist_tools()
```

### Testing MCP without a server

```python
from windlass.providers.mcp.fastmcp import StaticMCPClient

client = StaticMCPClient(
    tools={"shout": lambda text: text.upper()},
    resources={"file://greeting": "hello"},
    prompts={"welcome": "Hello, {name}!"},
)

agent = Windlass.agent().llm("fake", responses=["done"]).mcp(client)
```

Every MCP code path — discovery, namespacing, proxying, agent binding — is exercised without a subprocess. This is how the Windlass MCP tests run.

### Degradation

An unreachable server logs a warning and contributes no tools. A `MultiMCPClient` with one dead server still serves the others.

## Security

Tools are the point where a model touches the world, which makes them the attack surface.

**Least privilege.** Bind the smallest capability that does the job. `search_products(query)` scoped to one index is safer than `run_sql(query)` — prompt injection is real, and the strongest mitigation is that the injected instruction has nothing dangerous to reach.

**Validate inside the tool.** The schema is a hint to the model, not a security boundary. Check inputs in the function body.

**Gate consequential actions.** `requires_approval=True` on anything that spends money, sends a message or deletes data.

**Treat retrieved text as untrusted.** Documents and tool output can contain instructions:

```python
agent.guardrails(injection=True)
```

**Set timeouts.** A tool that never returns is an agent that never returns.

**Audit.** With tracing on, every tool call and result is recorded — which is what makes an incident investigable afterwards.
