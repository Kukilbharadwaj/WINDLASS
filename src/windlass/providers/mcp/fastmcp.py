"""FastMCP client adapter, plus a dependency-free in-process client for tests.

Model Context Protocol servers expose tools, resources and prompts over stdio or
HTTP. Once connected, their tools become ordinary Windlass tools — an agent
binding a local Python function and a remote MCP tool cannot tell them apart,
which is exactly the point.

Install with::

    pip install "windlass[mcp]"

Example:
    >>> from windlass import Windlass                                           # doctest: +SKIP
    >>> agent = Windlass.agent().mcp("filesystem",                             # doctest: +SKIP
    ...     command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "."])
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from windlass.core.exceptions import ConfigurationError, MCPError
from windlass.core.lazy import require
from windlass.core.registry import register
from windlass.interfaces.mcp import MCPClient, MCPPrompt, MCPResource, MCPToolProxy
from windlass.interfaces.tool import Tool
from windlass.tools import FunctionTool

__all__ = ["FastMCPClient", "MultiMCPClient", "StaticMCPClient"]


@register.mcp(
    "fastmcp",
    aliases=("mcp", "server"),
    description="Connects to MCP servers over stdio, SSE or streamable HTTP.",
)
class FastMCPClient(MCPClient):
    """MCP client built on ``fastmcp``.

    Args:
        server: Label for this server, used in logs and namespacing.
        command: Executable for a stdio server, e.g. ``"npx"`` or ``"python"``.
        args: Arguments for ``command``.
        url: HTTP or SSE endpoint, as an alternative to ``command``.
        env: Extra environment variables for a stdio server.
        config: A full FastMCP client configuration dict, for advanced setups.
        namespace: Prefix remote tool names with the server label, so two servers
            offering ``search`` do not collide.
        timeout: Per-call timeout in seconds.
        **kwargs: Forwarded to :class:`~windlass.interfaces.mcp.MCPClient`.

    Raises:
        MissingDependencyError: When ``fastmcp`` is not installed.
        ConfigurationError: When neither ``command``, ``url`` nor ``config`` is given.
        MCPError: When the server cannot be reached.

    Note:
        A stdio server is a subprocess this client starts and owns. Use it as an
        async context manager, or call :meth:`~windlass.interfaces.mcp.MCPClient.disconnect`,
        so the process is reaped.
    """

    provider_name = "fastmcp"

    def __init__(
        self,
        *,
        server: str = "",
        command: str | None = None,
        args: list[str] | None = None,
        url: str | None = None,
        env: dict[str, str] | None = None,
        config: dict[str, Any] | None = None,
        namespace: bool = False,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(server=server or "mcp", namespace=namespace, timeout=timeout, **kwargs)
        self._fastmcp = require("fastmcp", extra="mcp", feature="The MCP client")
        if not (command or url or config):
            raise ConfigurationError(
                "An MCP client needs a transport.",
                hint="Pass command=... (+args) for a stdio server, url=... for HTTP, "
                "or config=... for a full FastMCP configuration.",
            )
        self.command = command
        self.args = list(args or [])
        self.url = url
        self.env = dict(env or {})
        self.server_config = config
        self._client: Any = None
        self._session: Any = None

    def _transport(self) -> Any:
        """Build the FastMCP transport described by the configuration."""
        if self.server_config is not None:
            return self.server_config
        if self.url:
            return self.url
        spec: dict[str, Any] = {"command": self.command, "args": self.args}
        if self.env:
            spec["env"] = self.env
        return {"mcpServers": {self.server: spec}}

    def native(self) -> Any:
        """Return the underlying FastMCP client (Level 3 access)."""
        return self._client

    async def aconnect(self) -> None:
        """Open the transport and enter the MCP session.

        Idempotent: calling it twice does not start a second server.

        Raises:
            MCPError: When the server cannot be started or the handshake fails.
        """
        if self.connected:
            return
        try:
            self._client = self._fastmcp.Client(self._transport())
            self._session = await self._client.__aenter__()
            self.connected = True
        except Exception as exc:
            self._client = self._session = None
            raise MCPError(
                f"Could not connect to MCP server {self.server!r}: {exc}",
                hint="Check the command is on PATH (try running it by hand), or that "
                "the URL is reachable.",
                context={"server": self.server, "command": self.command, "url": self.url},
            ) from exc

    async def adisconnect(self) -> None:
        """Close the session and reap any subprocess."""
        if self._client is not None:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception as exc:
                self._log.debug("MCP disconnect from %s failed: %s", self.server, exc)
        self._client = self._session = None
        self.connected = False

    async def alist_tools(self) -> list[Tool]:
        """Discover the server's tools.

        Returns:
            :class:`~windlass.interfaces.mcp.MCPToolProxy` instances, ready to
            bind to an agent.

        Raises:
            MCPError: When discovery fails.
        """
        await self.aconnect()
        try:
            discovered = await self._client.list_tools()
        except Exception as exc:
            raise MCPError(
                f"Could not list tools on MCP server {self.server!r}: {exc}",
                context={"server": self.server},
            ) from exc

        tools: list[Tool] = []
        for item in discovered:
            remote_name = _attr(item, "name", "")
            if not remote_name:
                continue
            tools.append(
                MCPToolProxy(
                    self,
                    name=self._tool_name(remote_name),
                    description=_attr(item, "description", "") or "",
                    parameters=_attr(item, "inputSchema", None)
                    or _attr(item, "input_schema", None)
                    or {"type": "object", "properties": {}},
                    server=self.server,
                    timeout=self.timeout,
                )
            )
        return tools

    async def acall_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Invoke a remote tool.

        Args:
            name: Tool name, with any Windlass namespace prefix stripped.
            arguments: Arguments matching the remote schema.

        Returns:
            The unwrapped result — plain text where the server returned text,
            parsed JSON where it returned JSON, otherwise the raw payload.

        Raises:
            MCPError: When the call fails.
        """
        await self.aconnect()
        remote = _strip_namespace(name, self.server) if self.namespace else name
        try:
            result = await self._client.call_tool(remote, arguments)
        except Exception as exc:
            raise MCPError(
                f"MCP tool {remote!r} on {self.server!r} failed: {exc}",
                context={"server": self.server, "tool": remote},
            ) from exc
        return _unwrap_content(result)

    async def alist_resources(self) -> list[MCPResource]:
        """Discover the server's readable resources."""
        await self.aconnect()
        try:
            discovered = await self._client.list_resources()
        except Exception as exc:
            self._log.debug("MCP server %s does not support resources: %s", self.server, exc)
            return []
        return [
            MCPResource(
                uri=str(_attr(item, "uri", "")),
                name=_attr(item, "name", "") or "",
                description=_attr(item, "description", "") or "",
                mimetype=_attr(item, "mimeType", None) or _attr(item, "mimetype", None),
                server=self.server,
            )
            for item in discovered
            if _attr(item, "uri", None)
        ]

    async def aread_resource(self, uri: str) -> str:
        """Read a resource's contents.

        Args:
            uri: The resource identifier.

        Returns:
            The resource body as text.

        Raises:
            MCPError: When the resource cannot be read.
        """
        await self.aconnect()
        try:
            result = await self._client.read_resource(uri)
        except Exception as exc:
            raise MCPError(
                f"Could not read MCP resource {uri!r}: {exc}",
                context={"server": self.server, "uri": uri},
            ) from exc
        content = _unwrap_content(result)
        return content if isinstance(content, str) else json.dumps(content, default=str)

    async def alist_prompts(self) -> list[MCPPrompt]:
        """Discover the server's prompt templates."""
        await self.aconnect()
        try:
            discovered = await self._client.list_prompts()
        except Exception as exc:
            self._log.debug("MCP server %s does not support prompts: %s", self.server, exc)
            return []
        return [
            MCPPrompt(
                name=_attr(item, "name", ""),
                description=_attr(item, "description", "") or "",
                arguments=[_as_dict(a) for a in (_attr(item, "arguments", None) or [])],
                server=self.server,
            )
            for item in discovered
            if _attr(item, "name", None)
        ]

    async def aget_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Instantiate a prompt template.

        Args:
            name: Prompt name.
            arguments: Template arguments.

        Returns:
            The rendered prompt text.

        Raises:
            MCPError: When the prompt cannot be rendered.
        """
        await self.aconnect()
        try:
            result = await self._client.get_prompt(name, arguments or {})
        except Exception as exc:
            raise MCPError(
                f"Could not render MCP prompt {name!r}: {exc}",
                context={"server": self.server, "prompt": name},
            ) from exc

        messages = _attr(result, "messages", None)
        if messages:
            parts = []
            for message in messages:
                content = _attr(message, "content", "")
                parts.append(content if isinstance(content, str) else _unwrap_content(content))
            return "\n\n".join(str(p) for p in parts if p)
        content = _unwrap_content(result)
        return content if isinstance(content, str) else json.dumps(content, default=str)


@register.mcp(
    "static",
    aliases=("inprocess", "fake"),
    description="In-process MCP client for tests and local tool bundles (no dependencies).",
)
class StaticMCPClient(MCPClient):
    """An MCP client backed by local Python callables.

    Lets you exercise every MCP code path — discovery, namespacing, tool
    proxying, agent binding — without starting a subprocess or a server. That
    makes MCP integration testable in ordinary unit tests.

    Args:
        tools: Mapping of tool name to callable.
        resources: Mapping of URI to content.
        prompts: Mapping of prompt name to a template string, formatted with the
            supplied arguments.
        server: Server label.
        namespace: Prefix tool names with the server label.
        **kwargs: Forwarded to :class:`~windlass.interfaces.mcp.MCPClient`.

    Example:
        >>> client = StaticMCPClient(
        ...     tools={"shout": lambda text: text.upper()},
        ...     resources={"file://greeting": "hello"},
        ... )
        >>> client.connect()
        >>> client.call_tool("shout", {"text": "hi"})
        'HI'
        >>> client.read_resource("file://greeting")
        'hello'
    """

    provider_name = "static"

    def __init__(
        self,
        *,
        tools: dict[str, Callable[..., Any]] | None = None,
        resources: dict[str, str] | None = None,
        prompts: dict[str, str] | None = None,
        server: str = "static",
        namespace: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(server=server, namespace=namespace, **kwargs)
        self._tools = dict(tools or {})
        self._resources = dict(resources or {})
        self._prompts = dict(prompts or {})

    async def aconnect(self) -> None:
        """Mark the client connected. There is no transport to open."""
        self.connected = True

    async def alist_tools(self) -> list[Tool]:
        """Return one tool per registered callable, with derived schemas."""
        await self.aconnect()
        built: list[Tool] = []
        for name, fn in self._tools.items():
            wrapped = FunctionTool(fn, name=self._tool_name(name))
            built.append(
                MCPToolProxy(
                    self,
                    name=wrapped.name,
                    description=wrapped.description,
                    parameters=wrapped.parameters,
                    server=self.server,
                )
            )
        return built

    async def acall_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Invoke a registered callable.

        Args:
            name: Tool name, with any namespace prefix stripped.
            arguments: Keyword arguments.

        Returns:
            The callable's return value.

        Raises:
            MCPError: When no such tool is registered or the call fails.
        """
        remote = _strip_namespace(name, self.server) if self.namespace else name
        fn = self._tools.get(remote)
        if fn is None:
            raise MCPError(
                f"No tool named {remote!r} on server {self.server!r}.",
                hint=f"Available tools: {', '.join(sorted(self._tools)) or '(none)'}",
                context={"available": sorted(self._tools)},
            )
        import asyncio

        try:
            if asyncio.iscoroutinefunction(fn):
                return await fn(**arguments)
            return fn(**arguments)
        except Exception as exc:
            raise MCPError(f"MCP tool {remote!r} failed: {exc}") from exc

    async def alist_resources(self) -> list[MCPResource]:
        """Return the registered resources."""
        return [
            MCPResource(uri=uri, name=uri.rsplit("/", 1)[-1], server=self.server)
            for uri in self._resources
        ]

    async def aread_resource(self, uri: str) -> str:
        """Return a registered resource's content.

        Raises:
            MCPError: When the URI is not registered.
        """
        if uri not in self._resources:
            raise MCPError(
                f"No resource at {uri!r}.",
                hint=f"Available resources: {', '.join(sorted(self._resources)) or '(none)'}",
                context={"available": sorted(self._resources)},
            )
        return self._resources[uri]

    async def alist_prompts(self) -> list[MCPPrompt]:
        """Return the registered prompt templates."""
        return [MCPPrompt(name=name, server=self.server) for name in self._prompts]

    async def aget_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Render a registered prompt template.

        Raises:
            MCPError: When the prompt is not registered or a placeholder is missing.
        """
        template = self._prompts.get(name)
        if template is None:
            raise MCPError(
                f"No prompt named {name!r}.", context={"available": sorted(self._prompts)}
            )
        try:
            return template.format(**(arguments or {}))
        except KeyError as exc:
            raise MCPError(f"Prompt {name!r} is missing argument {exc}.") from exc


class MultiMCPClient(MCPClient):
    """Aggregates several MCP servers behind one client.

    Tool discovery unions every server's tools, and calls are routed back to
    whichever server advertised the tool. Namespacing is forced on, because two
    servers offering ``search`` is the normal case, not the exception.

    Args:
        clients: The servers to aggregate.
        server: Label for the aggregate.
        **kwargs: Forwarded to :class:`~windlass.interfaces.mcp.MCPClient`.

    Example:
        >>> a = StaticMCPClient(tools={"ping": lambda: "pong"}, server="alpha")
        >>> b = StaticMCPClient(tools={"ping": lambda: "pong"}, server="beta")
        >>> multi = MultiMCPClient(clients=[a, b])
        >>> sorted(t.name for t in multi.list_tools())
        ['alpha_ping', 'beta_ping']
    """

    provider_name = "multi"

    def __init__(
        self, *, clients: list[MCPClient] | None = None, server: str = "multi", **kwargs: Any
    ) -> None:
        super().__init__(server=server, namespace=True, **kwargs)
        self.clients = list(clients or [])
        for client in self.clients:
            client.namespace = True
        self._routes: dict[str, MCPClient] = {}

    def add(self, client: MCPClient) -> MultiMCPClient:
        """Add a server and return ``self``."""
        client.namespace = True
        self.clients.append(client)
        return self

    def native(self) -> Any:
        """Return the list of underlying clients."""
        return self.clients

    async def aconnect(self) -> None:
        """Connect every server, tolerating individual failures."""
        from windlass.core.concurrency import gather_bounded

        if not self.clients:
            self.connected = True
            return
        outcomes = await gather_bounded(
            [c.aconnect() for c in self.clients],
            limit=len(self.clients),
            return_exceptions=True,
        )
        for client, outcome in zip(self.clients, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                self._log.warning("MCP server %s unavailable: %s", client.server, outcome)
        self.connected = True

    async def adisconnect(self) -> None:
        """Disconnect every server."""
        for client in self.clients:
            await client.adisconnect()
        self.connected = False

    async def alist_tools(self) -> list[Tool]:
        """Union every server's tools, recording where each one came from."""
        await self.aconnect()
        tools: list[Tool] = []
        for client in self.clients:
            try:
                discovered = await client.alist_tools()
            except Exception as exc:
                self._log.warning("Tool discovery failed on %s: %s", client.server, exc)
                continue
            for item in discovered:
                self._routes[item.name] = client
                tools.append(item)
        return tools

    async def acall_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Route a call to the server that advertised the tool.

        Raises:
            MCPError: When no server owns the tool.
        """
        client = self._routes.get(name)
        if client is None:
            await self.alist_tools()
            client = self._routes.get(name)
        if client is None:
            raise MCPError(
                f"No MCP server provides a tool named {name!r}.",
                context={"available": sorted(self._routes)},
            )
        return await client.acall_tool(name, arguments)

    async def alist_resources(self) -> list[MCPResource]:
        """Union every server's resources."""
        await self.aconnect()
        resources: list[MCPResource] = []
        for client in self.clients:
            try:
                resources.extend(await client.alist_resources())
            except Exception as exc:
                self._log.debug("Resource listing failed on %s: %s", client.server, exc)
        return resources

    async def alist_prompts(self) -> list[MCPPrompt]:
        """Union every server's prompts."""
        await self.aconnect()
        prompts: list[MCPPrompt] = []
        for client in self.clients:
            try:
                prompts.extend(await client.alist_prompts())
            except Exception as exc:
                self._log.debug("Prompt listing failed on %s: %s", client.server, exc)
        return prompts


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read an attribute or dict key, whichever the MCP SDK happens to return."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_dict(obj: Any) -> dict[str, Any]:
    """Coerce an MCP argument descriptor into a plain dict."""
    if isinstance(obj, dict):
        return obj
    return {
        "name": _attr(obj, "name", ""),
        "description": _attr(obj, "description", "") or "",
        "required": bool(_attr(obj, "required", False)),
    }


def _strip_namespace(name: str, server: str) -> str:
    """Remove the ``server_`` prefix Windlass adds when namespacing is on."""
    prefix = f"{server.replace('-', '_').replace(' ', '_')}_"
    return name[len(prefix) :] if name.startswith(prefix) else name


def _unwrap_content(result: Any) -> Any:
    """Unwrap an MCP content envelope into a plain Python value.

    MCP results are lists of typed content blocks. Callers want the value, not
    the envelope, so text blocks collapse to a string (parsed as JSON when they
    contain JSON) and structured content passes through.

    Args:
        result: Whatever the MCP SDK returned.

    Returns:
        The unwrapped value.

    Example:
        >>> _unwrap_content([{"type": "text", "text": '{"ok": true}'}])
        {'ok': True}
        >>> _unwrap_content([{"type": "text", "text": "plain"}])
        'plain'
    """
    payload = _attr(result, "content", None)
    if payload is None:
        payload = _attr(result, "structuredContent", None)
    if payload is None:
        payload = result

    if isinstance(payload, list):
        texts: list[str] = []
        others: list[Any] = []
        for block in payload:
            kind = _attr(block, "type", None)
            if kind == "text" or (kind is None and isinstance(block, str)):
                texts.append(_attr(block, "text", block) if kind else str(block))
            else:
                others.append(_attr(block, "data", block))
        if texts and not others:
            joined = "\n".join(t for t in texts if t)
            try:
                return json.loads(joined)
            except (ValueError, TypeError):
                return joined
        if others and not texts:
            return others[0] if len(others) == 1 else others
        return {"text": "\n".join(texts), "data": others} if others else ""

    return payload
