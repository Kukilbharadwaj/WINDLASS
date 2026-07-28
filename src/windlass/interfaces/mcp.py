"""The Model Context Protocol (MCP) interface.

MCP servers expose three things: **tools** an agent can call, **resources** it
can read, and **prompts** it can instantiate. Windlass treats an MCP server as
one more source of tools — after connecting, its tools appear alongside your
local Python functions and the agent cannot tell them apart.

Implementers override :meth:`MCPClient.aconnect`, :meth:`MCPClient.alist_tools`
and :meth:`MCPClient.acall_tool`; resources and prompts are optional.

Example:
    >>> from windlass.providers.mcp.fastmcp import StaticMCPClient
    >>> client = StaticMCPClient(tools={"echo": lambda text: text})
    >>> client.connect()
    >>> [t.name for t in client.list_tools()]
    ['echo']
"""

from __future__ import annotations

import abc
from typing import Any

from pydantic import Field

from windlass.core.concurrency import run_sync
from windlass.core.exceptions import MCPError
from windlass.core.types import WindlassModel
from windlass.interfaces.base import Component
from windlass.interfaces.tool import Tool

__all__ = ["MCPClient", "MCPPrompt", "MCPResource", "MCPToolProxy"]


class MCPResource(WindlassModel):
    """A readable resource advertised by an MCP server.

    Attributes:
        uri: Resource identifier, e.g. ``file:///docs/readme.md``.
        name: Human readable name.
        description: What the resource contains.
        mimetype: Content type, when the server declares one.
        server: Which server advertised it.
    """

    uri: str
    name: str = ""
    description: str = ""
    mimetype: str | None = None
    server: str = ""


class MCPPrompt(WindlassModel):
    """A parameterised prompt template advertised by an MCP server.

    Attributes:
        name: Prompt identifier.
        description: What the prompt is for.
        arguments: Declared arguments, each a dict with ``name``, ``description``
            and ``required``.
        server: Which server advertised it.
    """

    name: str
    description: str = ""
    arguments: list[dict[str, Any]] = Field(default_factory=list)
    server: str = ""


class MCPToolProxy(Tool):
    """A local :class:`~windlass.interfaces.tool.Tool` backed by a remote MCP tool.

    Created by :meth:`MCPClient.alist_tools`. Calling it forwards to the server
    and returns whatever comes back, so agents bind it exactly like a local
    function.

    Args:
        client: The MCP client that owns the connection.
        name: Remote tool name.
        description: Remote tool description.
        parameters: Remote JSON Schema for the arguments.
        server: Server label, kept in metadata and used to disambiguate
            same-named tools across servers.
        **config: Forwarded to :class:`~windlass.interfaces.tool.Tool`.
    """

    provider_name = "mcp"

    def __init__(
        self,
        client: MCPClient,
        *,
        name: str,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        server: str = "",
        **config: Any,
    ) -> None:
        super().__init__(
            name=name,
            description=description or f"MCP tool {name!r} from {server or 'server'}.",
            parameters=parameters,
            **config,
        )
        self.client = client
        self.server = server

    async def acall(self, **kwargs: Any) -> Any:
        """Forward the call to the MCP server.

        Args:
            **kwargs: Arguments matching the remote schema.

        Returns:
            The server's result.

        Raises:
            MCPError: When the call fails or the server is unreachable.
        """
        return await self.client.acall_tool(self.name, kwargs)

    def native(self) -> Any:
        """Return the underlying MCP client session."""
        return self.client.native()


class MCPClient(Component):
    """Abstract MCP client.

    Args:
        server: Server label used in logs and to namespace tools.
        namespace: When True, remote tools are exposed as ``server_toolname`` so
            two servers offering ``search`` do not collide.
        timeout: Per-call timeout in seconds.
        name: Component name.
        **config: Transport-specific options (command, args, url, env, ...).

    Attributes:
        connected: Whether the transport is currently open.

    Example:
        Implementing a client means three methods::

            class MyClient(MCPClient):
                provider_name = "mine"

                async def aconnect(self): ...
                async def alist_tools(self): ...
                async def acall_tool(self, name, arguments): ...
    """

    kind = "mcp"
    provider_name: str = "mcp"

    def __init__(
        self,
        *,
        server: str = "",
        namespace: bool = False,
        timeout: float = 30.0,
        name: str | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            name=name or server or self.provider_name,
            server=server,
            namespace=namespace,
            timeout=timeout,
            **config,
        )
        self.server = server or self.provider_name
        self.namespace = namespace
        self.timeout = timeout
        self.connected = False

    # -- provider hooks ---------------------------------------------------
    @abc.abstractmethod
    async def aconnect(self) -> None:
        """Open the transport and perform the MCP handshake.

        Must be idempotent — calling it twice should not open two sessions.

        Raises:
            MCPError: When the server cannot be reached or the handshake fails.
        """

    @abc.abstractmethod
    async def alist_tools(self) -> list[Tool]:
        """Discover the server's tools.

        Returns:
            :class:`MCPToolProxy` instances ready to bind to an agent.

        Raises:
            MCPError: When discovery fails.
        """

    @abc.abstractmethod
    async def acall_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Invoke a remote tool.

        Args:
            name: Tool name as advertised by the server (without any namespace
                prefix Windlass may have added).
            arguments: Arguments matching the remote schema.

        Returns:
            The server's result, unwrapped from the MCP content envelope.

        Raises:
            MCPError: When the call fails.
        """

    async def alist_resources(self) -> list[MCPResource]:
        """Discover readable resources. Returns ``[]`` when unsupported."""
        return []

    async def aread_resource(self, uri: str) -> str:
        """Read a resource's contents.

        Args:
            uri: The resource identifier.

        Returns:
            The resource body as text.

        Raises:
            MCPError: When the resource cannot be read.
        """
        raise MCPError(
            f"{type(self).__name__} does not support reading resources.",
            context={"uri": uri},
        )

    async def alist_prompts(self) -> list[MCPPrompt]:
        """Discover prompt templates. Returns ``[]`` when unsupported."""
        return []

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
        raise MCPError(f"{type(self).__name__} does not support prompts.", context={"prompt": name})

    async def adisconnect(self) -> None:
        """Close the transport. Safe to call when already closed."""
        self.connected = False

    # -- sync API ---------------------------------------------------------
    def connect(self) -> None:
        """Blocking :meth:`aconnect`."""
        run_sync(self.aconnect())

    def list_tools(self) -> list[Tool]:
        """Blocking :meth:`alist_tools`."""
        return run_sync(self.alist_tools())

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Blocking :meth:`acall_tool`."""
        return run_sync(self.acall_tool(name, arguments or {}))

    def list_resources(self) -> list[MCPResource]:
        """Blocking :meth:`alist_resources`."""
        return run_sync(self.alist_resources())

    def read_resource(self, uri: str) -> str:
        """Blocking :meth:`aread_resource`."""
        return run_sync(self.aread_resource(uri))

    def list_prompts(self) -> list[MCPPrompt]:
        """Blocking :meth:`alist_prompts`."""
        return run_sync(self.alist_prompts())

    def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Blocking :meth:`aget_prompt`."""
        return run_sync(self.aget_prompt(name, arguments))

    def disconnect(self) -> None:
        """Blocking :meth:`adisconnect`."""
        run_sync(self.adisconnect())

    async def aclose(self) -> None:
        """Alias for :meth:`adisconnect`, so containers can close uniformly."""
        await self.adisconnect()

    # -- helpers ----------------------------------------------------------
    def _tool_name(self, remote_name: str) -> str:
        """Apply the namespace policy to a remote tool name."""
        if not self.namespace:
            return remote_name
        prefix = self.server.replace("-", "_").replace(" ", "_")
        return f"{prefix}_{remote_name}"[:64]

    async def __aenter__(self) -> MCPClient:
        await self.aconnect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.adisconnect()

    def __repr__(self) -> str:
        state = "connected" if self.connected else "disconnected"
        return f"{type(self).__name__}(server={self.server!r}, {state})"
