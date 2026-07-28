"""Windlass — the modular AI application framework.

One elegant API for agents, RAG, tools, MCP, guardrails, evaluation and
observability. Windlass is not a wrapper around other libraries; it is an
architecture that lets them be swapped, extended and replaced without rewriting
your application.

Quick start::

    from windlass import Windlass

    rag = Windlass.rag()
    rag.ingest("./documents")
    print(rag.ask("What changed in the API last quarter?"))

    agent = Windlass.agent().llm("openai").tool(my_function).memory()
    print(agent.run("Summarise my open tickets"))

Three levels of API, always:

1. **Simple** — ``rag.ask("...")`` works with sensible defaults.
2. **Configured** — ``rag.chunker("semantic", chunk_size=1000, overlap=200)``.
3. **Native** — ``rag.native_store()``, ``agent.native_graph()``. Windlass
   simplifies libraries; it never hides them.

Importing this module is cheap and pulls in no optional dependency. Providers are
registered by dotted path and imported only when you actually use them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from windlass._version import __version__
from windlass.api import Windlass
from windlass.core.config import WindlassSettings, configure, reset_settings, settings
from windlass.core.container import Container, root_container
from windlass.core.exceptions import (
    AgentError,
    AuthenticationError,
    ComponentNotFoundError,
    ConfigurationError,
    DuplicateComponentError,
    EvaluationError,
    GuardrailViolation,
    IngestionError,
    MaxIterationsExceeded,
    MCPError,
    MissingDependencyError,
    PipelineError,
    PluginError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
    RegistryError,
    ResponseError,
    RetrievalError,
    SerializationError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ValidationError,
    WindlassError,
)
from windlass.core.exceptions import InterruptedError_ as AgentInterrupt
from windlass.core.exceptions import MemoryError_ as WindlassMemoryError
from windlass.core.logging import configure_logging, get_logger
from windlass.core.registry import REGISTRY, ComponentSpec, Registry, available, create, register
from windlass.core.types import (
    AgentResponse,
    AgentStep,
    Chunk,
    Completion,
    Document,
    Embedding,
    EvaluationReport,
    EvaluationResult,
    FinishReason,
    GuardrailResult,
    Message,
    RAGAnswer,
    Role,
    ScoredChunk,
    SearchResult,
    StreamEvent,
    ToolCall,
    ToolResult,
    Usage,
)
from windlass.interfaces import (
    LLM,
    Chunker,
    Component,
    Embedder,
    EvalSample,
    Evaluator,
    Guardrail,
    Loader,
    MCPClient,
    Memory,
    Preprocessor,
    Retriever,
    Span,
    Tool,
    Tracer,
    VectorStore,
)
from windlass.providers import register_builtins
from windlass.tools import FunctionTool, ToolRegistry, tool

if TYPE_CHECKING:  # pragma: no cover - import-time cost avoided at runtime
    from windlass.agent import AgentBuilder, AgentRuntime, Supervisor
    from windlass.rag import RAGBuilder, RAGPipeline

# Register every built-in component by dotted path. No provider module is
# imported here — that happens on first use.
register_builtins()

__all__ = [
    "LLM",
    "REGISTRY",
    "AgentBuilder",
    "AgentError",
    "AgentInterrupt",
    "AgentResponse",
    "AgentRuntime",
    "AgentStep",
    "AuthenticationError",
    "Chunk",
    "Chunker",
    "Completion",
    "Component",
    "ComponentNotFoundError",
    "ComponentSpec",
    "ConfigurationError",
    "Container",
    "Document",
    "DuplicateComponentError",
    "Embedder",
    "Embedding",
    "EvalSample",
    "EvaluationError",
    "EvaluationReport",
    "EvaluationResult",
    "Evaluator",
    "FinishReason",
    "FunctionTool",
    "Guardrail",
    "GuardrailResult",
    "GuardrailViolation",
    "IngestionError",
    "Loader",
    "MCPClient",
    "MCPError",
    "MaxIterationsExceeded",
    "Memory",
    "Message",
    "MissingDependencyError",
    "PipelineError",
    "PluginError",
    "Preprocessor",
    "ProviderError",
    "ProviderTimeoutError",
    "RAGAnswer",
    "RAGBuilder",
    "RAGPipeline",
    "RateLimitError",
    "Registry",
    "RegistryError",
    "ResponseError",
    "RetrievalError",
    "Retriever",
    "Role",
    "ScoredChunk",
    "SearchResult",
    "SerializationError",
    "Span",
    "StreamEvent",
    "Supervisor",
    "Tool",
    "ToolCall",
    "ToolError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolRegistry",
    "ToolResult",
    "Tracer",
    "Usage",
    "ValidationError",
    "VectorStore",
    "Windlass",
    "WindlassError",
    "WindlassMemoryError",
    "WindlassSettings",
    "__version__",
    "available",
    "configure",
    "configure_logging",
    "create",
    "get_logger",
    "register",
    "reset_settings",
    "root_container",
    "settings",
    "tool",
]

#: Builder classes exported lazily, so ``import windlass`` never pays for the
#: agent or RAG subpackages until something actually touches them.
_LAZY: dict[str, tuple[str, str]] = {
    "AgentBuilder": ("windlass.agent.builder", "AgentBuilder"),
    "AgentRuntime": ("windlass.agent.runtime", "AgentRuntime"),
    "Supervisor": ("windlass.agent.supervisor", "Supervisor"),
    "RAGBuilder": ("windlass.rag.builder", "RAGBuilder"),
    "RAGPipeline": ("windlass.rag.pipeline", "RAGPipeline"),
}


def __getattr__(name: str) -> Any:
    """Resolve lazily exported names on first access.

    Args:
        name: The attribute being looked up.

    Returns:
        The resolved object.

    Raises:
        AttributeError: When the name is not exported by this package.
    """
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'windlass' has no attribute {name!r}")
    import importlib

    module, attribute = target
    resolved = getattr(importlib.import_module(module), attribute)
    globals()[name] = resolved
    return resolved


def __dir__() -> list[str]:  # pragma: no cover - REPL nicety
    return sorted(__all__)
