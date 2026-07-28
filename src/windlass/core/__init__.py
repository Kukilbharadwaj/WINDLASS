"""Windlass core: the framework machinery that knows nothing about AI.

This package holds the parts every component depends on and that depend on
nothing themselves:

* :mod:`~windlass.core.types` — the shared data model (``Document``, ``Chunk``,
  ``Message``, ``Completion``, ...).
* :mod:`~windlass.core.exceptions` — one error hierarchy for the whole framework.
* :mod:`~windlass.core.registry` — name-to-implementation lookup and plugins.
* :mod:`~windlass.core.container` — dependency injection.
* :mod:`~windlass.core.config` — settings from env, file and code.
* :mod:`~windlass.core.lazy` — optional dependency loading with good errors.
* :mod:`~windlass.core.concurrency` — async/sync bridging and bounded parallelism.
* :mod:`~windlass.core.retry`, :mod:`~windlass.core.cache` — reliability and speed.
* :mod:`~windlass.core.text`, :mod:`~windlass.core.vectors` — shared primitives.

Importing this package is cheap and pulls in no optional dependency.
"""

from __future__ import annotations

from windlass.core.cache import Cache, MemoryCache, NullCache, build_cache, make_key
from windlass.core.concurrency import batched, gather_bounded, iter_sync, map_async, run_sync
from windlass.core.config import WindlassSettings, configure, reset_settings, settings
from windlass.core.container import Container, root_container
from windlass.core.exceptions import (
    AgentError,
    AuthenticationError,
    ComponentNotFoundError,
    ConfigurationError,
    EvaluationError,
    GuardrailViolation,
    IngestionError,
    MaxIterationsExceeded,
    MCPError,
    MissingDependencyError,
    PipelineError,
    PluginError,
    ProviderError,
    RateLimitError,
    RegistryError,
    RetrievalError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ValidationError,
    WindlassError,
)
from windlass.core.lazy import is_available, require
from windlass.core.logging import configure_logging, get_logger
from windlass.core.registry import REGISTRY, ComponentSpec, Registry, available, create, register
from windlass.core.retry import retry_async, retry_sync, with_retry
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

__all__ = [
    "REGISTRY",
    "AgentError",
    "AgentResponse",
    "AgentStep",
    "AuthenticationError",
    "Cache",
    "Chunk",
    "Completion",
    "ComponentNotFoundError",
    "ComponentSpec",
    "ConfigurationError",
    "Container",
    "Document",
    "Embedding",
    "EvaluationError",
    "EvaluationReport",
    "EvaluationResult",
    "FinishReason",
    "GuardrailResult",
    "GuardrailViolation",
    "IngestionError",
    "MCPError",
    "MaxIterationsExceeded",
    "MemoryCache",
    "Message",
    "MissingDependencyError",
    "NullCache",
    "PipelineError",
    "PluginError",
    "ProviderError",
    "RAGAnswer",
    "RateLimitError",
    "Registry",
    "RegistryError",
    "RetrievalError",
    "Role",
    "ScoredChunk",
    "SearchResult",
    "StreamEvent",
    "ToolCall",
    "ToolError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolResult",
    "Usage",
    "ValidationError",
    "WindlassError",
    "WindlassSettings",
    "available",
    "batched",
    "build_cache",
    "configure",
    "configure_logging",
    "create",
    "gather_bounded",
    "get_logger",
    "is_available",
    "iter_sync",
    "make_key",
    "map_async",
    "register",
    "require",
    "reset_settings",
    "retry_async",
    "retry_sync",
    "root_container",
    "run_sync",
    "settings",
    "with_retry",
]
