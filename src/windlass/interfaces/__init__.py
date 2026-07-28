"""Windlass interfaces — the contracts every component implements.

This package is the architectural heart of the framework. Everything else
depends on these abstractions and nothing depends on a concrete provider, which
is what the dependency-inversion principle buys us in practice:

* Swapping OpenAI for Ollama is a string change, not a refactor.
* A custom retriever written in a user's repo is indistinguishable from a
  built-in one.
* The whole framework can be exercised in tests with fakes that implement these
  same interfaces.

There is exactly one interface per component kind:

============  =========================================================
Kind          Interface
============  =========================================================
``llm``       :class:`~windlass.interfaces.llm.LLM`
``embedding`` :class:`~windlass.interfaces.embedding.Embedder`
``loader``    :class:`~windlass.interfaces.loader.Loader`
``preprocessor`` :class:`~windlass.interfaces.preprocessor.Preprocessor`
``chunker``   :class:`~windlass.interfaces.chunker.Chunker`
``retriever`` :class:`~windlass.interfaces.retriever.Retriever`
``vectordb``  :class:`~windlass.interfaces.vectordb.VectorStore`
``memory``    :class:`~windlass.interfaces.memory.Memory`
``guardrail`` :class:`~windlass.interfaces.guardrail.Guardrail`
``evaluator`` :class:`~windlass.interfaces.evaluator.Evaluator`
``tracer``    :class:`~windlass.interfaces.tracer.Tracer`
``tool``      :class:`~windlass.interfaces.tool.Tool`
``mcp``       :class:`~windlass.interfaces.mcp.MCPClient`
============  =========================================================

Every one of them shares :class:`~windlass.interfaces.base.Component`, which
supplies naming, configuration, lifecycle and the ``native()`` escape hatch.
"""

from __future__ import annotations

from windlass.interfaces.base import Component, SupportsNative
from windlass.interfaces.chunker import Chunker
from windlass.interfaces.embedding import Embedder
from windlass.interfaces.evaluator import EvalSample, Evaluator
from windlass.interfaces.guardrail import Guardrail, GuardrailChain
from windlass.interfaces.llm import LLM, PromptLike
from windlass.interfaces.loader import Loader, SourceLike
from windlass.interfaces.mcp import MCPClient, MCPPrompt, MCPResource, MCPToolProxy
from windlass.interfaces.memory import Memory
from windlass.interfaces.preprocessor import Preprocessor, PreprocessorChain
from windlass.interfaces.retriever import Retriever
from windlass.interfaces.tool import Tool
from windlass.interfaces.tracer import NullTracer, Span, Tracer
from windlass.interfaces.vectordb import MetadataFilter, VectorStore

__all__ = [
    "LLM",
    "Chunker",
    "Component",
    "Embedder",
    "EvalSample",
    "Evaluator",
    "Guardrail",
    "GuardrailChain",
    "Loader",
    "MCPClient",
    "MCPPrompt",
    "MCPResource",
    "MCPToolProxy",
    "Memory",
    "MetadataFilter",
    "NullTracer",
    "Preprocessor",
    "PreprocessorChain",
    "PromptLike",
    "Retriever",
    "SourceLike",
    "Span",
    "SupportsNative",
    "Tool",
    "Tracer",
    "VectorStore",
]
