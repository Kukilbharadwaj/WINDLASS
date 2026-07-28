"""Canonical data model shared by every Windlass component.

These types are the *lingua franca* of the framework. A chunker produced by one
plugin can feed a retriever from another because both speak :class:`Chunk`.
Every model is a Pydantic v2 model, so it validates, serialises to JSON, and
round-trips through checkpoints for free.

The rule of thumb: **provider-specific shapes never cross a component boundary.**
Adapters translate into these types on the way in and out, and keep the untouched
provider payload on ``.raw`` for callers who need it (Level 3 access).
"""

from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "AgentResponse",
    "AgentStep",
    "Chunk",
    "Completion",
    "Document",
    "Embedding",
    "EvaluationReport",
    "EvaluationResult",
    "FinishReason",
    "GuardrailResult",
    "Message",
    "RAGAnswer",
    "Role",
    "ScoredChunk",
    "SearchResult",
    "StreamEvent",
    "ToolCall",
    "ToolResult",
    "Usage",
    "WindlassModel",
    "content_id",
    "new_id",
]


def new_id(prefix: str = "") -> str:
    """Return a short unique identifier.

    Args:
        prefix: Optional namespace prefix, e.g. ``"doc"`` yields ``"doc_1f3a..."``.

    Returns:
        A URL-safe identifier that is unique per process invocation.
    """
    raw = uuid.uuid4().hex[:16]
    return f"{prefix}_{raw}" if prefix else raw


def content_id(text: str, prefix: str = "") -> str:
    """Return a deterministic identifier derived from ``text``.

    Deterministic ids make ingestion idempotent: re-ingesting the same document
    overwrites the same rows instead of creating duplicates.

    Args:
        text: The content to hash.
        prefix: Optional namespace prefix.

    Returns:
        A stable 16-character BLAKE2b digest, optionally prefixed.
    """
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
    return f"{prefix}_{digest}" if prefix else digest


class Role(StrEnum):
    """Who authored a :class:`Message`.

    A :class:`~enum.StrEnum`, so members compare and format as their plain
    string value — ``Role.USER == "user"`` and ``f"{Role.USER}"`` both do what
    you would expect when the value reaches a provider's wire format.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(StrEnum):
    """Why the model stopped generating."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


class WindlassModel(BaseModel):
    """Base model with the conventions every Windlass type shares.

    * Extra keys are ignored rather than rejected, so a newer provider adapter
      can hand richer payloads to an older consumer without breaking it.
    * Arbitrary types are allowed so ``raw`` can hold native SDK objects.
    * Enum values serialise to their string form for clean JSON.
    """

    model_config = ConfigDict(
        extra="ignore",
        arbitrary_types_allowed=True,
        use_enum_values=False,
        populate_by_name=True,
        validate_assignment=False,
    )


class Usage(WindlassModel):
    """Token accounting for one or more model calls.

    Attributes:
        prompt_tokens: Tokens consumed by the prompt.
        completion_tokens: Tokens produced by the model.
        total_tokens: Sum of the two; computed when a provider omits it.
        cached_tokens: Prompt tokens served from the provider's prompt cache.
        cost_usd: Estimated spend, when the adapter can compute it.
        calls: How many provider round-trips this usage aggregates.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float | None = None
    calls: int = 1

    @model_validator(mode="after")
    def _fill_total(self) -> Self:
        if not self.total_tokens:
            object.__setattr__(self, "total_tokens", self.prompt_tokens + self.completion_tokens)
        return self

    def __add__(self, other: Usage) -> Usage:
        """Aggregate two usage records.

        Example:
            >>> Usage(prompt_tokens=10) + Usage(prompt_tokens=5, completion_tokens=2)
            Usage(prompt_tokens=15, completion_tokens=2, total_tokens=17, ...)
        """
        cost: float | None
        if self.cost_usd is None and other.cost_usd is None:
            cost = None
        else:
            cost = (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cost_usd=cost,
            calls=self.calls + other.calls,
        )


class ToolCall(WindlassModel):
    """A model's request to invoke a tool.

    Attributes:
        id: Provider-assigned correlation id; echoed back in the tool result.
        name: Registered tool name.
        arguments: Parsed keyword arguments for the tool.
        raw_arguments: The original JSON string, kept when parsing failed.
    """

    id: str = Field(default_factory=lambda: new_id("call"))
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str | None = None


class ToolResult(WindlassModel):
    """The outcome of executing a :class:`ToolCall`.

    Attributes:
        call_id: The :attr:`ToolCall.id` this answers.
        name: Tool that produced the result.
        content: String rendering handed back to the model.
        data: The unserialised Python return value, for programmatic access.
        is_error: True when the tool raised and the error text was returned.
        duration_ms: Wall-clock execution time.
    """

    call_id: str
    name: str
    content: str
    data: Any = None
    is_error: bool = False
    duration_ms: float = 0.0


class Message(WindlassModel):
    """One turn in a conversation.

    Attributes:
        role: Author of the message.
        content: Text body. May be empty when the turn is purely tool calls.
        tool_calls: Tools the assistant wants to invoke.
        tool_call_id: Set on ``tool`` messages to correlate with a call.
        name: Optional speaker name (multi-agent transcripts use this).
        metadata: Free-form annotations that never reach the provider.

    Example:
        >>> Message.user("hello")
        Message(role=<Role.USER: 'user'>, content='hello', ...)
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("role", mode="before")
    @classmethod
    def _coerce_role(cls, value: Any) -> Any:
        if isinstance(value, str):
            return Role(value.lower())
        return value

    # -- ergonomic constructors ------------------------------------------
    @classmethod
    def system(cls, content: str, **kw: Any) -> Message:
        """Build a system message."""
        return cls(role=Role.SYSTEM, content=content, **kw)

    @classmethod
    def user(cls, content: str, **kw: Any) -> Message:
        """Build a user message."""
        return cls(role=Role.USER, content=content, **kw)

    @classmethod
    def assistant(cls, content: str = "", **kw: Any) -> Message:
        """Build an assistant message."""
        return cls(role=Role.ASSISTANT, content=content, **kw)

    @classmethod
    def tool(cls, result: ToolResult) -> Message:
        """Build the ``tool`` message that answers a :class:`ToolCall`."""
        return cls(
            role=Role.TOOL,
            content=result.content,
            tool_call_id=result.call_id,
            name=result.name,
            metadata={"is_error": result.is_error},
        )

    def __str__(self) -> str:
        role = self.role.value if isinstance(self.role, Role) else str(self.role)
        if self.tool_calls and not self.content:
            calls = ", ".join(f"{c.name}({c.arguments})" for c in self.tool_calls)
            return f"{role}: -> {calls}"
        return f"{role}: {self.content}"


def normalize_messages(
    prompt: str | Message | Sequence[Message] | Sequence[dict[str, Any]],
) -> list[Message]:
    """Coerce any accepted prompt shape into a list of :class:`Message`.

    Accepts a bare string, a single message, a sequence of messages, or a
    sequence of OpenAI-style dicts. This is what lets every Windlass entry point
    take ``"hello"`` as well as a full transcript.

    Args:
        prompt: The prompt in any supported shape.

    Returns:
        A new list of validated messages.

    Raises:
        TypeError: If the input is not one of the supported shapes.
    """
    if isinstance(prompt, str):
        return [Message.user(prompt)]
    if isinstance(prompt, Message):
        return [prompt]
    if isinstance(prompt, Sequence):
        out: list[Message] = []
        for item in prompt:
            if isinstance(item, Message):
                out.append(item)
            elif isinstance(item, dict):
                out.append(Message.model_validate(item))
            else:
                raise TypeError(f"Cannot interpret {type(item).__name__} as a Message.")
        return out
    raise TypeError(f"Cannot interpret {type(prompt).__name__} as a prompt.")


class Completion(WindlassModel):
    """A non-streaming model response.

    Attributes:
        content: The generated text.
        tool_calls: Tools the model wants invoked, if any.
        finish_reason: Why generation stopped.
        model: Model identifier reported by the provider.
        usage: Token accounting.
        raw: The untouched provider response object (Level 3 escape hatch).
        latency_ms: Wall-clock time for the call.
    """

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: FinishReason = FinishReason.STOP
    model: str = ""
    usage: Usage = Field(default_factory=Usage)
    raw: Any = Field(default=None, repr=False, exclude=True)
    latency_ms: float = 0.0

    def to_message(self) -> Message:
        """Render this completion as an assistant :class:`Message`."""
        return Message(role=Role.ASSISTANT, content=self.content, tool_calls=self.tool_calls)

    def __str__(self) -> str:
        return self.content


class StreamEvent(WindlassModel):
    """One incremental update from a streaming call.

    Attributes:
        type: ``text`` for content deltas, ``tool_call`` when a call is complete,
            ``usage`` for accounting, ``done`` for the terminal event.
        delta: Incremental text for ``text`` events.
        tool_call: The completed call for ``tool_call`` events.
        finish_reason: Present on the ``done`` event.
        usage: Present on ``usage`` and ``done`` events when the provider reports it.
        raw: Untouched provider chunk.
    """

    type: Literal["text", "tool_call", "usage", "done"] = "text"
    delta: str = ""
    tool_call: ToolCall | None = None
    finish_reason: FinishReason | None = None
    usage: Usage | None = None
    raw: Any = Field(default=None, repr=False, exclude=True)

    def __str__(self) -> str:
        return self.delta


class Document(WindlassModel):
    """A source document before chunking.

    Attributes:
        id: Stable identifier. Defaults to a content hash so re-ingestion is
            idempotent.
        content: Extracted plain text.
        metadata: Anything worth filtering or citing on later (page, author,
            section, url, ...). Kept alongside every chunk derived from it.
        source: Where the document came from (path, URL, ...).
        mimetype: Detected content type, when known.
        created_at: Unix timestamp of ingestion.
    """

    id: str = ""
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None
    mimetype: str | None = None
    created_at: float = Field(default_factory=time.time)

    @model_validator(mode="after")
    def _default_id(self) -> Self:
        if not self.id:
            seed = f"{self.source or ''}::{self.content}"
            object.__setattr__(self, "id", content_id(seed, "doc"))
        return self

    def __len__(self) -> int:
        return len(self.content)


class Chunk(WindlassModel):
    """A retrievable slice of a :class:`Document`.

    Attributes:
        id: Stable identifier, derived from the document id and index.
        content: The chunk text as it will be embedded and shown to the model.
        metadata: Document metadata plus chunk-level annotations.
        document_id: The document this came from.
        index: Zero-based position within the document.
        embedding: Optional dense vector; populated during ingestion.
        parent_id: Set by hierarchical chunkers; points at the larger parent
            chunk that should be expanded at retrieval time.
        start_char: Character offset in the source document.
        end_char: Exclusive end offset in the source document.
    """

    id: str = ""
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    document_id: str = ""
    index: int = 0
    embedding: list[float] | None = Field(default=None, repr=False)
    parent_id: str | None = None
    start_char: int = 0
    end_char: int = 0

    @model_validator(mode="after")
    def _defaults(self) -> Self:
        if not self.id:
            object.__setattr__(
                self, "id", content_id(f"{self.document_id}:{self.index}:{self.content}", "chk")
            )
        if not self.end_char:
            object.__setattr__(self, "end_char", self.start_char + len(self.content))
        return self

    def __len__(self) -> int:
        return len(self.content)

    def with_embedding(self, vector: list[float]) -> Chunk:
        """Return a copy carrying ``vector`` as its embedding."""
        return self.model_copy(update={"embedding": vector})


class ScoredChunk(WindlassModel):
    """A chunk together with its retrieval score.

    Attributes:
        chunk: The retrieved chunk.
        score: Relevance score. Higher is better; the scale depends on the
            retriever (cosine similarity, BM25, reciprocal rank fusion, ...).
        rank: One-based position in the result list.
        retriever: Which retriever produced this hit. Hybrid search keeps this
            so you can see which leg contributed each result.
    """

    chunk: Chunk
    score: float = 0.0
    rank: int = 0
    retriever: str = ""

    @property
    def content(self) -> str:
        """Shortcut to ``chunk.content``."""
        return self.chunk.content

    @property
    def metadata(self) -> dict[str, Any]:
        """Shortcut to ``chunk.metadata``."""
        return self.chunk.metadata


class Embedding(WindlassModel):
    """A dense vector plus the text it encodes.

    Attributes:
        vector: The embedding values.
        text: The text that was embedded.
        model: Model identifier that produced the vector.
    """

    vector: list[float]
    text: str = ""
    model: str = ""

    def __len__(self) -> int:
        return len(self.vector)


class SearchResult(WindlassModel):
    """The full result of a retrieval call.

    Attributes:
        query: The query that was executed (post rewriting, if any).
        hits: Ranked chunks.
        total: Number of candidates considered before truncation.
        latency_ms: Wall-clock retrieval time.
    """

    query: str
    hits: list[ScoredChunk] = Field(default_factory=list)
    total: int = 0
    latency_ms: float = 0.0

    def __iter__(self) -> Any:  # type: ignore[override]
        return iter(self.hits)

    def __len__(self) -> int:
        return len(self.hits)

    def __getitem__(self, index: int) -> ScoredChunk:
        return self.hits[index]

    def texts(self) -> list[str]:
        """Return just the chunk texts, ranked."""
        return [h.chunk.content for h in self.hits]


class RAGAnswer(WindlassModel):
    """The answer produced by a RAG pipeline.

    Attributes:
        answer: The generated answer text.
        question: The question that was asked.
        contexts: Chunks that were placed in the prompt, in prompt order.
        usage: Token accounting for the generation call.
        latency_ms: End-to-end wall-clock time including retrieval.
        metadata: Pipeline annotations (guardrail decisions, rewrites, ...).
    """

    answer: str = ""
    question: str = ""
    contexts: list[ScoredChunk] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def sources(self) -> list[str]:
        """Unique, order-preserving list of the sources cited in the context."""
        seen: dict[str, None] = {}
        for hit in self.contexts:
            src = hit.chunk.metadata.get("source")
            if src and src not in seen:
                seen[str(src)] = None
        return list(seen)

    def __str__(self) -> str:
        return self.answer


class AgentStep(WindlassModel):
    """One iteration of an agent's reason/act loop.

    Attributes:
        index: Zero-based step number.
        thought: Assistant text emitted during this step, if any.
        tool_calls: Tools the model requested.
        tool_results: Results of executing them.
        usage: Tokens spent on this step.
        node: Name of the graph node that ran (multi-agent / LangGraph).
    """

    index: int = 0
    thought: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    node: str = ""


class AgentResponse(WindlassModel):
    """The final result of an agent run.

    Attributes:
        output: The agent's answer.
        messages: Full transcript including tool traffic.
        steps: Structured per-iteration trace.
        usage: Aggregate token accounting across the whole run.
        latency_ms: End-to-end wall-clock time.
        thread_id: Checkpoint thread, when checkpointing is enabled.
        interrupted: True when the run paused for human approval.
        metadata: Runtime annotations.
    """

    output: str = ""
    messages: list[Message] = Field(default_factory=list)
    steps: list[AgentStep] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    thread_id: str | None = None
    interrupted: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Every tool call made during the run, in order."""
        return [call for step in self.steps for call in step.tool_calls]

    def __str__(self) -> str:
        return self.output


class GuardrailResult(WindlassModel):
    """Verdict from a guardrail check.

    Attributes:
        allowed: False when the content must be blocked.
        content: Possibly rewritten content (redaction returns a sanitised copy).
        detections: One entry per rule that fired.
        rule: The first rule that blocked, if any.
        stage: ``input`` or ``output``.
    """

    allowed: bool = True
    content: str = ""
    detections: list[dict[str, Any]] = Field(default_factory=list)
    rule: str | None = None
    stage: str = "input"

    @property
    def modified(self) -> bool:
        """True when the guardrail rewrote the content."""
        return bool(self.detections) and self.allowed


class EvaluationResult(WindlassModel):
    """Score for a single metric on a single sample.

    Attributes:
        metric: Metric name, e.g. ``faithfulness``.
        score: Normalised score, conventionally in ``[0, 1]``.
        passed: Whether the score met the metric's threshold.
        threshold: The threshold that was applied.
        reason: Explanation, when the metric produces one.
        sample_id: Identifier of the evaluated sample.
    """

    metric: str
    score: float = 0.0
    passed: bool = True
    threshold: float | None = None
    reason: str = ""
    sample_id: str = ""


class EvaluationReport(WindlassModel):
    """Aggregate results of an evaluation run.

    Attributes:
        results: Every individual metric/sample score.
        summary: Mean score per metric.
        pass_rate: Fraction of results that passed their threshold.
        samples: Number of samples evaluated.
    """

    results: list[EvaluationResult] = Field(default_factory=list)
    summary: dict[str, float] = Field(default_factory=dict)
    pass_rate: float = 0.0
    samples: int = 0

    @model_validator(mode="after")
    def _summarise(self) -> Self:
        if self.results and not self.summary:
            buckets: dict[str, list[float]] = {}
            for res in self.results:
                buckets.setdefault(res.metric, []).append(res.score)
            object.__setattr__(
                self,
                "summary",
                {name: sum(vals) / len(vals) for name, vals in buckets.items()},
            )
        if self.results and not self.pass_rate:
            passed = sum(1 for r in self.results if r.passed)
            object.__setattr__(self, "pass_rate", passed / len(self.results))
        return self

    def __str__(self) -> str:
        if not self.summary:
            return "EvaluationReport(empty)"
        rows = "\n".join(f"  {k:<24} {v:.3f}" for k, v in sorted(self.summary.items()))
        return f"EvaluationReport({self.samples} samples, pass_rate={self.pass_rate:.1%})\n{rows}"
