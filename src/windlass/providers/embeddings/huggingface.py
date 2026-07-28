"""Hugging Face embedding models via ``sentence-transformers``.

Runs entirely locally: no API key, no per-token cost, and your documents never
leave the machine. The trade-off is a one-time model download and CPU/GPU time
per batch.

Install with::

    pip install "windlass[embeddings]"

Model choice matters more than anything else in a RAG stack. Sensible starting
points:

* ``BAAI/bge-small-en-v1.5`` — 384 dims, fast, strong for English (the default).
* ``BAAI/bge-m3`` — multilingual, 1024 dims, heavier.
* ``intfloat/multilingual-e5-large`` — multilingual, needs the ``query:`` /
  ``passage:`` prefixes this adapter applies for you.

Example:
    >>> from windlass import Windlass                                   # doctest: +SKIP
    >>> emb = Windlass.embedding("huggingface")                        # doctest: +SKIP
    >>> len(emb.embed_one("hello"))                                   # doctest: +SKIP
    384
"""

from __future__ import annotations

from typing import Any

from windlass.core.concurrency import to_thread
from windlass.core.exceptions import ProviderError
from windlass.core.lazy import require
from windlass.core.registry import register
from windlass.interfaces.embedding import Embedder

__all__ = ["HuggingFaceEmbedder"]

#: Models that expect asymmetric instruction prefixes on queries vs documents.
_PREFIX_RULES: dict[str, tuple[str, str]] = {
    "e5": ("query: ", "passage: "),
    "bge": ("Represent this sentence for searching relevant passages: ", ""),
    "gte": ("", ""),
    "nomic": ("search_query: ", "search_document: "),
}


@register.embedding(
    "huggingface",
    aliases=("hf", "sentence-transformers", "st"),
    description="Local sentence-transformers embedding models.",
)
class HuggingFaceEmbedder(Embedder):
    """Local embeddings from a ``sentence-transformers`` model.

    Args:
        model: Model id on the Hugging Face Hub.
        device: ``"cpu"``, ``"cuda"``, ``"mps"``, or ``None`` to auto-detect.
        trust_remote_code: Allow the model repo to execute its own code. Only
            enable this for repos you trust.
        auto_prefix: Apply the model family's query/document instruction
            prefixes automatically. Turn it off if you prefix text yourself.
        model_kwargs: Extra keyword arguments for ``SentenceTransformer``.
        encode_kwargs: Extra keyword arguments for ``.encode()``.
        **config: Forwarded to :class:`~windlass.interfaces.embedding.Embedder`.

    Raises:
        MissingDependencyError: When ``sentence-transformers`` is not installed.
        ProviderError: When the model cannot be loaded.

    Performance:
        The model is loaded once and reused. Encoding runs on a worker thread so
        it never blocks the event loop, and batching is handled by the base
        class — raise ``batch_size`` on a GPU, lower it if you hit memory limits.
    """

    provider_name = "huggingface"

    def __init__(
        self,
        model: str = "",
        *,
        device: str | None = None,
        trust_remote_code: bool = False,
        auto_prefix: bool = True,
        model_kwargs: dict[str, Any] | None = None,
        encode_kwargs: dict[str, Any] | None = None,
        **config: Any,
    ) -> None:
        super().__init__(model=model, **config)
        st = require(
            "sentence_transformers",
            extra="embeddings",
            feature="The HuggingFace embedding provider",
        )
        try:
            self._model = st.SentenceTransformer(
                self.model,
                device=device,
                trust_remote_code=trust_remote_code,
                **(model_kwargs or {}),
            )
        except Exception as exc:
            raise ProviderError(
                f"Could not load the embedding model {self.model!r}: {exc}",
                provider="huggingface",
                hint="Check the model id, your network connection, and disk space "
                "in ~/.cache/huggingface.",
                original=exc,
            ) from exc

        self.encode_kwargs = {"show_progress_bar": False, **(encode_kwargs or {})}
        if auto_prefix:
            self.query_prefix, self.document_prefix = _prefixes_for(self.model)
        self._dimensions = self._dimensions or int(self._model.get_sentence_embedding_dimension())

    @classmethod
    def default_model(cls) -> str:
        """Return ``"BAAI/bge-small-en-v1.5"`` — small, fast and accurate."""
        return "BAAI/bge-small-en-v1.5"

    def native(self) -> Any:
        """Return the underlying ``SentenceTransformer`` instance."""
        return self._model

    def dimension(self) -> int:
        """Return the model's embedding dimensionality."""
        return int(self._dimensions or self._model.get_sentence_embedding_dimension())

    async def aembed_texts(self, texts: list[str], *, kind: str = "document") -> list[list[float]]:
        """Encode a batch of texts on a worker thread.

        Args:
            texts: The texts to embed. Prefixes are already applied by the base
                class.
            kind: ``"document"`` or ``"query"``.

        Returns:
            One vector per input.

        Raises:
            ProviderError: When encoding fails.
        """

        def _encode() -> list[list[float]]:
            vectors = self._model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,  # the base class normalises
                **self.encode_kwargs,
            )
            return [[float(x) for x in row] for row in vectors]

        try:
            return await to_thread(_encode)
        except Exception as exc:
            raise ProviderError(
                f"Embedding failed for a batch of {len(texts)} texts: {exc}",
                provider="huggingface",
                original=exc,
            ) from exc


def _prefixes_for(model: str) -> tuple[str, str]:
    """Return the ``(query, document)`` instruction prefixes for a model family.

    Getting these wrong is a silent quality bug — the model still returns
    vectors, they are just measurably worse — so Windlass applies them by default.

    Args:
        model: The model id.

    Returns:
        A ``(query_prefix, document_prefix)`` pair; ``("", "")`` when the family
        is unknown or symmetric.

    Example:
        >>> _prefixes_for("intfloat/multilingual-e5-large")
        ('query: ', 'passage: ')
        >>> _prefixes_for("sentence-transformers/all-MiniLM-L6-v2")
        ('', '')
    """
    lowered = model.lower()
    for family, prefixes in _PREFIX_RULES.items():
        if family in lowered:
            return prefixes
    return ("", "")
