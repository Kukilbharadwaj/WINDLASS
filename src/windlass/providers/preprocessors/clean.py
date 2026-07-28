r"""Cleaning, language detection and metadata enrichment — all dependency-free.

These are the preprocessors almost every pipeline wants. ``clean`` alone
typically removes 5-15% of a PDF corpus's characters (headers, page numbers,
control codes) without losing meaning, which is 5-15% off your embedding bill.

Example:
    >>> from windlass.core.types import Document
    >>> doc = Document(content="Hello   world\n\n\n\nAgain")
    >>> CleanPreprocessor(min_length=0).process([doc])[0].content
    'Hello world\n\nAgain'
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from windlass.core.registry import register
from windlass.core.text import (
    count_tokens,
    detect_language,
    normalize_unicode,
    normalize_whitespace,
    strip_html,
)
from windlass.core.types import Document
from windlass.interfaces.preprocessor import Preprocessor

__all__ = ["CleanPreprocessor", "LanguagePreprocessor", "MetadataPreprocessor"]

_URL_RE = re.compile(r"https?://\S+")
_PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:page\s*)?[-–—]?\s*\d+\s*(?:of\s*\d+)?\s*[-–—]?\s*$", re.IGNORECASE
)
_BULLET_RE = re.compile(r"^[\s]*[•▪◦‣·]\s*", re.MULTILINE)


@register.preprocessor(
    "clean",
    aliases=("cleaner", "normalize"),
    description="Normalises whitespace and Unicode, strips boilerplate (no dependencies).",
)
class CleanPreprocessor(Preprocessor):
    """Normalises text and removes common noise.

    Args:
        unicode: Apply NFKC normalisation and drop control characters. Fixes the
            ligatures and non-breaking spaces that PDF extraction produces.
        whitespace: Collapse runs of spaces and blank lines.
        strip_html_tags: Remove HTML tags left over from a mixed-format source.
        remove_urls: Delete bare URLs, which embed poorly and add no meaning.
        remove_page_numbers: Delete lines that are nothing but a page number.
        normalize_bullets: Rewrite Unicode bullets as ``- ``.
        min_length: Drop documents shorter than this after cleaning. This is how
            blank pages and extraction artefacts get filtered out.
        max_length: Truncate documents longer than this. ``None`` means no limit.
        **config: Forwarded to :class:`~windlass.interfaces.preprocessor.Preprocessor`.

    Example:
        >>> from windlass.core.types import Document
        >>> p = CleanPreprocessor(remove_urls=True, min_length=0)
        >>> p.process([Document(content="see https://x.com now")])[0].content
        'see now'
    """

    provider_name = "clean"

    def __init__(
        self,
        *,
        unicode: bool = True,
        whitespace: bool = True,
        strip_html_tags: bool = False,
        remove_urls: bool = False,
        remove_page_numbers: bool = True,
        normalize_bullets: bool = True,
        min_length: int = 20,
        max_length: int | None = None,
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self.unicode = unicode
        self.whitespace = whitespace
        self.strip_html_tags = strip_html_tags
        self.remove_urls = remove_urls
        self.remove_page_numbers = remove_page_numbers
        self.normalize_bullets = normalize_bullets
        self.min_length = min_length
        self.max_length = max_length

    async def aprocess_one(self, document: Document) -> list[Document]:
        """Clean one document.

        Args:
            document: The document to clean.

        Returns:
            A single-element list, or ``[]`` when the cleaned text is shorter
            than :attr:`min_length`.
        """
        text = document.content
        if self.strip_html_tags and "<" in text:
            text = strip_html(text)
        if self.unicode:
            text = normalize_unicode(text)
        if self.remove_urls:
            text = _URL_RE.sub(" ", text)
        if self.remove_page_numbers:
            text = "\n".join(line for line in text.split("\n") if not _PAGE_NUMBER_RE.match(line))
        if self.normalize_bullets:
            text = _BULLET_RE.sub("- ", text)
        if self.whitespace:
            text = normalize_whitespace(text)
        if self.max_length and len(text) > self.max_length:
            text = text[: self.max_length]

        if len(text) < self.min_length:
            self._log.debug(
                "Dropping %s: %d characters after cleaning (minimum %d).",
                document.source or document.id,
                len(text),
                self.min_length,
            )
            return []

        return [
            document.model_copy(
                update={
                    "content": text,
                    "metadata": {
                        **document.metadata,
                        "cleaned": True,
                        "original_length": len(document.content),
                    },
                }
            )
        ]


@register.preprocessor(
    "language",
    aliases=("lang", "language-filter"),
    description="Detects language and optionally filters documents by it.",
)
class LanguagePreprocessor(Preprocessor):
    """Tags each document with a detected language, and can filter on it.

    Mixed-language corpora quietly degrade retrieval: a monolingual embedding
    model maps text it does not understand to roughly meaningless vectors. Tag
    first, then either filter or route to a multilingual model.

    Args:
        allowed: Language codes to keep. ``None`` tags without filtering.
        default: Language assumed when detection is inconclusive.
        field: Metadata key to write the detected code to.
        **config: Forwarded to :class:`~windlass.interfaces.preprocessor.Preprocessor`.

    Note:
        Uses ``langdetect`` when installed, and a stop-word plus script
        heuristic otherwise. The heuristic is good enough for tagging and
        routing, not for anything adversarial.

    Example:
        >>> from windlass.core.types import Document
        >>> p = LanguagePreprocessor()
        >>> p.process([Document(content="the quick brown fox is here")])[0].metadata["language"]
        'en'
    """

    provider_name = "language"

    def __init__(
        self,
        *,
        allowed: Sequence[str] | None = None,
        default: str = "en",
        field: str = "language",
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self.allowed = {a.lower() for a in allowed} if allowed else None
        self.default = default
        self.field = field

    async def aprocess_one(self, document: Document) -> list[Document]:
        """Detect the language and tag (or drop) the document.

        Args:
            document: The document to inspect.

        Returns:
            The tagged document, or ``[]`` when its language is not allowed.
        """
        language = detect_language(document.content, default=self.default)
        if self.allowed is not None and language not in self.allowed:
            self._log.debug(
                "Dropping %s: language %r not in %s",
                document.source or document.id,
                language,
                sorted(self.allowed),
            )
            return []
        return [
            document.model_copy(update={"metadata": {**document.metadata, self.field: language}})
        ]


@register.preprocessor(
    "metadata",
    aliases=("enrich", "stats"),
    description="Adds computed statistics and keywords to document metadata.",
)
class MetadataPreprocessor(Preprocessor):
    """Enriches metadata with cheap, useful signals.

    Everything here is computed locally with no model call: character and word
    counts, an estimated token count, a reading-time estimate, and the top
    keywords by frequency. Those fields make retrieval filters (``token_count <
    2000``) and result display far more useful.

    Args:
        extra: Static key/value pairs merged into every document — a tenant id,
            a corpus name, an ingestion timestamp.
        keywords: How many top keywords to extract. ``0`` disables it.
        estimate_tokens: Compute a token count. Costs a tokenisation pass.
        **config: Forwarded to :class:`~windlass.interfaces.preprocessor.Preprocessor`.

    Example:
        >>> from windlass.core.types import Document
        >>> meta = MetadataPreprocessor().process(
        ...     [Document(content="alpha beta alpha gamma")]
        ... )[0].metadata
        >>> meta["word_count"]
        4
    """

    provider_name = "metadata"

    def __init__(
        self,
        *,
        extra: dict[str, Any] | None = None,
        keywords: int = 8,
        estimate_tokens: bool = True,
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self.extra = dict(extra or {})
        self.keywords = keywords
        self.estimate_tokens = estimate_tokens

    async def aprocess_one(self, document: Document) -> list[Document]:
        """Compute and attach metadata.

        Args:
            document: The document to enrich.

        Returns:
            A single-element list with the enriched document.
        """
        text = document.content
        words = text.split()
        metadata: dict[str, Any] = {
            **document.metadata,
            **self.extra,
            "char_count": len(text),
            "word_count": len(words),
            "reading_time_minutes": round(len(words) / 220, 1),
        }
        if self.estimate_tokens:
            metadata["token_count"] = count_tokens(text)
        if self.keywords:
            metadata["keywords"] = _top_keywords(text, self.keywords)
        return [document.model_copy(update={"metadata": metadata})]


def _top_keywords(text: str, limit: int) -> list[str]:
    """Return the most frequent meaningful words in ``text``.

    A frequency count with stop-words removed. Not TF-IDF — that needs corpus
    statistics this preprocessor deliberately does not hold — but enough to give
    a document a usable set of tags.

    Args:
        text: The text to analyse.
        limit: How many keywords to return.

    Returns:
        Keywords ordered by descending frequency.

    Example:
        >>> _top_keywords("retrieval retrieval augmented generation the", 2)
        ['retrieval', 'augmented']
    """
    from windlass.core.text import tokenize_words
    from windlass.providers.retrievers.bm25 import STOPWORDS

    tokens = [
        token
        for token in tokenize_words(text)
        if len(token) > 2 and token not in STOPWORDS and not token.isdigit()
    ]
    return [word for word, _ in Counter(tokens).most_common(limit)]
