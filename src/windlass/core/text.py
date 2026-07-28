"""Text utilities shared by loaders, chunkers, retrievers and prompts.

Everything here is dependency-free and deterministic. Token counting uses
``tiktoken`` when it is installed and a calibrated heuristic otherwise, so a
chunker configured in tokens behaves sensibly on a core-only install.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from collections.abc import Iterable, Iterator
from typing import Any, Literal

from windlass.core.lazy import is_available

__all__ = [
    "count_tokens",
    "detect_language",
    "normalize_unicode",
    "normalize_whitespace",
    "sliding_window",
    "split_paragraphs",
    "split_sentences",
    "strip_html",
    "text_hash",
    "tokenize_words",
    "truncate_tokens",
]

#: Rough characters-per-token ratio for English prose, used when tiktoken is
#: unavailable. Empirically within ~10% for cl100k_base on natural language.
_CHARS_PER_TOKEN = 4.0

_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
#: Candidate sentence break: end punctuation, optional closing quote/bracket,
#: whitespace, then something that can start a sentence. Abbreviations are
#: rejected afterwards in :func:`split_sentences` — Python's ``re`` only accepts
#: fixed-width lookbehind, so the exclusion cannot live in the pattern.
_SENTENCE_BREAK_RE = re.compile(r"([.!?])([\"')\]]*)(\s+)(?=[A-Z0-9\"'(\[])")

#: Words that end in a period without ending a sentence.
_ABBREVIATIONS: frozenset[str] = frozenset(
    [
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "st",
        "jr",
        "sr",
        "rev",
        "hon",
        "gen",
        "col",
        "sgt",
        "lt",
        "capt",
        "etc",
        "vs",
        "approx",
        "est",
        "dept",
        "univ",
        "inc",
        "ltd",
        "llc",
        "co",
        "corp",
        "no",
        "vol",
        "fig",
        "eq",
        "ref",
        "ch",
        "sec",
        "art",
        "pp",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sept",
        "sep",
        "oct",
        "nov",
        "dec",
        "e.g",
        "i.e",
        "a.m",
        "p.m",
        "u.s",
        "u.k",
    ]
)
_PARAGRAPH_RE = re.compile(r"\n\s*\n+")
_WORD_RE = re.compile(r"\b\w[\w'-]*\b", re.UNICODE)
_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
#: The document head holds the title and metadata, not body text. Leaving it in
#: means every page's title is prepended to its first chunk — and a bare
#: ``<title>`` outside a head does the same, so both are matched.
_HEAD_RE = re.compile(
    r"<head\b[^>]*>.*?</head>|<title\b[^>]*>.*?</title>", re.IGNORECASE | re.DOTALL
)


@functools.lru_cache(maxsize=8)
def _encoder(model: str) -> Any:
    """Return a cached tiktoken encoder, or ``None`` when unavailable."""
    if not is_available("tiktoken"):
        return None
    try:
        import tiktoken

        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Estimate how many tokens ``text`` occupies.

    Exact when ``tiktoken`` is installed; otherwise a character-ratio estimate
    that is close enough for chunk sizing and budget checks.

    Args:
        text: The text to measure.
        model: Model name used to pick the tokeniser.

    Returns:
        Token count. Always at least 1 for non-empty text.

    Example:
        >>> count_tokens("") == 0
        True
        >>> count_tokens("hello world") >= 1
        True
    """
    if not text:
        return 0
    enc = _encoder(model)
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def truncate_tokens(text: str, max_tokens: int, model: str = "gpt-4o", suffix: str = "") -> str:
    """Trim ``text`` so it fits within ``max_tokens``.

    Args:
        text: The text to trim.
        max_tokens: Ceiling, inclusive.
        model: Model name used to pick the tokeniser.
        suffix: Appended when truncation happened, e.g. ``"…"``. Its own tokens
            are accounted for.

    Returns:
        The original text when it already fits, otherwise a truncated copy.

    Raises:
        ValueError: If ``max_tokens`` is not positive.

    Example:
        >>> truncate_tokens("short", 100)
        'short'
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if count_tokens(text, model) <= max_tokens:
        return text

    budget = max(1, max_tokens - count_tokens(suffix, model)) if suffix else max_tokens
    enc = _encoder(model)
    if enc is not None:
        return enc.decode(enc.encode(text, disallowed_special=())[:budget]) + suffix
    return text[: int(budget * _CHARS_PER_TOKEN)] + suffix


def normalize_whitespace(text: str, *, collapse_newlines: bool = True) -> str:
    r"""Collapse runs of spaces and (optionally) blank lines.

    Preserves paragraph structure — two newlines survive, three or more collapse
    to two — because chunkers rely on paragraph boundaries.

    Args:
        text: Input text.
        collapse_newlines: Also collapse runs of 3+ newlines to 2.

    Returns:
        Cleaned text with trailing whitespace stripped from each line.

    Example:
        >>> normalize_whitespace("a  \t b\n\n\n\nc")
        'a b\n\nc'
    """
    lines = [_WHITESPACE_RE.sub(" ", line).rstrip() for line in text.split("\n")]
    out = "\n".join(lines)
    if collapse_newlines:
        out = _MULTI_NEWLINE_RE.sub("\n\n", out)
    return out.strip()


def normalize_unicode(
    text: str,
    *,
    form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFKC",
    strip_control: bool = True,
) -> str:
    """Apply Unicode normalisation and drop control characters.

    PDF extraction routinely emits ligatures, non-breaking spaces and stray
    control codes; leaving them in produces chunk boundaries and embeddings that
    do not match what a user would type.

    Args:
        text: Input text.
        form: Normalisation form (``NFC``, ``NFD``, ``NFKC``, ``NFKD``).
        strip_control: Remove C0/C1 control characters except tab and newline.

    Returns:
        Normalised text.

    Example:
        >>> normalize_unicode("ﬁle")
        'file'
    """
    out = unicodedata.normalize(form, text)
    if strip_control:
        out = "".join(ch for ch in out if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    return out


def strip_html(html: str) -> str:
    """Remove tags from an HTML fragment, keeping the readable text.

    A dependency-free fallback. The ``html`` loader prefers BeautifulSoup, which
    handles malformed markup far better; this exists so that HTML embedded in
    other formats can still be cleaned on a core-only install.

    Args:
        html: HTML source.

    Returns:
        Plain text with entities resolved and whitespace normalised.

    Example:
        >>> strip_html("<p>Hello <b>world</b></p>")
        'Hello world'
    """
    import html as html_module

    text = _HEAD_RE.sub(" ", html)
    text = _SCRIPT_RE.sub(" ", text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|li|tr|h[1-6])>", "\n", text, flags=re.IGNORECASE)
    text = _TAG_RE.sub(" ", text)
    return normalize_whitespace(html_module.unescape(text))


def split_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Tuned to avoid the classic false positives: single-letter initials
    (``J. Smith``), common abbreviations (``Dr.``, ``etc.``, ``Inc.``) and
    decimals (``$1.50``). Good enough for semantic chunking without dragging in
    an NLP toolkit.

    Args:
        text: Input text.

    Returns:
        Non-empty, stripped sentences in reading order.

    Example:
        >>> split_sentences("Dr. Smith paid $1.50. Then he left.")
        ['Dr. Smith paid $1.50.', 'Then he left.']
        >>> split_sentences("A. B. Jones arrived. He waited.")
        ['A. B. Jones arrived.', 'He waited.']
    """
    if not text.strip():
        return []

    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_BREAK_RE.finditer(text):
        if match.group(1) == "." and _is_abbreviation(text, match.start(1)):
            continue
        end = match.end(2)
        piece = text[start:end].strip()
        if piece:
            sentences.append(piece)
        start = match.end(3)

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _is_abbreviation(text: str, period_index: int) -> bool:
    """Return whether the period at ``period_index`` ends an abbreviation.

    Args:
        text: The full text.
        period_index: Offset of the ``.`` being considered as a break.

    Returns:
        True when the preceding token is a single letter or a known
        abbreviation, meaning the period does not end a sentence.
    """
    cursor = period_index
    while cursor > 0 and (text[cursor - 1].isalnum() or text[cursor - 1] == "."):
        cursor -= 1
    token = text[cursor:period_index].lower().strip(".")
    if not token:
        return False
    if len(token) == 1 and token.isalpha():
        return True
    return token in _ABBREVIATIONS


def split_paragraphs(text: str) -> list[str]:
    r"""Split text on blank lines.

    Args:
        text: Input text.

    Returns:
        Non-empty, stripped paragraphs.

    Example:
        >>> split_paragraphs("one\n\ntwo")
        ['one', 'two']
    """
    return [p.strip() for p in _PARAGRAPH_RE.split(text) if p.strip()]


def tokenize_words(text: str, *, lowercase: bool = True) -> list[str]:
    """Split text into word tokens for lexical search.

    Used by the BM25 retriever. Unicode-aware, keeps intra-word apostrophes and
    hyphens so ``state-of-the-art`` and ``don't`` survive intact.

    Args:
        text: Input text.
        lowercase: Case-fold the output.

    Returns:
        Word tokens in document order.

    Example:
        >>> tokenize_words("State-of-the-art RAG, don't you think?")
        ['state-of-the-art', 'rag', "don't", 'you', 'think']
    """
    tokens = _WORD_RE.findall(text)
    return [t.lower() for t in tokens] if lowercase else tokens


def sliding_window(items: list[Any], size: int, overlap: int = 0) -> Iterator[list[Any]]:
    """Yield overlapping windows over a list.

    Args:
        items: The sequence to window.
        size: Window length.
        overlap: How many items each window shares with the previous one.

    Yields:
        Windows of at most ``size`` items.

    Raises:
        ValueError: If ``size`` is not positive or ``overlap >= size``.

    Example:
        >>> list(sliding_window([1, 2, 3, 4], size=2, overlap=1))
        [[1, 2], [2, 3], [3, 4]]
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    step = size - overlap
    for start in range(0, max(1, len(items)), step):
        window = items[start : start + size]
        if not window:
            return
        yield window
        if start + size >= len(items):
            return


#: Character-frequency signatures for the cheap language detector.
_LANG_HINTS: dict[str, set[str]] = {
    "de": {"der", "die", "und", "ist", "nicht", "ein", "mit", "auch"},
    "fr": {"le", "la", "les", "est", "une", "avec", "pour", "dans"},
    "es": {"el", "la", "los", "una", "con", "para", "que", "por"},
    "it": {"il", "la", "che", "per", "con", "una", "sono", "del"},
    "pt": {"o", "de", "que", "para", "com", "uma", "nao", "por"},
    "nl": {"de", "het", "een", "van", "niet", "en", "met", "voor"},
    "en": {"the", "and", "is", "of", "to", "in", "that", "for"},
}


def detect_language(text: str, *, default: str = "en") -> str:
    """Guess the ISO 639-1 language code of ``text``.

    Prefers ``langdetect`` when installed. The fallback uses stop-word overlap
    plus script detection (CJK, Cyrillic, Arabic, Devanagari), which is accurate
    enough to route documents or tag metadata but is not a replacement for a
    real classifier.

    Args:
        text: Input text; the first 2000 characters are inspected.
        default: Returned when detection is inconclusive.

    Returns:
        A two-letter language code.

    Example:
        >>> detect_language("the quick brown fox is in the garden")
        'en'
        >>> detect_language("")
        'en'
    """
    sample = text[:2000].strip()
    if not sample:
        return default

    if is_available("langdetect"):
        try:
            from langdetect import DetectorFactory, detect

            DetectorFactory.seed = 0
            return str(detect(sample))
        except Exception:
            pass

    for ch in sample:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF:
            return "zh"
        if 0x3040 <= code <= 0x30FF:
            return "ja"
        if 0xAC00 <= code <= 0xD7AF:
            return "ko"
        if 0x0400 <= code <= 0x04FF:
            return "ru"
        if 0x0600 <= code <= 0x06FF:
            return "ar"
        if 0x0900 <= code <= 0x097F:
            return "hi"

    words = set(tokenize_words(sample)[:200])
    if not words:
        return default
    scores = {lang: len(words & hints) for lang, hints in _LANG_HINTS.items()}
    best = max(scores, key=scores.__getitem__)
    return best if scores[best] >= 2 else default


def text_hash(text: str, *, normalize: bool = True) -> str:
    """Return a stable hash of ``text`` for deduplication.

    Args:
        text: Input text.
        normalize: Case-fold and collapse whitespace first, so cosmetic
            differences do not defeat deduplication.

    Returns:
        A 32-character hex digest.

    Example:
        >>> text_hash("Hello  World") == text_hash("hello world")
        True
    """
    import hashlib

    payload = " ".join(text.lower().split()) if normalize else text
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def join_nonempty(parts: Iterable[str], separator: str = "\n\n") -> str:
    """Join the truthy, stripped members of ``parts``.

    Args:
        parts: Candidate strings.
        separator: Separator inserted between kept parts.

    Returns:
        The joined string.

    Example:
        >>> join_nonempty(["a", "", "  ", "b"], "-")
        'a-b'
    """
    return separator.join(p.strip() for p in parts if p and p.strip())
