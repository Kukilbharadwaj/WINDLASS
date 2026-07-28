"""PII detection and redaction.

Once personal data is embedded into a vector index it is very hard to get back
out — you cannot un-ring that bell, and "delete the row" does not undo the
copies in your backups. Redacting at ingestion is the only reliable point of
control.

This preprocessor ships a dependency-free detector covering the categories that
actually turn up in enterprise documents (email, phone, SSN, credit card, IBAN,
IP, passport, API keys), with a Luhn check on card numbers to keep false
positives down. When ``presidio-analyzer`` is installed it is used instead, for
NER-based detection of names, locations and organisations.

Example:
    >>> from windlass.core.types import Document
    >>> p = PIIPreprocessor()
    >>> p.process([Document(content="Reach me at ada@example.com or 555-123-4567")])[0].content
    'Reach me at [EMAIL] or [PHONE]'
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, NamedTuple

from windlass.core.lazy import is_available
from windlass.core.registry import register
from windlass.core.types import Document
from windlass.interfaces.preprocessor import Preprocessor

__all__ = ["PII_PATTERNS", "PIIMatch", "PIIPreprocessor", "detect_pii", "redact_pii"]


class PIIMatch(NamedTuple):
    """One detected piece of personal data.

    Attributes:
        kind: Category, e.g. ``"email"``.
        value: The matched text.
        start: Start offset in the source string.
        end: Exclusive end offset.
    """

    kind: str
    value: str
    start: int
    end: int


#: Detection patterns, ordered so that more specific ones win. Each entry maps a
#: category name to its regex.
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "ssn": re.compile(r"\b(?!000|666|9\d\d)\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    "phone": re.compile(
        r"(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{3}[\s.-]\d{3,4}[\s.-]?\d{0,4}\b"
    ),
    "ipv4": re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
    "passport": re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"),
    "api_key": re.compile(
        r"\b(?:sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{36}|AKIA[0-9A-Z]{16}|"
        r"AIza[0-9A-Za-z_-]{35})\b"
    ),
}

#: Replacement token per category.
_PLACEHOLDERS = {kind: f"[{kind.upper()}]" for kind in PII_PATTERNS}


def _luhn(digits: str) -> bool:
    """Return whether ``digits`` passes the Luhn checksum.

    Credit-card-shaped numbers are common in ordinary text (order ids, phone
    strings). The Luhn check removes nearly all of those false positives.

    Args:
        digits: The digit string to validate.

    Returns:
        True when the checksum is valid.

    Example:
        >>> _luhn("4111111111111111")
        True
        >>> _luhn("1234567812345678")
        False
    """
    values = [int(c) for c in digits if c.isdigit()]
    if len(values) < 13:
        return False
    total = 0
    for position, value in enumerate(reversed(values)):
        if position % 2:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def detect_pii(text: str, kinds: Sequence[str] | None = None) -> list[PIIMatch]:
    """Find personal data in ``text``.

    Overlapping matches are resolved in favour of the earlier, longer one, so an
    email address is never also reported as a phone number.

    Args:
        text: The text to scan.
        kinds: Categories to look for. ``None`` scans every category in
            :data:`PII_PATTERNS`.

    Returns:
        Matches sorted by position.

    Example:
        >>> [m.kind for m in detect_pii("write to a@b.com")]
        ['email']
    """
    wanted = list(kinds) if kinds else list(PII_PATTERNS)
    found: list[PIIMatch] = []

    for kind in wanted:
        pattern = PII_PATTERNS.get(kind)
        if pattern is None:
            continue
        for match in pattern.finditer(text):
            value = match.group(0)
            if kind == "credit_card" and not _luhn(value):
                continue
            if kind == "phone" and sum(c.isdigit() for c in value) < 7:
                continue
            found.append(PIIMatch(kind, value, match.start(), match.end()))

    found.sort(key=lambda m: (m.start, -(m.end - m.start)))
    resolved: list[PIIMatch] = []
    cursor = -1
    for candidate in found:
        if candidate.start >= cursor:
            resolved.append(candidate)
            cursor = candidate.end
    return resolved


def redact_pii(
    text: str,
    kinds: Sequence[str] | None = None,
    *,
    placeholder: str | None = None,
) -> tuple[str, list[PIIMatch]]:
    """Replace personal data in ``text`` with placeholders.

    Args:
        text: The text to redact.
        kinds: Categories to redact. ``None`` redacts every category.
        placeholder: Fixed replacement for every match. ``None`` uses a
            per-category token such as ``[EMAIL]``, which preserves the semantic
            hint that *something* was there.

    Returns:
        A ``(redacted_text, matches)`` pair.

    Example:
        >>> redact_pii("call 555-123-4567")[0]
        'call [PHONE]'
    """
    matches = detect_pii(text, kinds)
    if not matches:
        return text, []
    out: list[str] = []
    cursor = 0
    for match in matches:
        out.append(text[cursor : match.start])
        out.append(placeholder or _PLACEHOLDERS.get(match.kind, "[REDACTED]"))
        cursor = match.end
    out.append(text[cursor:])
    return "".join(out), matches


@register.preprocessor(
    "pii",
    aliases=("privacy", "redact"),
    description="Detects and redacts personal data before it reaches the index.",
)
class PIIPreprocessor(Preprocessor):
    """Detects and optionally redacts personal data.

    Args:
        kinds: Categories to act on. ``None`` means every category in
            :data:`PII_PATTERNS`.
        action: ``"redact"`` replaces matches, ``"drop"`` discards any document
            containing PII, ``"tag"`` only records what it found.
        placeholder: Fixed replacement token, or ``None`` for per-category
            tokens.
        use_presidio: Use ``presidio-analyzer`` when installed, adding NER-based
            detection of names, locations and organisations.
        language: Language passed to Presidio.
        **config: Forwarded to :class:`~windlass.interfaces.preprocessor.Preprocessor`.

    Raises:
        ValueError: For an unknown ``action``.

    Note:
        Regex detection is precise for structured identifiers and blind to
        unstructured ones — it will not find "Ada Lovelace" as a person's name.
        Install ``windlass[pii]`` and set ``use_presidio=True`` when you need
        that, and treat any automated detector as a control, not a guarantee.

    Example:
        >>> from windlass.core.types import Document
        >>> p = PIIPreprocessor(action="tag")
        >>> p.process([Document(content="a@b.com")])[0].metadata["pii_kinds"]
        ['email']
    """

    provider_name = "pii"

    def __init__(
        self,
        *,
        kinds: Sequence[str] | None = None,
        action: str = "redact",
        placeholder: str | None = None,
        use_presidio: bool = False,
        language: str = "en",
        **config: Any,
    ) -> None:
        if action not in {"redact", "drop", "tag"}:
            raise ValueError("action must be 'redact', 'drop' or 'tag'")
        super().__init__(**config)
        self.kinds = list(kinds) if kinds else None
        self.action = action
        self.placeholder = placeholder
        self.language = language
        self.use_presidio = use_presidio and is_available("presidio_analyzer")
        if use_presidio and not self.use_presidio:
            self._log.warning(
                "presidio-analyzer is not installed; falling back to regex detection. "
                'Install it with: pip install "windlass[pii]"'
            )
        self._analyzer: Any = None

    async def aprocess_one(self, document: Document) -> list[Document]:
        """Scan a document and apply the configured action.

        Args:
            document: The document to scan.

        Returns:
            The processed document, or ``[]`` when ``action='drop'`` and PII was
            found.
        """
        if self.use_presidio:
            text, matches = self._presidio(document.content)
        else:
            text, matches = redact_pii(document.content, self.kinds, placeholder=self.placeholder)

        if not matches:
            return [document]

        kinds = sorted({m.kind for m in matches})
        if self.action == "drop":
            self._log.info(
                "Dropping %s: contains %s", document.source or document.id, ", ".join(kinds)
            )
            return []

        metadata = {
            **document.metadata,
            "pii_detected": True,
            "pii_kinds": kinds,
            "pii_count": len(matches),
        }
        content = text if self.action == "redact" else document.content
        return [document.model_copy(update={"content": content, "metadata": metadata})]

    def _presidio(self, text: str) -> tuple[str, list[PIIMatch]]:
        """Detect and redact using Presidio, falling back to regex on failure."""
        try:
            if self._analyzer is None:
                from presidio_analyzer import AnalyzerEngine

                self._analyzer = AnalyzerEngine()
            results = self._analyzer.analyze(text=text, language=self.language)
        except Exception as exc:
            self._log.warning("Presidio analysis failed, using regex detection: %s", exc)
            return redact_pii(text, self.kinds, placeholder=self.placeholder)

        matches = [
            PIIMatch(str(r.entity_type).lower(), text[r.start : r.end], r.start, r.end)
            for r in sorted(results, key=lambda r: r.start)
        ]
        if not matches:
            return text, []

        out: list[str] = []
        cursor = 0
        for match in matches:
            if match.start < cursor:
                continue
            out.append(text[cursor : match.start])
            out.append(self.placeholder or f"[{match.kind.upper()}]")
            cursor = match.end
        out.append(text[cursor:])
        return "".join(out), matches
