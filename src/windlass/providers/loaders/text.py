r"""Plain-text, Markdown, JSON and CSV loaders — all dependency-free.

These four cover a surprising share of real corpora and they need nothing beyond
the standard library, so ``pip install windlass`` alone can ingest a folder of
notes, an exported dataset or a JSONL dump.

Example:
    >>> import tempfile, pathlib
    >>> d = pathlib.Path(tempfile.mkdtemp())
    >>> _ = (d / "note.md").write_text("# Title\n\nBody text.")
    >>> docs = MarkdownLoader().load(d)
    >>> docs[0].metadata["title"]
    'Title'
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Any

from windlass.core.exceptions import IngestionError
from windlass.core.registry import register
from windlass.core.types import Document
from windlass.interfaces.loader import Loader, SourceLike

__all__ = ["CSVLoader", "JSONLoader", "MarkdownLoader", "TextLoader"]

_TITLE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@register.loader(
    "text",
    aliases=("txt", "plain"),
    description="Plain-text files (.txt, .log, .rst and friends).",
)
class TextLoader(Loader):
    """Loads plain-text files.

    Args:
        encoding: Text encoding. Falls back to a lenient decode when the file
            turns out not to be valid in this encoding, so one mislabelled file
            does not abort a corpus.
        **config: Forwarded to :class:`~windlass.interfaces.loader.Loader`.

    Example:
        >>> import tempfile, pathlib
        >>> p = pathlib.Path(tempfile.mkdtemp()) / "a.txt"
        >>> _ = p.write_text("hello")
        >>> TextLoader().load(p)[0].content
        'hello'
    """

    provider_name = "text"
    extensions = (".txt", ".text", ".log", ".rst", ".rtf", ".tex", ".ini", ".cfg", ".env")

    async def aload_source(self, source: SourceLike) -> list[Document]:
        """Read one text file.

        Args:
            source: A path or a bytes payload.

        Returns:
            A single-element list holding the file's text.

        Raises:
            IngestionError: When the file cannot be read.
        """
        raw = self._read_bytes(source)
        text = _decode(raw, self.encoding)
        metadata = self._base_metadata(source) if not isinstance(source, bytes) else {}
        return [
            Document(
                content=text,
                metadata={**metadata, "lines": text.count("\n") + 1},
                source=str(source) if not isinstance(source, bytes) else None,
                mimetype="text/plain",
            )
        ]


@register.loader(
    "markdown",
    aliases=("md",),
    description="Markdown files, with front-matter and title extraction.",
)
class MarkdownLoader(Loader):
    r"""Loads Markdown files and extracts their structure.

    Pulls out the YAML front-matter (parsed as key/value pairs without needing
    PyYAML), the first ``#`` heading as a title, and the full heading list — all
    of which become metadata you can filter and cite on.

    Args:
        strip_frontmatter: Remove the front-matter block from the content.
        **config: Forwarded to :class:`~windlass.interfaces.loader.Loader`.

    Example:
        >>> import tempfile, pathlib
        >>> p = pathlib.Path(tempfile.mkdtemp()) / "d.md"
        >>> _ = p.write_text("---\ntag: guide\n---\n# Setup\n\nRun it.")
        >>> doc = MarkdownLoader().load(p)[0]
        >>> doc.metadata["tag"], doc.metadata["title"]
        ('guide', 'Setup')
    """

    provider_name = "markdown"
    extensions = (".md", ".markdown", ".mdx", ".mdown")

    def __init__(self, *, strip_frontmatter: bool = True, **config: Any) -> None:
        super().__init__(**config)
        self.strip_frontmatter = strip_frontmatter

    async def aload_source(self, source: SourceLike) -> list[Document]:
        """Read one Markdown file.

        Args:
            source: A path or a bytes payload.

        Returns:
            A single-element list.
        """
        raw = self._read_bytes(source)
        text = _decode(raw, self.encoding)
        metadata: dict[str, Any] = (
            self._base_metadata(source) if not isinstance(source, bytes) else {}
        )

        match = _FRONTMATTER_RE.match(text)
        if match:
            metadata.update(_parse_frontmatter(match.group(1)))
            if self.strip_frontmatter:
                text = text[match.end() :]

        title = _TITLE_RE.search(text)
        if title:
            metadata.setdefault("title", title.group(1).strip())
        headings = [h.strip() for h in re.findall(r"^#{1,6}\s+(.+)$", text, re.MULTILINE)]
        if headings:
            metadata["headings"] = headings[:50]

        return [
            Document(
                content=text.strip(),
                metadata=metadata,
                source=str(source) if not isinstance(source, bytes) else None,
                mimetype="text/markdown",
            )
        ]


@register.loader(
    "json",
    aliases=("jsonl", "ndjson"),
    description="JSON and JSON Lines files, one document per record.",
)
class JSONLoader(Loader):
    """Loads JSON and JSON Lines files.

    Args:
        content_key: Field to use as document content. When omitted the whole
            record is pretty-printed, which is right for heterogeneous data.
        metadata_keys: Fields copied into metadata. ``None`` copies every scalar
            field, which is usually what you want for filtering.
        jq_path: Dotted path into the document to find the array of records,
            e.g. ``"data.items"``. Supports ``[]`` for list traversal.
        **config: Forwarded to :class:`~windlass.interfaces.loader.Loader`.

    Example:
        >>> import tempfile, pathlib, json
        >>> p = pathlib.Path(tempfile.mkdtemp()) / "d.json"
        >>> _ = p.write_text(json.dumps([{"text": "a", "id": 1}, {"text": "b", "id": 2}]))
        >>> docs = JSONLoader(content_key="text").load(p)
        >>> [d.content for d in docs]
        ['a', 'b']
    """

    provider_name = "json"
    extensions = (".json", ".jsonl", ".ndjson")

    def __init__(
        self,
        *,
        content_key: str | None = None,
        metadata_keys: list[str] | None = None,
        jq_path: str | None = None,
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self.content_key = content_key
        self.metadata_keys = metadata_keys
        self.jq_path = jq_path

    async def aload_source(self, source: SourceLike) -> list[Document]:
        """Read a JSON or JSONL file.

        Args:
            source: A path or a bytes payload.

        Returns:
            One document per record.

        Raises:
            IngestionError: When the file is not valid JSON.
        """
        raw = self._read_bytes(source)
        text = _decode(raw, self.encoding)
        path = str(source) if not isinstance(source, bytes) else None
        base = self._base_metadata(source) if not isinstance(source, bytes) else {}

        records = self._parse(text, path)
        documents: list[Document] = []
        for index, record in enumerate(records):
            content, metadata = self._render(record)
            documents.append(
                Document(
                    content=content,
                    metadata={**base, **metadata, "record_index": index},
                    source=path,
                    mimetype="application/json",
                )
            )
        return documents

    def _parse(self, text: str, path: str | None) -> list[Any]:
        """Decode the payload into a list of records."""
        stripped = text.strip()
        if not stripped:
            return []

        # JSON Lines: one object per line.
        if (path or "").endswith((".jsonl", ".ndjson")) or (
            "\n" in stripped and not stripped.startswith(("[", "{"))
        ):
            records = []
            for number, line in enumerate(stripped.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise IngestionError(
                        f"Invalid JSON on line {number} of {path or 'input'}: {exc}"
                    ) from exc
            return records

        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise IngestionError(f"Invalid JSON in {path or 'input'}: {exc}") from exc

        if self.jq_path:
            data = _dig(data, self.jq_path)
        return data if isinstance(data, list) else [data]

    def _render(self, record: Any) -> tuple[str, dict[str, Any]]:
        """Turn one record into ``(content, metadata)``."""
        if not isinstance(record, dict):
            return (str(record), {})

        if self.content_key:
            content = str(record.get(self.content_key, ""))
        else:
            content = json.dumps(record, indent=2, ensure_ascii=False, default=str)

        if self.metadata_keys is not None:
            metadata = {k: record[k] for k in self.metadata_keys if k in record}
        else:
            metadata = {
                k: v
                for k, v in record.items()
                if k != self.content_key and isinstance(v, str | int | float | bool)
            }
        return content, metadata


@register.loader(
    "csv",
    aliases=("tsv",),
    description="CSV and TSV files, one document per row or one per file.",
)
class CSVLoader(Loader):
    r"""Loads delimited data files.

    Args:
        mode: ``"row"`` makes one document per row (right for records — support
            tickets, product listings); ``"table"`` makes one document for the
            whole file (right for small reference tables the model should see
            whole).
        content_columns: Columns included in the content. ``None`` uses all.
        metadata_columns: Columns copied into metadata.
        delimiter: Field separator. Inferred from the extension when omitted.
        max_rows: Stop after this many rows. ``None`` reads everything.
        **config: Forwarded to :class:`~windlass.interfaces.loader.Loader`.

    Example:
        >>> import tempfile, pathlib
        >>> p = pathlib.Path(tempfile.mkdtemp()) / "d.csv"
        >>> _ = p.write_text("name,role\nada,engineer\ngrace,admiral\n")
        >>> docs = CSVLoader().load(p)
        >>> len(docs), "ada" in docs[0].content
        (2, True)
    """

    provider_name = "csv"
    extensions = (".csv", ".tsv", ".psv")

    def __init__(
        self,
        *,
        mode: str = "row",
        content_columns: list[str] | None = None,
        metadata_columns: list[str] | None = None,
        delimiter: str | None = None,
        max_rows: int | None = None,
        **config: Any,
    ) -> None:
        if mode not in {"row", "table"}:
            raise ValueError("mode must be 'row' or 'table'")
        super().__init__(**config)
        self.mode = mode
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns
        self.delimiter = delimiter
        self.max_rows = max_rows

    async def aload_source(self, source: SourceLike) -> list[Document]:
        """Read a delimited file.

        Args:
            source: A path or a bytes payload.

        Returns:
            One document per row, or one for the whole table.

        Raises:
            IngestionError: When the file cannot be parsed.
        """
        raw = self._read_bytes(source)
        text = _decode(raw, self.encoding)
        path = str(source) if not isinstance(source, bytes) else None
        base = self._base_metadata(source) if not isinstance(source, bytes) else {}
        delimiter = self.delimiter or _sniff_delimiter(path, text)

        try:
            reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
            rows = list(reader)[: self.max_rows] if self.max_rows else list(reader)
        except csv.Error as exc:
            raise IngestionError(f"Could not parse {path or 'input'} as CSV: {exc}") from exc

        if not rows:
            return []

        if self.mode == "table":
            # Render as a Markdown table: models read the header/column
            # relationship far more reliably from this than from raw CSV.
            headers = [h for h in rows[0] if h is not None]
            lines = [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join("---" for _ in headers) + " |",
            ]
            lines += [
                "| " + " | ".join(str(row.get(h, "") or "") for h in headers) + " |" for row in rows
            ]
            return [
                Document(
                    content="\n".join(lines),
                    metadata={**base, "rows": len(rows), "columns": headers},
                    source=path,
                    mimetype="text/csv",
                )
            ]

        documents: list[Document] = []
        for index, row in enumerate(rows):
            columns = self.content_columns or list(row.keys())
            content = "\n".join(f"{k}: {row.get(k, '')}" for k in columns if k)
            metadata = (
                {k: row[k] for k in self.metadata_columns if k in row}
                if self.metadata_columns
                else {}
            )
            documents.append(
                Document(
                    content=content,
                    metadata={**base, **metadata, "row_index": index},
                    source=path,
                    mimetype="text/csv",
                )
            )
        return documents


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _decode(raw: bytes, encoding: str) -> str:
    """Decode bytes, degrading gracefully rather than raising.

    A single file saved in the wrong encoding should not abort the ingestion of
    a 10,000-document corpus, so this tries the declared encoding, then UTF-8,
    then falls back to a replacing decode.

    Args:
        raw: The bytes to decode.
        encoding: The preferred encoding.

    Returns:
        The decoded text.
    """
    for candidate in (encoding, "utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(candidate)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode(encoding, errors="replace")


def _parse_frontmatter(block: str) -> dict[str, Any]:
    r"""Parse a simple ``key: value`` YAML front-matter block.

    Deliberately minimal — it handles the flat scalar and inline-list cases that
    cover the overwhelming majority of real front-matter, without requiring
    PyYAML in the core install.

    Args:
        block: The text between the ``---`` fences.

    Returns:
        The parsed key/value pairs.

    Example:
        >>> _parse_frontmatter("title: Guide\ntags: [a, b]\ndraft: true")
        {'title': 'Guide', 'tags': ['a', 'b'], 'draft': True}
    """
    out: dict[str, Any] = {}
    for line in block.splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip().strip("\"'")
        if not key:
            continue
        if value.startswith("[") and value.endswith("]"):
            out[key] = [v.strip().strip("\"'") for v in value[1:-1].split(",") if v.strip()]
        elif value.lower() in {"true", "false"}:
            out[key] = value.lower() == "true"
        elif value.lstrip("-").isdigit():
            out[key] = int(value)
        else:
            out[key] = value
    return out


def _dig(data: Any, path: str) -> Any:
    """Follow a dotted path into nested JSON.

    Args:
        data: The decoded JSON.
        path: A dotted path; ``[]`` maps the remainder over a list.

    Returns:
        Whatever the path points at.

    Raises:
        IngestionError: When the path does not exist.

    Example:
        >>> _dig({"data": {"items": [1, 2]}}, "data.items")
        [1, 2]
    """
    current = data
    for part in path.split("."):
        if not part:
            continue
        if part == "[]":
            if not isinstance(current, list):
                raise IngestionError(
                    f"Path segment '[]' expected a list, got {type(current).__name__}."
                )
            continue
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise IngestionError(
                f"JSON path {path!r} not found (failed at {part!r}).",
                hint="Check the structure of the file, or drop jq_path to load it whole.",
            )
    return current


def _sniff_delimiter(path: str | None, text: str) -> str:
    """Guess the field separator from the extension, then the content."""
    if path:
        suffix = Path(path).suffix.lower()
        if suffix == ".tsv":
            return "\t"
        if suffix == ".psv":
            return "|"
    try:
        return csv.Sniffer().sniff(text[:4096], delimiters=",;\t|").delimiter
    except csv.Error:
        return ","
