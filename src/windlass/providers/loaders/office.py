"""PDF, Word, PowerPoint and Excel loaders.

Install with::

    pip install "windlass[loaders]"

Every loader here runs its blocking parser on a worker thread, so ingesting a
folder of 500 PDFs uses the whole machine instead of one core.

Example:
    >>> from windlass import Windlass                     # doctest: +SKIP
    >>> docs = Windlass.loader("pdf").load("./reports")  # doctest: +SKIP
"""

from __future__ import annotations

import io
from typing import Any

from windlass.core.concurrency import to_thread
from windlass.core.exceptions import IngestionError
from windlass.core.lazy import require
from windlass.core.registry import register
from windlass.core.text import normalize_unicode, normalize_whitespace
from windlass.core.types import Document
from windlass.interfaces.loader import Loader, SourceLike

__all__ = ["DocxLoader", "PDFLoader", "PptxLoader", "XlsxLoader"]


@register.loader(
    "pdf",
    description="PDF documents, one document per page or per file.",
)
class PDFLoader(Loader):
    """Extracts text from PDF files using ``pypdf``.

    Args:
        per_page: Emit one document per page (recommended — page numbers become
            citations) rather than one per file.
        extract_metadata: Copy the PDF's own metadata (title, author, ...) onto
            every document.
        password: Password for encrypted PDFs.
        min_page_chars: Skip pages with less text than this. Filters out scan
            artefacts and blank separator pages.
        **config: Forwarded to :class:`~windlass.interfaces.loader.Loader`.

    Raises:
        MissingDependencyError: When ``pypdf`` is not installed.
        IngestionError: When the PDF is corrupt or encrypted without a password.

    Note:
        ``pypdf`` extracts *embedded* text. A scanned PDF contains images, not
        text, and will come back empty — run it through the ``ocr`` preprocessor
        or the ``image`` loader instead.
    """

    provider_name = "pdf"
    extensions = (".pdf",)
    mimetypes = ("application/pdf",)

    def __init__(
        self,
        *,
        per_page: bool = True,
        extract_metadata: bool = True,
        password: str | None = None,
        min_page_chars: int = 10,
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self.per_page = per_page
        self.extract_metadata = extract_metadata
        self.password = password
        self.min_page_chars = min_page_chars

    async def aload_source(self, source: SourceLike) -> list[Document]:
        """Extract text from one PDF.

        Args:
            source: A path or a bytes payload.

        Returns:
            One document per page, or one for the whole file.

        Raises:
            IngestionError: When the file cannot be parsed.
        """
        pypdf = require("pypdf", extra="loaders", feature="The PDF loader")
        raw = self._read_bytes(source)
        path = str(source) if not isinstance(source, bytes) else None
        base = self._base_metadata(source) if not isinstance(source, bytes) else {}

        def _extract() -> tuple[list[str], dict[str, Any]]:
            try:
                reader = pypdf.PdfReader(io.BytesIO(raw))
                if reader.is_encrypted and reader.decrypt(self.password or "") == 0:
                    raise IngestionError(
                        f"{path or 'This PDF'} is encrypted.",
                        hint="Pass password=... to the loader.",
                    )
                pages = [(page.extract_text() or "") for page in reader.pages]
            except IngestionError:
                raise
            except Exception as exc:
                raise IngestionError(
                    f"Could not read {path or 'the PDF'}: {exc}",
                    hint="The file may be corrupt. Try opening it in a viewer first.",
                ) from exc

            meta: dict[str, Any] = {"page_count": len(pages)}
            if self.extract_metadata and reader.metadata:
                for key, value in reader.metadata.items():
                    clean = str(key).lstrip("/").lower()
                    if value and clean in {"title", "author", "subject", "creator", "producer"}:
                        meta[clean] = str(value)
            return pages, meta

        pages, pdf_meta = await to_thread(_extract)
        metadata = {**base, **pdf_meta}

        if self.per_page:
            documents = []
            for number, text in enumerate(pages, start=1):
                cleaned = normalize_whitespace(normalize_unicode(text))
                if len(cleaned) < self.min_page_chars:
                    continue
                documents.append(
                    Document(
                        content=cleaned,
                        metadata={**metadata, "page": number},
                        source=path,
                        mimetype="application/pdf",
                    )
                )
            if not documents and pages:
                self._log.warning(
                    "%s produced no extractable text — it is probably a scan. "
                    "Try the OCR preprocessor.",
                    path or "PDF",
                )
            return documents

        joined = normalize_whitespace(normalize_unicode("\n\n".join(pages)))
        if not joined:
            return []
        return [
            Document(content=joined, metadata=metadata, source=path, mimetype="application/pdf")
        ]


@register.loader(
    "docx",
    aliases=("word", "doc"),
    description="Microsoft Word documents, including tables.",
)
class DocxLoader(Loader):
    """Extracts text from ``.docx`` files using ``python-docx``.

    Args:
        include_tables: Render tables as pipe-delimited text and append them.
        include_headers: Include header and footer text.
        **config: Forwarded to :class:`~windlass.interfaces.loader.Loader`.

    Raises:
        MissingDependencyError: When ``python-docx`` is not installed.

    Note:
        Only the modern ``.docx`` format is supported. Legacy ``.doc`` files must
        be converted first (``libreoffice --convert-to docx``).
    """

    provider_name = "docx"
    extensions = (".docx",)

    def __init__(
        self, *, include_tables: bool = True, include_headers: bool = False, **config: Any
    ) -> None:
        super().__init__(**config)
        self.include_tables = include_tables
        self.include_headers = include_headers

    async def aload_source(self, source: SourceLike) -> list[Document]:
        """Extract text from one Word document.

        Args:
            source: A path or a bytes payload.

        Returns:
            A single-element list.

        Raises:
            IngestionError: When the file cannot be parsed.
        """
        docx = require("docx", extra="loaders", feature="The DOCX loader")
        raw = self._read_bytes(source)
        path = str(source) if not isinstance(source, bytes) else None
        base = self._base_metadata(source) if not isinstance(source, bytes) else {}

        def _extract() -> tuple[str, dict[str, Any]]:
            try:
                document = docx.Document(io.BytesIO(raw))
            except Exception as exc:
                raise IngestionError(
                    f"Could not read {path or 'the document'}: {exc}",
                    hint="Legacy .doc files are not supported; convert them to .docx.",
                ) from exc

            parts: list[str] = []
            headings: list[str] = []
            for paragraph in document.paragraphs:
                text = paragraph.text.strip()
                if not text:
                    continue
                style = (paragraph.style.name or "").lower() if paragraph.style else ""
                if style.startswith("heading"):
                    headings.append(text)
                    level = "".join(ch for ch in style if ch.isdigit()) or "1"
                    parts.append(f"{'#' * min(6, int(level))} {text}")
                else:
                    parts.append(text)

            if self.include_tables:
                for index, table in enumerate(document.tables, start=1):
                    rows = [
                        " | ".join(cell.text.strip() for cell in row.cells) for row in table.rows
                    ]
                    if rows:
                        parts.append(f"\n[Table {index}]\n" + "\n".join(rows))

            if self.include_headers:
                for section in document.sections:
                    for container in (section.header, section.footer):
                        text = "\n".join(
                            p.text.strip() for p in container.paragraphs if p.text.strip()
                        )
                        if text:
                            parts.append(text)

            meta: dict[str, Any] = {"paragraphs": len(document.paragraphs)}
            if headings:
                meta["headings"] = headings[:50]
            core = getattr(document, "core_properties", None)
            if core is not None:
                for field in ("title", "author", "subject"):
                    value = getattr(core, field, None)
                    if value:
                        meta[field] = str(value)
            return "\n\n".join(parts), meta

        text, meta = await to_thread(_extract)
        if not text.strip():
            return []
        return [
            Document(
                content=normalize_whitespace(text),
                metadata={**base, **meta},
                source=path,
                mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        ]


@register.loader(
    "pptx",
    aliases=("powerpoint", "ppt"),
    description="PowerPoint decks, one document per slide.",
)
class PptxLoader(Loader):
    """Extracts text from ``.pptx`` decks using ``python-pptx``.

    Args:
        per_slide: One document per slide (recommended — slide numbers are
            natural citations).
        include_notes: Include the speaker notes, which often carry the actual
            argument the slide only gestures at.
        **config: Forwarded to :class:`~windlass.interfaces.loader.Loader`.

    Raises:
        MissingDependencyError: When ``python-pptx`` is not installed.
    """

    provider_name = "pptx"
    extensions = (".pptx",)

    def __init__(
        self, *, per_slide: bool = True, include_notes: bool = True, **config: Any
    ) -> None:
        super().__init__(**config)
        self.per_slide = per_slide
        self.include_notes = include_notes

    async def aload_source(self, source: SourceLike) -> list[Document]:
        """Extract text from one deck.

        Args:
            source: A path or a bytes payload.

        Returns:
            One document per slide, or one for the whole deck.

        Raises:
            IngestionError: When the file cannot be parsed.
        """
        pptx = require("pptx", extra="loaders", feature="The PPTX loader")
        raw = self._read_bytes(source)
        path = str(source) if not isinstance(source, bytes) else None
        base = self._base_metadata(source) if not isinstance(source, bytes) else {}

        def _extract() -> list[tuple[int, str, str]]:
            try:
                deck = pptx.Presentation(io.BytesIO(raw))
            except Exception as exc:
                raise IngestionError(f"Could not read {path or 'the deck'}: {exc}") from exc

            slides: list[tuple[int, str, str]] = []
            for number, slide in enumerate(deck.slides, start=1):
                parts = [
                    shape.text.strip()
                    for shape in slide.shapes
                    if getattr(shape, "has_text_frame", False) and shape.text.strip()
                ]
                notes = ""
                if self.include_notes and slide.has_notes_slide:
                    frame = slide.notes_slide.notes_text_frame
                    notes = (frame.text or "").strip() if frame else ""
                slides.append((number, "\n".join(parts), notes))
            return slides

        slides = await to_thread(_extract)
        mimetype = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

        if self.per_slide:
            documents = []
            for number, body, notes in slides:
                content = body
                if notes:
                    content = f"{body}\n\n[Speaker notes]\n{notes}" if body else notes
                if not content.strip():
                    continue
                documents.append(
                    Document(
                        content=normalize_whitespace(content),
                        metadata={**base, "slide": number, "slide_count": len(slides)},
                        source=path,
                        mimetype=mimetype,
                    )
                )
            return documents

        joined = "\n\n".join(
            f"[Slide {n}]\n{body}" + (f"\n[Notes] {notes}" if notes else "")
            for n, body, notes in slides
            if body or notes
        )
        if not joined.strip():
            return []
        return [
            Document(
                content=normalize_whitespace(joined),
                metadata={**base, "slide_count": len(slides)},
                source=path,
                mimetype=mimetype,
            )
        ]


@register.loader(
    "xlsx",
    aliases=("excel", "xls", "spreadsheet"),
    description="Excel workbooks, one document per sheet.",
)
class XlsxLoader(Loader):
    """Extracts data from ``.xlsx`` workbooks using ``openpyxl``.

    Sheets are rendered as Markdown tables, which models read considerably
    better than raw CSV — the header row stays visually attached to its columns.

    Args:
        per_sheet: One document per sheet.
        max_rows: Row ceiling per sheet. Guards against a stray million-row sheet.
        header_row: Which row holds the column names, 1-based. ``0`` means none.
        skip_empty: Skip rows where every cell is empty.
        **config: Forwarded to :class:`~windlass.interfaces.loader.Loader`.

    Raises:
        MissingDependencyError: When ``openpyxl`` is not installed.
    """

    provider_name = "xlsx"
    extensions = (".xlsx", ".xlsm")

    def __init__(
        self,
        *,
        per_sheet: bool = True,
        max_rows: int = 10_000,
        header_row: int = 1,
        skip_empty: bool = True,
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self.per_sheet = per_sheet
        self.max_rows = max_rows
        self.header_row = header_row
        self.skip_empty = skip_empty

    async def aload_source(self, source: SourceLike) -> list[Document]:
        """Extract data from one workbook.

        Args:
            source: A path or a bytes payload.

        Returns:
            One document per sheet, or one for the whole workbook.

        Raises:
            IngestionError: When the file cannot be parsed.
        """
        openpyxl = require("openpyxl", extra="loaders", feature="The XLSX loader")
        raw = self._read_bytes(source)
        path = str(source) if not isinstance(source, bytes) else None
        base = self._base_metadata(source) if not isinstance(source, bytes) else {}

        def _extract() -> list[tuple[str, str, int]]:
            try:
                workbook = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
            except Exception as exc:
                raise IngestionError(
                    f"Could not read {path or 'the workbook'}: {exc}",
                    hint="Legacy .xls files are not supported; convert them to .xlsx.",
                ) from exc

            sheets: list[tuple[str, str, int]] = []
            try:
                for worksheet in workbook.worksheets:
                    rows: list[list[str]] = []
                    for index, row in enumerate(worksheet.iter_rows(values_only=True)):
                        if index >= self.max_rows:
                            break
                        cells = ["" if c is None else str(c).strip() for c in row]
                        if self.skip_empty and not any(cells):
                            continue
                        rows.append(cells)
                    if rows:
                        sheets.append(
                            (worksheet.title, _to_markdown(rows, self.header_row), len(rows))
                        )
            finally:
                workbook.close()
            return sheets

        sheets = await to_thread(_extract)
        mimetype = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        if self.per_sheet:
            return [
                Document(
                    content=table,
                    metadata={**base, "sheet": name, "rows": rows},
                    source=path,
                    mimetype=mimetype,
                )
                for name, table, rows in sheets
            ]

        joined = "\n\n".join(f"## {name}\n\n{table}" for name, table, _ in sheets)
        if not joined.strip():
            return []
        return [
            Document(
                content=joined,
                metadata={**base, "sheets": [name for name, _, _ in sheets]},
                source=path,
                mimetype=mimetype,
            )
        ]


def _to_markdown(rows: list[list[str]], header_row: int) -> str:
    """Render a grid of cells as a Markdown table.

    Args:
        rows: The cell grid.
        header_row: 1-based index of the header row; ``0`` for no header.

    Returns:
        A Markdown table.

    Example:
        >>> print(_to_markdown([["a", "b"], ["1", "2"]], 1))
        | a | b |
        | --- | --- |
        | 1 | 2 |
    """
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    padded = [[*r, *([""] * (width - len(r)))] for r in rows]

    if header_row and len(padded) >= header_row:
        header = padded[header_row - 1]
        body = padded[header_row:]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
        ]
        lines += ["| " + " | ".join(r) + " |" for r in body]
        return "\n".join(lines)

    return "\n".join("| " + " | ".join(r) + " |" for r in padded)
