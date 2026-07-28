"""OCR fallback and LLM-driven metadata extraction.

Two preprocessors that reach outside the text they are given:

* :class:`OCRPreprocessor` rescues scanned documents. ``pypdf`` returns empty
  text for an image-only PDF; this notices and re-reads the file with OCR.
* :class:`LLMMetadataPreprocessor` asks a model for structured metadata —
  title, summary, topics, entities, document type — which turns a flat corpus
  into something you can filter and route on.

Example:
    >>> from windlass.core.types import Document
    >>> from windlass.providers.llm.fake import FakeLLM
    >>> llm = FakeLLM(responses=['{"title": "Q3 Report", "topics": ["revenue"]}'])
    >>> doc = LLMMetadataPreprocessor(llm=llm).process([Document(content="...")])[0]
    >>> doc.metadata["title"]
    'Q3 Report'
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from windlass.core.exceptions import ConfigurationError
from windlass.core.registry import register
from windlass.core.text import truncate_tokens
from windlass.core.types import Document
from windlass.interfaces.llm import LLM
from windlass.interfaces.preprocessor import Preprocessor

__all__ = ["METADATA_PROMPT", "LLMMetadataPreprocessor", "OCRPreprocessor"]

#: Extensions the OCR fallback knows how to re-read.
_OCR_SOURCES = (".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp")

METADATA_PROMPT = """\
Read the document excerpt below and extract structured metadata.

Reply with a single JSON object and nothing else, using exactly these keys:
  "title":       a short descriptive title
  "summary":     one or two sentences
  "topics":      up to five topic keywords, as a list of strings
  "entities":    up to eight named entities (people, organisations, products)
  "document_type": one of report, email, contract, article, manual, code, other

<document>
{content}
</document>
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@register.preprocessor(
    "ocr",
    description="Re-reads scanned pages with OCR when text extraction came back empty.",
)
class OCRPreprocessor(Preprocessor):
    """OCR fallback for documents whose text layer is missing or unusable.

    A scanned PDF is a stack of images. Text extraction returns nothing, and the
    document silently disappears from your index. This preprocessor detects that
    and re-reads the source with OCR.

    Args:
        min_chars: Documents with at least this much text are passed through
            untouched — OCR is only a fallback, never a replacement.
        language: Tesseract language code(s).
        dpi: Rendering resolution for PDF pages. 300 is the usual sweet spot;
            higher is slower with diminishing returns.
        max_pages: Ceiling on pages rendered per PDF, to bound the cost of a
            mistakenly-included 900-page scan.
        **config: Forwarded to :class:`~windlass.interfaces.preprocessor.Preprocessor`.

    Raises:
        MissingDependencyError: When the OCR extras are not installed and a
            document actually needs OCR.

    Note:
        PDF rendering needs ``pdf2image`` plus Poppler, which Windlass does not
        install for you. When rendering is unavailable the document is passed
        through unchanged and a warning is logged, rather than failing the run.
    """

    provider_name = "ocr"

    def __init__(
        self,
        *,
        min_chars: int = 50,
        language: str = "eng",
        dpi: int = 300,
        max_pages: int = 50,
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self.min_chars = min_chars
        self.language = language
        self.dpi = dpi
        self.max_pages = max_pages

    async def aprocess_one(self, document: Document) -> list[Document]:
        """Re-read the document with OCR when its text is missing.

        Args:
            document: The document to check.

        Returns:
            The original document, or an OCR'd replacement.
        """
        if len(document.content.strip()) >= self.min_chars:
            return [document]

        source = document.source
        if not source or Path(source).suffix.lower() not in _OCR_SOURCES:
            return [document]

        suffix = Path(source).suffix.lower()
        try:
            text = (
                await self._ocr_pdf(source, document)
                if suffix == ".pdf"
                else await self._ocr_image(source)
            )
        except Exception as exc:
            self._log.warning("OCR fallback failed for %s: %s", source, exc)
            return [document]

        if len(text.strip()) < self.min_chars:
            return [document]

        return [
            document.model_copy(
                update={
                    "content": text,
                    "metadata": {**document.metadata, "ocr": True, "ocr_language": self.language},
                }
            )
        ]

    async def _ocr_image(self, source: str) -> str:
        """OCR a single image file."""
        from windlass.providers.loaders.media import ImageLoader

        loader = ImageLoader(language=self.language, min_chars=0)
        documents = await loader.aload_source(source)
        return documents[0].content if documents else ""

    async def _ocr_pdf(self, source: str, document: Document) -> str:
        """Render a PDF's pages to images and OCR them."""
        from windlass.core.concurrency import to_thread
        from windlass.core.lazy import is_available, require

        if not is_available("pdf2image"):
            self._log.warning(
                "OCR for PDFs needs pdf2image and Poppler; %s was left as-is. "
                "Install with: pip install pdf2image (plus the Poppler binaries).",
                source,
            )
            return ""

        pdf2image = require("pdf2image", extra="ocr", feature="PDF OCR")
        pytesseract = require("pytesseract", extra="ocr", feature="PDF OCR")
        page = document.metadata.get("page")

        def _render_and_read() -> str:
            kwargs: dict[str, Any] = {"dpi": self.dpi}
            if isinstance(page, int):
                kwargs["first_page"] = kwargs["last_page"] = page
            images = pdf2image.convert_from_path(source, **kwargs)[: self.max_pages]
            return "\n\n".join(
                pytesseract.image_to_string(image, lang=self.language) for image in images
            )

        return await to_thread(_render_and_read)


@register.preprocessor(
    "llm_metadata",
    aliases=("extract", "auto-metadata"),
    description="Extracts title, summary, topics and entities with a model.",
)
class LLMMetadataPreprocessor(Preprocessor):
    """Extracts structured metadata using a language model.

    Args:
        llm: The model to ask. Use a small, cheap one — this is extraction, not
            reasoning.
        max_input_tokens: How much of each document to show the model. The
            opening of a document usually carries its title and topic.
        fields: Metadata keys to keep from the model's reply.
        overwrite: Replace metadata keys that already exist. Off by default, so
            metadata the loader extracted from the file itself wins.
        **config: Forwarded to :class:`~windlass.interfaces.preprocessor.Preprocessor`.

    Raises:
        ConfigurationError: When no model is supplied.

    Performance:
        One model call per document, run with the batch concurrency of the
        preprocessor base class. On a large corpus this is the most expensive
        preprocessor by a wide margin — measure before enabling it on millions
        of documents.
    """

    provider_name = "llm_metadata"

    def __init__(
        self,
        *,
        llm: LLM | None = None,
        max_input_tokens: int = 1500,
        fields: tuple[str, ...] = ("title", "summary", "topics", "entities", "document_type"),
        overwrite: bool = False,
        **config: Any,
    ) -> None:
        if llm is None:
            raise ConfigurationError(
                "Metadata extraction needs a language model.",
                hint="Pass llm=Windlass.llm('openai', model='gpt-4o-mini').",
            )
        super().__init__(**config)
        self.llm = llm
        self.max_input_tokens = max_input_tokens
        self.fields = tuple(fields)
        self.overwrite = overwrite

    async def aprocess_one(self, document: Document) -> list[Document]:
        """Ask the model for metadata and merge it in.

        A failed or unparseable reply leaves the document untouched — metadata
        enrichment is an improvement, never a gate.

        Args:
            document: The document to enrich.

        Returns:
            A single-element list.
        """
        excerpt = truncate_tokens(document.content, self.max_input_tokens, suffix="\n[…]")
        try:
            completion = await self.llm.acomplete(
                METADATA_PROMPT.format(content=excerpt), max_tokens=400
            )
            extracted = _parse_json(completion.content)
        except Exception as exc:
            self._log.warning("Metadata extraction failed for %s: %s", document.id, exc)
            return [document]

        if not extracted:
            return [document]

        metadata = dict(document.metadata)
        for key in self.fields:
            value = extracted.get(key)
            if value in (None, "", [], {}):
                continue
            if key in metadata and not self.overwrite:
                continue
            metadata[key] = value
        metadata["llm_metadata"] = True
        return [document.model_copy(update={"metadata": metadata})]


def _parse_json(text: str) -> dict[str, Any]:
    r"""Extract a JSON object from a model reply.

    Models wrap JSON in prose and code fences no matter how firmly you ask them
    not to, so this finds the first balanced object rather than trusting the
    whole reply to parse.

    Args:
        text: The model's reply.

    Returns:
        The parsed object, or ``{}`` when none could be found.

    Example:
        >>> _parse_json('Sure!\\n```json\\n{"a": 1}\\n```')
        {'a': 1}
        >>> _parse_json("no json here")
        {}
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.MULTILINE)
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    match = _JSON_RE.search(text)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
