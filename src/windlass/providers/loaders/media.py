"""Image (OCR) and audio (transcription) loaders.

Both turn non-text media into text so it can be chunked, embedded and retrieved
exactly like a document.

Install with::

    pip install "windlass[ocr]"     # images — also needs the Tesseract binary
    pip install "windlass[audio]"   # audio

Example:
    >>> from windlass import Windlass                          # doctest: +SKIP
    >>> Windlass.loader("image").load("./scans")              # doctest: +SKIP
    >>> Windlass.loader("audio").load("./meeting.mp3")        # doctest: +SKIP
"""

from __future__ import annotations

import io
from typing import Any, ClassVar

from windlass.core.concurrency import to_thread
from windlass.core.exceptions import IngestionError
from windlass.core.lazy import require
from windlass.core.registry import register
from windlass.core.text import normalize_whitespace
from windlass.core.types import Document
from windlass.interfaces.loader import Loader, SourceLike

__all__ = ["AudioLoader", "ImageLoader"]


@register.loader(
    "image",
    aliases=("ocr", "png", "jpg"),
    description="Images, transcribed with Tesseract OCR.",
)
class ImageLoader(Loader):
    """Extracts text from images using Tesseract.

    Args:
        language: Tesseract language code(s), e.g. ``"eng"`` or ``"eng+deu"``.
        psm: Tesseract page segmentation mode. ``3`` (fully automatic) suits
            documents; ``6`` (assume a uniform block) suits screenshots.
        preprocess: Convert to greyscale and auto-contrast before OCR, which
            measurably improves accuracy on photographs and low-contrast scans.
        min_chars: Return no document when fewer characters were recognised —
            below this it is noise, not text.
        **config: Forwarded to :class:`~windlass.interfaces.loader.Loader`.

    Raises:
        MissingDependencyError: When ``pytesseract`` or ``pillow`` is missing.
        IngestionError: When the Tesseract binary is not installed, or the image
            cannot be read.

    Note:
        ``pytesseract`` is a wrapper, not an OCR engine. The Tesseract binary
        must be installed separately (``apt install tesseract-ocr``,
        ``brew install tesseract``, or the Windows installer). The error message
        says so if it is missing.
    """

    provider_name = "image"
    extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp")
    mimetypes = ("image/png", "image/jpeg", "image/tiff", "image/webp")

    def __init__(
        self,
        *,
        language: str = "eng",
        psm: int = 3,
        preprocess: bool = True,
        min_chars: int = 5,
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self.language = language
        self.psm = psm
        self.preprocess = preprocess
        self.min_chars = min_chars

    async def aload_source(self, source: SourceLike) -> list[Document]:
        """OCR one image.

        Args:
            source: A path or a bytes payload.

        Returns:
            A single-element list, or empty when too little text was recognised.

        Raises:
            IngestionError: When OCR fails or Tesseract is unavailable.
        """
        pytesseract = require("pytesseract", extra="ocr", feature="The image OCR loader")
        pil = require("PIL.Image", extra="ocr", feature="The image OCR loader")
        raw = self._read_bytes(source)
        path = str(source) if not isinstance(source, bytes) else None
        base = self._base_metadata(source) if not isinstance(source, bytes) else {}

        def _ocr() -> tuple[str, dict[str, Any]]:
            try:
                image = pil.open(io.BytesIO(raw))
                meta = {"width": image.width, "height": image.height, "format": image.format}
                if self.preprocess:
                    image = _prepare(image)
                text = pytesseract.image_to_string(
                    image, lang=self.language, config=f"--psm {self.psm}"
                )
            except Exception as exc:
                if "tesseract" in str(exc).lower():
                    raise IngestionError(
                        "The Tesseract OCR binary is not installed.",
                        hint="Install it: apt install tesseract-ocr | "
                        "brew install tesseract | choco install tesseract",
                    ) from exc
                raise IngestionError(f"OCR failed for {path or 'the image'}: {exc}") from exc
            return text, meta

        text, meta = await to_thread(_ocr)
        cleaned = normalize_whitespace(text)
        if len(cleaned) < self.min_chars:
            self._log.debug("OCR produced only %d characters for %s", len(cleaned), path)
            return []
        return [
            Document(
                content=cleaned,
                metadata={**base, **meta, "ocr_language": self.language},
                source=path,
                mimetype="text/plain",
            )
        ]


@register.loader(
    "audio",
    aliases=("mp3", "wav", "speech"),
    description="Audio files, transcribed locally with faster-whisper.",
)
class AudioLoader(Loader):
    """Transcribes audio using ``faster-whisper``.

    Args:
        model: Whisper model size — ``tiny``, ``base``, ``small``, ``medium``,
            ``large-v3``. Bigger is more accurate and slower.
        language: Force a language code, or ``None`` to auto-detect.
        device: ``"cpu"``, ``"cuda"``, or ``"auto"``.
        compute_type: Quantisation, e.g. ``"int8"`` on CPU or ``"float16"`` on GPU.
        per_segment: One document per transcript segment, each carrying its
            timestamp — the right choice for anything you want to cite.
        vad_filter: Skip silence with voice-activity detection, which speeds up
            transcription of sparse recordings considerably.
        **config: Forwarded to :class:`~windlass.interfaces.loader.Loader`.

    Raises:
        MissingDependencyError: When ``faster-whisper`` is not installed.
        IngestionError: When the audio cannot be transcribed.

    Performance:
        Transcription is slow and CPU-heavy. It runs on a worker thread so it
        never blocks the event loop, but expect roughly real-time on CPU with
        the ``base`` model.
    """

    provider_name = "audio"
    extensions = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".webm", ".mp4")

    #: Model instances are cached per configuration; loading one costs seconds.
    _models: ClassVar[dict[tuple[str, str, str], Any]] = {}

    def __init__(
        self,
        *,
        model: str = "base",
        language: str | None = None,
        device: str = "auto",
        compute_type: str = "int8",
        per_segment: bool = False,
        vad_filter: bool = True,
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self.model_size = model
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.per_segment = per_segment
        self.vad_filter = vad_filter

    def _model(self) -> Any:
        """Return the cached Whisper model, loading it on first use."""
        key = (self.model_size, self.device, self.compute_type)
        cached = type(self)._models.get(key)
        if cached is not None:
            return cached
        fw = require("faster_whisper", extra="audio", feature="The audio loader")
        try:
            model = fw.WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
        except Exception as exc:
            raise IngestionError(
                f"Could not load the Whisper model {self.model_size!r}: {exc}",
                hint="Check the model name and that there is disk space for the download.",
            ) from exc
        type(self)._models[key] = model
        return model

    async def aload_source(self, source: SourceLike) -> list[Document]:
        """Transcribe one audio file.

        Args:
            source: A path. Byte payloads are written to a temporary file first,
                because the decoder needs a seekable path.

        Returns:
            One document per segment, or one for the whole transcript.

        Raises:
            IngestionError: When transcription fails.
        """
        path = str(source) if not isinstance(source, bytes) else None
        base = self._base_metadata(source) if not isinstance(source, bytes) else {}

        if path is None:
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as handle:
                handle.write(source)  # type: ignore[arg-type]
                path = handle.name

        def _transcribe() -> tuple[list[Any], Any]:
            try:
                segments, info = self._model().transcribe(
                    path,
                    language=self.language,
                    vad_filter=self.vad_filter,
                    beam_size=5,
                )
                return list(segments), info
            except Exception as exc:
                raise IngestionError(
                    f"Could not transcribe {path}: {exc}",
                    hint="Check the file is valid audio and that ffmpeg is installed.",
                ) from exc

        segments, info = await to_thread(_transcribe)
        meta = {
            **base,
            "language": getattr(info, "language", self.language or "unknown"),
            "duration_seconds": round(getattr(info, "duration", 0.0)),
            "whisper_model": self.model_size,
        }

        if self.per_segment:
            documents = []
            for segment in segments:
                text = (segment.text or "").strip()
                if not text:
                    continue
                start = float(segment.start)
                documents.append(
                    Document(
                        content=text,
                        metadata={
                            **meta,
                            "start_seconds": round(start, 1),
                            "end_seconds": round(float(segment.end), 1),
                            "timestamp": f"{int(start) // 60:02d}:{int(start) % 60:02d}",
                        },
                        source=path,
                        mimetype="text/plain",
                    )
                )
            return documents

        text = normalize_whitespace(" ".join((s.text or "").strip() for s in segments))
        if not text:
            return []
        return [Document(content=text, metadata=meta, source=path, mimetype="text/plain")]


def _prepare(image: Any) -> Any:
    """Greyscale and auto-contrast an image to improve OCR accuracy.

    Args:
        image: A PIL image.

    Returns:
        The preprocessed image, or the original if preprocessing fails.
    """
    try:
        from PIL import ImageOps

        return ImageOps.autocontrast(image.convert("L"))
    except Exception:
        return image
