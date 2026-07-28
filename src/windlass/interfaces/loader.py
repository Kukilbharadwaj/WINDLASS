"""The document-loader interface.

A loader turns a source — a path, a glob, a URL, a byte stream — into
:class:`~windlass.core.types.Document` objects. Format detection is automatic:
:func:`~windlass.rag.loading.load` inspects the extension and MIME type and picks
the right registered loader, so ``.ingest("./docs")`` handles a folder of mixed
PDFs, spreadsheets and Markdown without any configuration.

Implementers override one coroutine, :meth:`Loader.aload_source`.

Example:
    >>> import tempfile, pathlib
    >>> p = pathlib.Path(tempfile.mkdtemp()) / "note.txt"
    >>> _ = p.write_text("hello world")
    >>> from windlass.providers.loaders.text import TextLoader
    >>> TextLoader().load(p)[0].content
    'hello world'
"""

from __future__ import annotations

import abc
import os
from collections.abc import AsyncIterator, Iterable, Sequence
from pathlib import Path
from typing import Any

from windlass.core.concurrency import gather_bounded, run_sync
from windlass.core.config import settings
from windlass.core.exceptions import IngestionError
from windlass.core.types import Document
from windlass.interfaces.base import Component

__all__ = ["Loader", "SourceLike"]


def _label(source: SourceLike) -> str:
    """Render a source for an error message.

    Byte payloads have no useful string form — ``f"{b'...'}"`` produces
    ``b'...'`` — so they are described by size instead.

    Args:
        source: The source being reported on.

    Returns:
        A short human-readable label.
    """
    if isinstance(source, bytes):
        return f"<{len(source)} bytes>"
    return str(source)


#: Anything a loader accepts as a source.
SourceLike = str | Path | bytes | Any


class Loader(Component):
    """Abstract document loader.

    Args:
        encoding: Text encoding used for character-based formats.
        metadata: Extra metadata merged into every produced document — useful
            for tagging a whole corpus with a tenant or collection id.
        recursive: Whether directory sources are walked recursively.
        on_error: ``"raise"`` aborts the batch, ``"skip"`` logs and continues.
            ``"skip"`` is the right default for a 10,000-file corpus where one
            corrupt PDF should not lose the run.
        name: Component name for traces.
        **config: Loader-specific options.

    Attributes:
        extensions: File suffixes this loader claims, used for auto-detection.
        mimetypes: MIME types this loader claims.

    Example:
        Implementing a loader takes one method::

            class MyLoader(Loader):
                provider_name = "mine"
                extensions = (".mine",)

                async def aload_source(self, source):
                    return [Document(content=read(source), source=str(source))]
    """

    kind = "loader"
    provider_name: str = "loader"

    #: File suffixes (lowercase, with dot) this loader handles.
    extensions: tuple[str, ...] = ()

    #: MIME types this loader handles.
    mimetypes: tuple[str, ...] = ()

    #: True when the loader takes URLs rather than filesystem paths.
    handles_urls: bool = False

    def __init__(
        self,
        *,
        encoding: str = "utf-8",
        metadata: dict[str, Any] | None = None,
        recursive: bool = True,
        on_error: str = "skip",
        name: str | None = None,
        **config: Any,
    ) -> None:
        if on_error not in {"raise", "skip"}:
            raise ValueError("on_error must be 'raise' or 'skip'")
        super().__init__(
            name=name or self.provider_name,
            encoding=encoding,
            metadata=metadata or {},
            recursive=recursive,
            on_error=on_error,
            **config,
        )
        self.encoding = encoding
        self.extra_metadata: dict[str, Any] = dict(metadata or {})
        self.recursive = recursive
        self.on_error = on_error

    # -- provider hooks ---------------------------------------------------
    @abc.abstractmethod
    async def aload_source(self, source: SourceLike) -> list[Document]:
        """Load one concrete source.

        Called once per file. Directory expansion and error handling are done
        for you by :meth:`aload`.

        Args:
            source: A single path, URL or byte payload.

        Returns:
            The documents extracted from it. A PDF may return one document per
            page or one for the whole file — that is the loader's choice.

        Raises:
            IngestionError: When the source cannot be parsed.
        """

    @classmethod
    def can_handle(cls, source: SourceLike) -> bool:
        """Return whether this loader claims ``source``.

        Args:
            source: Path, URL or payload to test.

        Returns:
            True when the extension or scheme matches this loader.

        Example:
            >>> from windlass.providers.loaders.text import TextLoader
            >>> TextLoader.can_handle("notes.txt")
            True
            >>> TextLoader.can_handle("scan.pdf")
            False
        """
        if isinstance(source, str | Path):
            text = str(source)
            if text.startswith(("http://", "https://")):
                return cls.handles_urls
            return Path(text).suffix.lower() in cls.extensions
        return False

    # -- public API -------------------------------------------------------
    async def aload(
        self, source: SourceLike | Sequence[SourceLike], *, concurrency: int | None = None
    ) -> list[Document]:
        """Load one or many sources.

        Directories are expanded (respecting :attr:`recursive` and this loader's
        :attr:`extensions`), and files are read concurrently.

        Args:
            source: A path, URL, byte payload, or a sequence of them. A
                directory path expands to its matching files.
            concurrency: Maximum simultaneous reads. Defaults to the global
                ``max_concurrency`` setting.

        Returns:
            Every extracted document, with :attr:`extra_metadata` merged in.

        Raises:
            IngestionError: When ``on_error='raise'`` and a source fails, or
                when no source could be read at all.

        Performance:
            Blocking parsers run on the thread pool, so a folder of 500 PDFs is
            parsed in parallel rather than serially.
        """
        sources = self._expand(source)
        if not sources:
            return []

        limit = concurrency or settings().max_concurrency
        results = await gather_bounded(
            [self._load_guarded(s) for s in sources], limit=limit, return_exceptions=True
        )

        documents: list[Document] = []
        failures: list[tuple[Any, BaseException]] = []
        for src, result in zip(sources, results, strict=True):
            if isinstance(result, BaseException):
                failures.append((src, result))
                continue
            documents.extend(result)

        if failures:
            if self.on_error == "raise":
                src, exc = failures[0]
                raise IngestionError(
                    f"Failed to load {src}: {exc}",
                    context={"source": str(src), "failures": len(failures)},
                ) from exc
            for src, exc in failures:
                self._log.warning("Skipping %s: %s", src, exc)
            if not documents:
                src, exc = failures[0]
                raise IngestionError(
                    f"No documents could be loaded; the first failure was {src}: {exc}",
                    hint="Check the paths and that the right extra is installed.",
                    context={"failures": len(failures)},
                ) from exc

        for doc in documents:
            if self.extra_metadata:
                doc.metadata = {**self.extra_metadata, **doc.metadata}
            doc.metadata.setdefault("loader", self.name)
        return documents

    def load(self, source: SourceLike | Sequence[SourceLike]) -> list[Document]:
        """Blocking :meth:`aload`."""
        return run_sync(self.aload(source))

    async def astream_load(
        self, source: SourceLike | Sequence[SourceLike]
    ) -> AsyncIterator[Document]:
        """Yield documents one at a time instead of building a list.

        Use this for corpora too large to hold in memory: combined with
        ``Pipeline.aingest_stream`` it keeps peak memory flat regardless of
        corpus size.

        Args:
            source: A path, URL, payload, or sequence of them.

        Yields:
            Documents as each source finishes parsing.
        """
        for item in self._expand(source):
            try:
                docs = await self.aload_source(item)
            except Exception as exc:
                if self.on_error == "raise":
                    raise IngestionError(f"Failed to load {_label(item)}: {exc}") from exc
                self._log.warning("Skipping %s: %s", item, exc)
                continue
            for doc in docs:
                if self.extra_metadata:
                    doc.metadata = {**self.extra_metadata, **doc.metadata}
                doc.metadata.setdefault("loader", self.name)
                yield doc

    # -- helpers ----------------------------------------------------------
    async def _load_guarded(self, source: SourceLike) -> list[Document]:
        """Load one source, wrapping unexpected errors as IngestionError."""
        try:
            return await self.aload_source(source)
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(
                f"{type(self).__name__} could not read {_label(source)}: {exc}",
                context={"source": str(source)},
            ) from exc

    def _expand(self, source: SourceLike | Sequence[SourceLike]) -> list[SourceLike]:
        """Normalise a source argument into a flat list of concrete sources."""
        if isinstance(source, str | Path | bytes):
            candidates: Iterable[Any] = [source]
        elif isinstance(source, Sequence):
            candidates = source
        else:
            candidates = [source]

        expanded: list[SourceLike] = []
        for item in candidates:
            if isinstance(item, bytes):
                expanded.append(item)
                continue
            if isinstance(item, str) and item.startswith(("http://", "https://")):
                expanded.append(item)
                continue
            path = Path(item) if isinstance(item, str | Path) else None
            if path is None:
                expanded.append(item)
                continue
            if path.is_dir():
                expanded.extend(self._walk(path))
            elif any(ch in str(path) for ch in "*?[") and not path.exists():
                base = Path(str(path).split("*")[0]).parent or Path(".")
                pattern = str(path.name) if base == path.parent else str(path)
                expanded.extend(sorted(base.glob(pattern)))
            else:
                expanded.append(path)
        return expanded

    def _walk(self, directory: Path) -> list[Path]:
        """List the files in ``directory`` this loader can handle."""
        pattern = "**/*" if self.recursive else "*"
        files = [p for p in sorted(directory.glob(pattern)) if p.is_file()]
        if self.extensions:
            files = [p for p in files if p.suffix.lower() in self.extensions]
        return files

    @staticmethod
    def _base_metadata(path: str | Path) -> dict[str, Any]:
        """Return the metadata every file-backed document should carry."""
        p = Path(path)
        meta: dict[str, Any] = {"source": str(p), "filename": p.name}
        try:
            stat = p.stat()
            meta["size_bytes"] = stat.st_size
            meta["modified_at"] = stat.st_mtime
        except OSError:  # pragma: no cover - source may be virtual
            pass
        meta["extension"] = p.suffix.lower()
        return meta

    def _read_bytes(self, source: SourceLike) -> bytes:
        """Read a path or pass bytes through, with a clear error on failure."""
        if isinstance(source, bytes):
            return source
        path = Path(source)
        if not path.is_file():
            raise IngestionError(
                f"{path} is not a readable file.",
                hint="Check the path, or pass bytes directly.",
                context={"source": str(path)},
            )
        max_bytes = int(os.getenv("WINDLASS_MAX_FILE_BYTES", "0") or 0)
        if max_bytes and path.stat().st_size > max_bytes:
            raise IngestionError(
                f"{path} exceeds WINDLASS_MAX_FILE_BYTES ({max_bytes} bytes).",
                context={"source": str(path), "size": path.stat().st_size},
            )
        return path.read_bytes()
