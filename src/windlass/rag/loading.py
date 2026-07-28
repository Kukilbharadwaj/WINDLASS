"""Automatic loader selection.

``rag.ingest("./docs")`` on a folder of mixed PDFs, spreadsheets, Markdown and
URLs should just work. That is this module's job: inspect each source, pick the
registered loader that claims it, and dispatch.

Selection order:

1. A URL goes to the YouTube loader if it looks like a video, otherwise the web
   loader.
2. A file goes to whichever registered loader claims its extension.
3. An unclaimed text-like file falls back to the plain-text loader.
4. Anything else raises with a list of what *is* supported, so the fix is
   obvious (usually ``pip install "windlass[loaders]"``).

Example:
    >>> loader_for("notes.md").provider_name
    'markdown'
    >>> loader_for("https://example.com").provider_name
    'web'
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from windlass.core.exceptions import IngestionError
from windlass.core.registry import REGISTRY
from windlass.core.types import Document
from windlass.interfaces.loader import Loader, SourceLike

__all__ = ["AutoLoader", "extension_map", "loader_for"]

#: Extensions treated as plain text when no loader claims them.
_TEXT_FALLBACK = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".c",
    ".h",
    ".cpp",
    ".cs",
    ".sh",
    ".bat",
    ".ps1",
    ".sql",
    ".yaml",
    ".yml",
    ".toml",
    ".xml",
    ".conf",
    ".properties",
    ".gradle",
    ".dockerfile",
    "",
}


def extension_map() -> dict[str, str]:
    """Return the extension-to-loader-name mapping.

    Built by asking every registered loader which extensions it claims, so a
    third-party loader registered via a plugin participates automatically.

    Returns:
        A ``{".pdf": "pdf", ...}`` mapping.

    Example:
        >>> extension_map()[".md"]
        'markdown'
    """
    mapping: dict[str, str] = {}
    for spec in REGISTRY.specs("loader"):
        try:
            target = spec.resolve()
        except Exception:
            continue
        for extension in getattr(target, "extensions", ()):
            mapping.setdefault(extension, spec.name)
    return mapping


def loader_for(
    source: SourceLike | Sequence[SourceLike],
    *,
    metadata: dict[str, Any] | None = None,
    **config: Any,
) -> Loader:
    """Return a loader able to read ``source``.

    A heterogeneous sequence (or a directory containing several formats) yields
    an :class:`AutoLoader`, which dispatches per file.

    Args:
        source: A path, URL, directory, byte payload, or sequence of them.
        metadata: Metadata attached to every produced document.
        **config: Forwarded to the chosen loader's constructor.

    Returns:
        A ready-to-use loader.

    Raises:
        IngestionError: When nothing can read the source.

    Example:
        >>> loader_for("report.pdf").provider_name
        'pdf'
        >>> loader_for(b"raw bytes").provider_name
        'text'
    """
    if isinstance(source, Sequence) and not isinstance(source, str | bytes | Path):
        return AutoLoader(metadata=metadata, **config)

    if isinstance(source, bytes):
        return REGISTRY.create("loader", "text", metadata=metadata, **config)

    if isinstance(source, str) and source.startswith(("http://", "https://")):
        name = "youtube" if _is_youtube(source) else "web"
        return REGISTRY.create("loader", name, metadata=metadata, **config)

    path = Path(str(source))
    if path.is_dir() or any(ch in str(path) for ch in "*?["):
        return AutoLoader(metadata=metadata, **config)

    suffix = path.suffix.lower()
    claimed = extension_map().get(suffix)
    if claimed:
        return REGISTRY.create("loader", claimed, metadata=metadata, **config)

    if suffix in _TEXT_FALLBACK:
        return REGISTRY.create("loader", "text", metadata=metadata, **config)

    supported = ", ".join(sorted(extension_map())) or "(no loaders registered)"
    raise IngestionError(
        f"No loader can read {suffix or 'that source'!r}.",
        hint=f"Supported extensions: {supported}\n"
        'Most formats need: pip install "windlass[loaders]"',
        context={"source": str(source), "extension": suffix},
    )


class AutoLoader(Loader):
    """Dispatches each source to the loader that claims it.

    This is what makes ``ingest('./docs')`` handle a folder of mixed formats.
    Files with no matching loader are skipped with a warning rather than
    aborting the run, so one unreadable file cannot lose a large ingestion.

    Args:
        metadata: Metadata attached to every produced document.
        recursive: Walk directories recursively.
        on_error: ``"skip"`` (default) or ``"raise"``.
        loader_config: Per-loader keyword arguments, keyed by loader name, e.g.
            ``{"pdf": {"per_page": False}}``.
        **config: Forwarded to :class:`~windlass.interfaces.loader.Loader`.

    Example:
        >>> import tempfile, pathlib
        >>> folder = pathlib.Path(tempfile.mkdtemp())
        >>> _ = (folder / "a.txt").write_text("plain")
        >>> _ = (folder / "b.md").write_text("# heading")
        >>> sorted(d.metadata["extension"] for d in AutoLoader().load(folder))
        ['.md', '.txt']
    """

    provider_name = "auto"
    handles_urls = True

    def __init__(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        recursive: bool = True,
        on_error: str = "skip",
        loader_config: dict[str, dict[str, Any]] | None = None,
        **config: Any,
    ) -> None:
        super().__init__(metadata=metadata, recursive=recursive, on_error=on_error, **config)
        self.loader_config = dict(loader_config or {})
        self._cache: dict[str, Loader] = {}

    @classmethod
    def can_handle(cls, source: SourceLike) -> bool:
        """Return True — the auto loader accepts anything and routes it."""
        return True

    def _walk(self, directory: Path) -> list[Path]:
        """List every file in ``directory``, since routing happens per file."""
        pattern = "**/*" if self.recursive else "*"
        return [p for p in sorted(directory.glob(pattern)) if p.is_file()]

    async def aload_source(self, source: SourceLike) -> list[Document]:
        """Route one source to its loader and read it.

        Args:
            source: A single path, URL or byte payload.

        Returns:
            The extracted documents, or ``[]`` when no loader claims the source
            and ``on_error='skip'``.

        Raises:
            IngestionError: When no loader claims the source and
                ``on_error='raise'``.
        """
        try:
            delegate = self._delegate(source)
        except IngestionError:
            if self.on_error == "raise":
                raise
            self._log.debug("No loader for %s; skipping.", source)
            return []
        return await delegate.aload_source(source)

    def _delegate(self, source: SourceLike) -> Loader:
        """Return (and memoise) the loader for one source."""
        probe = loader_for(source, metadata=self.extra_metadata)
        name = probe.provider_name
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        options = self.loader_config.get(name, {})
        built = (
            REGISTRY.create("loader", name, metadata=self.extra_metadata, **options)
            if options
            else probe
        )
        self._cache[name] = built
        return built


def _is_youtube(url: str) -> bool:
    """Return whether ``url`` points at a YouTube video.

    Example:
        >>> _is_youtube("https://youtu.be/dQw4w9WgXcQ")
        True
        >>> _is_youtube("https://example.com/video")
        False
    """
    return "youtube.com/" in url or "youtu.be/" in url
