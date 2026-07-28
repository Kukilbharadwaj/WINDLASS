r"""Structure-aware chunking for Markdown and source code.

Generic splitters treat a document as a flat string. These two respect the
structure that is already there:

* :class:`MarkdownChunker` splits on headings and prepends the heading path to
  every chunk, so a chunk that says "the timeout defaults to 30s" retrieves as
  "Configuration > Networking > the timeout defaults to 30s". That context is
  frequently the difference between a hit and a miss.
* :class:`CodeChunker` splits at function and class boundaries for a dozen
  languages, so a retrieved chunk is a complete callable rather than its
  middle eight lines.

Neither has any dependencies.

Example:
    >>> md = "# Guide\n\nIntro text.\n\n## Setup\n\nRun the installer."
    >>> chunks = MarkdownChunker(chunk_size=200).split_text(md)
    >>> "Guide > Setup" in chunks[-1]
    True
"""

from __future__ import annotations

import re
from typing import Any

from windlass.core.registry import register
from windlass.interfaces.chunker import Chunker
from windlass.providers.chunkers.recursive import RecursiveChunker

__all__ = ["LANGUAGE_SEPARATORS", "CodeChunker", "MarkdownChunker"]

_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*#*\s*$", re.MULTILINE)
_FENCE_RE = re.compile(r"^(```|~~~)")


@register.chunker(
    "markdown",
    aliases=("md",),
    description="Splits Markdown on headings and prefixes each chunk with its heading path.",
)
class MarkdownChunker(Chunker):
    r"""Heading-aware Markdown splitter.

    Args:
        chunk_size: Target chunk length in characters. Sections longer than this
            are split further with the recursive splitter.
        overlap: Characters repeated when a section must be sub-split.
        max_heading_level: Deepest heading treated as a split point. ``2`` splits
            on ``#`` and ``##`` only, which keeps subsections together.
        include_heading_path: Prepend the breadcrumb (``Guide > Setup``) to each
            chunk. Strongly recommended — it is nearly free context.
        strip_code_fences: Keep fenced code blocks intact rather than letting a
            ``#`` comment inside them be mistaken for a heading.
        **config: Forwarded to :class:`~windlass.interfaces.chunker.Chunker`.

    Example:
        >>> chunker = MarkdownChunker(chunk_size=500, include_heading_path=False)
        >>> chunker.split_text("# A\n\ntext")[0]
        '# A\n\ntext'
    """

    provider_name = "markdown"
    unit = "char"

    def __init__(
        self,
        *,
        chunk_size: int = 1500,
        overlap: int | None = None,
        max_heading_level: int = 3,
        include_heading_path: bool = True,
        strip_code_fences: bool = True,
        **config: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, overlap=overlap, **config)
        self.max_heading_level = max_heading_level
        self.include_heading_path = include_heading_path
        self.strip_code_fences = strip_code_fences
        self._fallback = RecursiveChunker(chunk_size=chunk_size, overlap=overlap, min_chunk_size=0)

    def split_text(self, text: str) -> list[str]:
        """Split Markdown into heading-scoped chunks.

        Args:
            text: The Markdown source.

        Returns:
            Chunk strings, each prefixed with its heading path when
            :attr:`include_heading_path` is on.
        """
        if not text.strip():
            return []

        sections = self._sections(text)
        chunks: list[str] = []
        for path, raw in sections:
            body = raw.strip()
            if not body:
                continue

            prefix = ""
            if path and self.include_heading_path:
                prefix = f"{' > '.join(path)}\n\n"
                # The breadcrumb already names this section, so repeating its
                # own heading line in the body is pure duplication.
                body = _drop_leading_heading(body)
                if not body:
                    # A heading with no content of its own contributes to the
                    # path of the sections below it and nothing else.
                    continue

            whole = f"{prefix}{body}"
            if len(whole) <= self.chunk_size:
                chunks.append(whole)
            else:
                for piece in self._fallback.split_text(body):
                    chunks.append(f"{prefix}{piece}")
        return self._merge_undersized(chunks)

    def _sections(self, text: str) -> list[tuple[list[str], str]]:
        """Walk the document, tracking the current heading path.

        Returns:
            ``(heading_path, body)`` pairs in document order.
        """
        lines = text.split("\n")
        sections: list[tuple[list[str], str]] = []
        path: list[str] = []
        buffer: list[str] = []
        in_fence = False

        def flush() -> None:
            if buffer:
                sections.append((list(path), "\n".join(buffer)))
                buffer.clear()

        for line in lines:
            if self.strip_code_fences and _FENCE_RE.match(line.strip()):
                in_fence = not in_fence
                buffer.append(line)
                continue
            if in_fence:
                buffer.append(line)
                continue

            match = _HEADING_RE.match(line)
            if match and len(match.group("hashes")) <= self.max_heading_level:
                flush()
                level = len(match.group("hashes"))
                title = match.group("title").strip()
                path = [*path[: level - 1], title]
                buffer.append(line)
                continue
            buffer.append(line)

        flush()
        return sections


def _drop_leading_heading(body: str) -> str:
    r"""Remove a section's own heading line from the start of its body.

    The heading path already carries that text, so keeping the raw ``## Refunds``
    line as well wastes prompt tokens and gives the embedding a duplicated term.

    Args:
        body: The section body, starting with its heading line.

    Returns:
        The body without that line, stripped. An empty string when the section
        contained nothing but its heading.

    Example:
        >>> _drop_leading_heading("## Refunds\n\nWithin 30 days.")
        'Within 30 days.'
        >>> _drop_leading_heading("## Refunds")
        ''
    """
    lines = body.split("\n")
    if lines and _HEADING_RE.match(lines[0]):
        return "\n".join(lines[1:]).strip()
    return body


#: Split points per language, ordered from the strongest boundary downwards.
LANGUAGE_SEPARATORS: dict[str, tuple[str, ...]] = {
    "python": ("\nclass ", "\ndef ", "\n\tdef ", "\n    def ", "\n\n", "\n", " ", ""),
    "javascript": (
        "\nclass ",
        "\nfunction ",
        "\nconst ",
        "\nlet ",
        "\nexport ",
        "\n\n",
        "\n",
        " ",
        "",
    ),
    "typescript": (
        "\ninterface ",
        "\ntype ",
        "\nclass ",
        "\nfunction ",
        "\nconst ",
        "\nexport ",
        "\n\n",
        "\n",
        " ",
        "",
    ),
    "java": ("\nclass ", "\npublic ", "\nprivate ", "\nprotected ", "\n\n", "\n", " ", ""),
    "csharp": ("\nnamespace ", "\nclass ", "\npublic ", "\nprivate ", "\n\n", "\n", " ", ""),
    "go": ("\nfunc ", "\ntype ", "\nvar ", "\nconst ", "\n\n", "\n", " ", ""),
    "rust": ("\nimpl ", "\nfn ", "\npub fn ", "\nstruct ", "\nenum ", "\n\n", "\n", " ", ""),
    "cpp": ("\nclass ", "\nstruct ", "\nnamespace ", "\nvoid ", "\n\n", "\n", " ", ""),
    "c": ("\nstruct ", "\nvoid ", "\nint ", "\nstatic ", "\n\n", "\n", " ", ""),
    "ruby": ("\nclass ", "\nmodule ", "\ndef ", "\n\n", "\n", " ", ""),
    "php": ("\nclass ", "\nfunction ", "\npublic function ", "\n\n", "\n", " ", ""),
    "sql": ("\nCREATE ", "\nINSERT ", "\nSELECT ", "\nUPDATE ", ";\n", "\n\n", "\n", " ", ""),
    "html": ("\n<div", "\n<section", "\n<article", "\n<p", "\n\n", "\n", " ", ""),
    "generic": ("\n\n", "\n", " ", ""),
}

#: Maps a file extension onto a language key.
_EXTENSION_LANGUAGES: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".rb": "ruby",
    ".php": "php",
    ".sql": "sql",
    ".html": "html",
    ".htm": "html",
}


@register.chunker(
    "code",
    aliases=("source",),
    description="Splits source code at function and class boundaries.",
)
class CodeChunker(RecursiveChunker):
    r"""Language-aware source-code splitter.

    Args:
        language: One of the keys in :data:`LANGUAGE_SEPARATORS`, or
            ``"generic"``. Ignored when :meth:`for_path` picks one for you.
        chunk_size: Target chunk length in characters. Code is dense, so a
            smaller value than for prose usually works better.
        overlap: Characters repeated between chunks. Small values are right for
            code — duplicating half a function helps nobody.
        **config: Forwarded to :class:`RecursiveChunker`.

    Raises:
        ValueError: For an unknown language.

    Example:
        >>> code = "def a():\n    return 1\n\ndef b():\n    return 2\n"
        >>> chunks = CodeChunker(language="python", chunk_size=25).split_text(code)
        >>> len(chunks) >= 1
        True
    """

    provider_name = "code"

    def __init__(
        self,
        *,
        language: str = "generic",
        chunk_size: int = 800,
        overlap: int | None = None,
        **config: Any,
    ) -> None:
        key = language.lower()
        if key not in LANGUAGE_SEPARATORS:
            raise ValueError(
                f"Unknown language {language!r}. "
                f"Supported: {', '.join(sorted(LANGUAGE_SEPARATORS))}."
            )
        super().__init__(
            chunk_size=chunk_size,
            overlap=overlap,
            separators=LANGUAGE_SEPARATORS[key],
            **config,
        )
        self.language = key

    @classmethod
    def for_path(cls, path: str, **config: Any) -> CodeChunker:
        """Build a chunker configured for a file's language.

        Args:
            path: File path or name; only the extension is inspected.
            **config: Forwarded to the constructor.

        Returns:
            A chunker using that language's separators, falling back to
            ``generic`` for unknown extensions.

        Example:
            >>> CodeChunker.for_path("app/main.py").language
            'python'
            >>> CodeChunker.for_path("notes.xyz").language
            'generic'
        """
        from pathlib import Path

        suffix = Path(path).suffix.lower()
        return cls(language=_EXTENSION_LANGUAGES.get(suffix, "generic"), **config)
