"""HTML, web-page and YouTube loaders.

The HTML loader works with no dependencies (falling back to a regex-based tag
stripper) but does a considerably better job with BeautifulSoup installed. The
web loader adds HTTP fetching, and the YouTube loader pulls transcripts.

Install the optional pieces with::

    pip install "windlass[loaders]"

Example:
    >>> HTMLLoader().load(b"<html><body><p>Hi</p></body></html>")[0].content
    'Hi'
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

import httpx

from windlass.core.concurrency import gather_bounded, to_thread
from windlass.core.config import settings
from windlass.core.exceptions import IngestionError
from windlass.core.lazy import is_available, require
from windlass.core.registry import register
from windlass.core.text import normalize_whitespace, strip_html
from windlass.core.types import Document
from windlass.interfaces.loader import Loader, SourceLike

__all__ = ["HTMLLoader", "WebLoader", "YouTubeLoader"]

#: Elements whose text is never useful as document content.
_NOISE_TAGS = ("script", "style", "nav", "footer", "header", "aside", "noscript", "form")

#: Elements that usually hold the article body, tried in order.
_CONTENT_SELECTORS = ("article", "main", "[role=main]", "#content", ".content", ".post", "body")

_YOUTUBE_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?v=|embed/|shorts/|live/)|youtu\.be/)([A-Za-z0-9_-]{11})"
)


@register.loader(
    "html",
    aliases=("htm",),
    description="HTML files, with boilerplate stripped.",
)
class HTMLLoader(Loader):
    """Extracts readable text from HTML.

    Args:
        selector: CSS selector for the content region. When omitted the loader
            tries :data:`_CONTENT_SELECTORS` in order.
        remove_tags: Elements to strip entirely before extraction.
        keep_links: Render links as ``text (url)`` so the model can cite them.
        **config: Forwarded to :class:`~windlass.interfaces.loader.Loader`.

    Note:
        Works without ``beautifulsoup4`` using a regex stripper, which is fine
        for well-formed markup. Install ``windlass[loaders]`` for real
        parsing, boilerplate removal and CSS selectors.
    """

    provider_name = "html"
    extensions: tuple[str, ...] = (".html", ".htm", ".xhtml")
    mimetypes = ("text/html",)

    def __init__(
        self,
        *,
        selector: str | None = None,
        remove_tags: tuple[str, ...] = _NOISE_TAGS,
        keep_links: bool = False,
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self.selector = selector
        self.remove_tags = remove_tags
        self.keep_links = keep_links

    async def aload_source(self, source: SourceLike) -> list[Document]:
        """Extract text from one HTML document.

        Args:
            source: A path or a bytes payload.

        Returns:
            A single-element list, empty when no text could be extracted.
        """
        raw = self._read_bytes(source)
        html = raw.decode("utf-8", errors="replace")
        path = str(source) if not isinstance(source, bytes) else None
        base = self._base_metadata(source) if not isinstance(source, bytes) else {}
        text, meta = await to_thread(self.extract, html)
        if not text.strip():
            return []
        return [
            Document(
                content=text,
                metadata={**base, **meta},
                source=path,
                mimetype="text/html",
            )
        ]

    def extract(self, html: str) -> tuple[str, dict[str, Any]]:
        """Turn HTML into text plus metadata.

        Args:
            html: The HTML source.

        Returns:
            A ``(text, metadata)`` pair. Metadata may include ``title``,
            ``description``, ``headings`` and ``links``.

        Example:
            >>> text, meta = HTMLLoader().extract("<title>T</title><p>Body</p>")
            >>> meta["title"], text
            ('T', 'Body')
        """
        if not is_available("bs4"):
            title = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
            fallback: dict[str, Any] = (
                {"title": normalize_whitespace(title.group(1))} if title else {}
            )
            return strip_html(html), fallback

        bs4 = require("bs4", extra="loaders", feature="The HTML loader")
        parser = "lxml" if is_available("lxml") else "html.parser"
        soup = bs4.BeautifulSoup(html, parser)

        meta: dict[str, Any] = {}
        if soup.title and soup.title.string:
            meta["title"] = normalize_whitespace(str(soup.title.string))
        description = soup.find("meta", attrs={"name": "description"})
        if description and description.get("content"):
            meta["description"] = normalize_whitespace(str(description["content"]))

        for tag in soup(list(self.remove_tags)):
            tag.decompose()

        root = None
        selectors = [self.selector] if self.selector else list(_CONTENT_SELECTORS)
        for candidate in selectors:
            if not candidate:
                continue
            found = soup.select_one(candidate)
            if found is not None and found.get_text(strip=True):
                root = found
                break
        root = root or soup

        headings = [
            normalize_whitespace(h.get_text())
            for h in root.find_all(["h1", "h2", "h3"])
            if h.get_text(strip=True)
        ]
        if headings:
            meta["headings"] = headings[:50]

        if self.keep_links:
            links = []
            for anchor in root.find_all("a", href=True):
                label = normalize_whitespace(anchor.get_text())
                if label:
                    anchor.replace_with(f"{label} ({anchor['href']})")
                    links.append(anchor["href"])
            if links:
                meta["links"] = links[:100]

        return normalize_whitespace(root.get_text(separator="\n")), meta


@register.loader(
    "web",
    aliases=("url", "http"),
    description="Fetches and extracts text from web pages.",
)
class WebLoader(HTMLLoader):
    """Fetches URLs over HTTP and extracts their text.

    Args:
        headers: Extra request headers. A browser-like ``User-Agent`` is sent by
            default, because many sites reject the Python default.
        follow_redirects: Follow 3xx responses.
        timeout: Per-request timeout in seconds.
        max_bytes: Refuse responses larger than this, so one enormous page
            cannot exhaust memory.
        **config: Forwarded to :class:`HTMLLoader`.

    Raises:
        IngestionError: On a network failure or a non-2xx response.

    Performance:
        URLs are fetched concurrently by the base class, bounded by the global
        ``max_concurrency`` setting. Be considerate of the sites you crawl.
    """

    provider_name = "web"
    handles_urls = True
    extensions: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        headers: dict[str, str] | None = None,
        follow_redirects: bool = True,
        timeout: float | None = None,
        max_bytes: int = 10_000_000,
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; WindlassBot/0.1; +https://github.com/windlass)",
            "Accept": "text/html,application/xhtml+xml",
            **(headers or {}),
        }
        self.follow_redirects = follow_redirects
        self.timeout = timeout or settings().request_timeout
        self.max_bytes = max_bytes

    @classmethod
    def can_handle(cls, source: SourceLike) -> bool:
        """Return whether ``source`` looks like an HTTP(S) URL."""
        return isinstance(source, str) and source.startswith(("http://", "https://"))

    async def aload_source(self, source: SourceLike) -> list[Document]:
        """Fetch a URL and extract its text.

        Args:
            source: The URL to fetch.

        Returns:
            A single-element list, empty when the page had no extractable text.

        Raises:
            IngestionError: On a network error or an HTTP error status.
        """
        url = str(source)
        try:
            async with httpx.AsyncClient(
                headers=self.headers,
                follow_redirects=self.follow_redirects,
                timeout=self.timeout,
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                if len(response.content) > self.max_bytes:
                    raise IngestionError(
                        f"{url} returned {len(response.content)} bytes, "
                        f"over the {self.max_bytes} limit."
                    )
                html = response.text
        except httpx.HTTPStatusError as exc:
            raise IngestionError(
                f"{url} returned HTTP {exc.response.status_code}.",
                context={"url": url, "status": exc.response.status_code},
            ) from exc
        except httpx.HTTPError as exc:
            raise IngestionError(f"Could not fetch {url}: {exc}", context={"url": url}) from exc

        text, meta = await to_thread(self.extract, html)
        if not text.strip():
            return []
        parsed = urlparse(url)
        return [
            Document(
                content=text,
                metadata={**meta, "url": url, "domain": parsed.netloc, "source": url},
                source=url,
                mimetype="text/html",
            )
        ]

    async def acrawl(self, urls: list[str], *, concurrency: int | None = None) -> list[Document]:
        """Fetch several URLs concurrently.

        Args:
            urls: The URLs to fetch.
            concurrency: Maximum simultaneous requests.

        Returns:
            Every document that loaded successfully; failures are logged and
            skipped.
        """
        limit = concurrency or settings().max_concurrency
        results = await gather_bounded(
            [self.aload_source(u) for u in urls], limit=limit, return_exceptions=True
        )
        documents: list[Document] = []
        for url, result in zip(urls, results, strict=True):
            if isinstance(result, BaseException):
                self._log.warning("Skipping %s: %s", url, result)
                continue
            documents.extend(result)
        return documents


@register.loader(
    "youtube",
    aliases=("yt",),
    description="YouTube video transcripts.",
)
class YouTubeLoader(Loader):
    """Loads transcripts for YouTube videos.

    Args:
        languages: Preferred transcript languages, in order of preference.
        chunk_by_time: Emit one document per ``segment_seconds`` window, so
            citations carry a timestamp you can link to.
        segment_seconds: Window length when ``chunk_by_time`` is on.
        **config: Forwarded to :class:`~windlass.interfaces.loader.Loader`.

    Raises:
        MissingDependencyError: When ``youtube-transcript-api`` is not installed.
        IngestionError: When the video has no accessible transcript.

    Note:
        Only videos with captions (auto-generated or human) can be loaded. There
        is no audio transcription here — use the ``audio`` loader for that.
    """

    provider_name = "youtube"
    handles_urls = True
    extensions: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        languages: tuple[str, ...] = ("en",),
        chunk_by_time: bool = False,
        segment_seconds: int = 300,
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self.languages = tuple(languages)
        self.chunk_by_time = chunk_by_time
        self.segment_seconds = segment_seconds

    @classmethod
    def can_handle(cls, source: SourceLike) -> bool:
        """Return whether ``source`` is a YouTube URL or a bare video id."""
        if not isinstance(source, str):
            return False
        return bool(_YOUTUBE_ID_RE.search(source)) or (
            len(source) == 11 and source.replace("-", "").replace("_", "").isalnum()
        )

    async def aload_source(self, source: SourceLike) -> list[Document]:
        """Fetch one video's transcript.

        Args:
            source: A YouTube URL or an 11-character video id.

        Returns:
            One document per time segment, or one for the whole transcript.

        Raises:
            IngestionError: When no transcript is available.
        """
        api = require("youtube_transcript_api", extra="loaders", feature="The YouTube loader")
        video_id = extract_video_id(str(source))

        def _fetch() -> list[dict[str, Any]]:
            try:
                return api.YouTubeTranscriptApi.get_transcript(
                    video_id, languages=list(self.languages)
                )
            except Exception as exc:
                raise IngestionError(
                    f"No transcript available for video {video_id}: {exc}",
                    hint="The video may have captions disabled, be private, or not "
                    f"offer any of {self.languages}.",
                    context={"video_id": video_id},
                ) from exc

        entries = await to_thread(_fetch)
        url = f"https://www.youtube.com/watch?v={video_id}"
        base = {"video_id": video_id, "url": url, "source": url}

        if not self.chunk_by_time:
            text = " ".join(e["text"].strip() for e in entries if e.get("text"))
            duration = entries[-1]["start"] + entries[-1].get("duration", 0) if entries else 0
            return [
                Document(
                    content=normalize_whitespace(text),
                    metadata={**base, "duration_seconds": round(duration)},
                    source=url,
                    mimetype="text/plain",
                )
            ]

        documents: list[Document] = []
        buffer: list[str] = []
        window_start = 0.0
        for entry in entries:
            start = float(entry.get("start", 0.0))
            if buffer and start - window_start >= self.segment_seconds:
                documents.append(self._segment(buffer, window_start, base, url))
                buffer, window_start = [], start
            buffer.append(entry.get("text", "").strip())
        if buffer:
            documents.append(self._segment(buffer, window_start, base, url))
        return documents

    @staticmethod
    def _segment(parts: list[str], start: float, base: dict[str, Any], url: str) -> Document:
        """Build one time-windowed transcript document."""
        stamp = f"{int(start) // 60:02d}:{int(start) % 60:02d}"
        return Document(
            content=normalize_whitespace(" ".join(p for p in parts if p)),
            metadata={
                **base,
                "start_seconds": round(start),
                "timestamp": stamp,
                "url": f"{url}&t={int(start)}s",
            },
            source=url,
            mimetype="text/plain",
        )


def extract_video_id(source: str) -> str:
    """Pull the 11-character video id out of a YouTube URL.

    Args:
        source: A YouTube URL in any of its common shapes, or a bare id.

    Returns:
        The video id.

    Raises:
        IngestionError: When no id can be found.

    Example:
        >>> extract_video_id("https://youtu.be/dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
        >>> extract_video_id("dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
    """
    match = _YOUTUBE_ID_RE.search(source)
    if match:
        return match.group(1)
    if len(source) == 11:
        return source
    raise IngestionError(
        f"Could not find a YouTube video id in {source!r}.",
        hint="Pass a watch/embed/short URL or the bare 11-character id.",
    )
