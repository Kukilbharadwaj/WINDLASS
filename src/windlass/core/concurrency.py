"""Async/sync bridging, bounded parallelism and batching helpers.

Every Windlass interface is **async-first**: adapters implement ``a*`` coroutines
and the base classes derive the blocking variants from them. That gives one
implementation, two APIs, and no risk of the two drifting apart.

The tricky part is calling async code from sync code *safely*. ``asyncio.run``
explodes inside a running event loop (Jupyter, FastAPI, LangServe), so
:func:`run_sync` detects that case and hands the coroutine to a dedicated
background loop thread instead.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future
from typing import Any, TypeVar

__all__ = [
    "batched",
    "gather_bounded",
    "iter_sync",
    "map_async",
    "run_sync",
    "shutdown_background_loop",
    "to_thread",
]

_T = TypeVar("_T")
_R = TypeVar("_R")


class _BackgroundLoop:
    """A lazily started event loop running on its own daemon thread.

    Every blocking call in the process runs here. Started on first use, reused
    forever, torn down at interpreter exit. Sharing one loop is what allows a
    provider to hold a long-lived connection pool across blocking calls.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def is_current_thread(self) -> bool:
        """Return whether the caller is already running on the background loop."""
        return self._thread is not None and self._thread is threading.current_thread()

    def loop(self) -> asyncio.AbstractEventLoop:
        """Return the running background loop, starting it if necessary."""
        if self._loop is not None and not self._loop.is_closed():
            return self._loop
        with self._lock:
            if self._loop is not None and not self._loop.is_closed():
                return self._loop
            ready = threading.Event()
            loop = asyncio.new_event_loop()

            def _run() -> None:
                asyncio.set_event_loop(loop)
                ready.set()
                loop.run_forever()

            thread = threading.Thread(target=_run, name="windlass-bg-loop", daemon=True)
            thread.start()
            ready.wait(timeout=10)
            self._loop, self._thread = loop, thread
            return loop

    def submit(self, coro: Awaitable[_T]) -> Future[_T]:
        """Schedule ``coro`` on the background loop and return its future."""
        loop = self.loop()
        return asyncio.run_coroutine_threadsafe(_ensure_coro(coro), loop)

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop the background loop and join its thread."""
        with self._lock:
            loop, thread = self._loop, self._thread
            self._loop = self._thread = None
        if loop is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=timeout)
        with contextlib.suppress(RuntimeError):
            loop.close()


async def _ensure_coro(awaitable: Awaitable[_T]) -> _T:
    return await awaitable


_BACKGROUND = _BackgroundLoop()
atexit.register(_BACKGROUND.shutdown)


def shutdown_background_loop() -> None:
    """Tear down the background event loop used by :func:`run_sync`.

    Called automatically at interpreter exit. Exposed for tests and for hosts
    that want deterministic cleanup.
    """
    _BACKGROUND.shutdown()


def run_sync(awaitable: Awaitable[_T], *, timeout: float | None = None) -> _T:
    """Run an awaitable to completion from synchronous code.

    Every call is dispatched to one long-lived background event loop, whether or
    not the caller is already inside a loop. That is a correctness requirement,
    not an optimisation — the same reasoning as :func:`iter_sync`.

    :func:`asyncio.run` creates a loop and closes it again on every call. Any
    resource bound to that loop dies with it, and the most common such resource
    is a connection pool: every provider that keeps an ``httpx.AsyncClient``
    alive between calls (Ollama, and every vendor SDK underneath OpenAI,
    Anthropic, Groq and Gemini) pools keep-alive sockets on the loop that opened
    them. With a fresh loop per call, the *first* blocking call succeeds and the
    second raises ``RuntimeError: Event loop is closed`` from deep inside httpx
    — a failure that mock transports never reproduce, because they open no
    sockets.

    Sharing one loop also removes the per-call cost of standing a loop up and
    tearing it down, which dominates short calls.

    Args:
        awaitable: The coroutine or future to run.
        timeout: Optional ceiling in seconds.

    Returns:
        The awaited result.

    Raises:
        TimeoutError: If ``timeout`` elapses first.
        RuntimeError: If called from inside the background loop itself, where
            blocking on it could never return.
        Exception: Anything the awaitable raises, re-raised unchanged.

    Note:
        Calling this from inside a coroutine works but blocks the calling
        thread. Prefer ``await`` on the ``a*`` method when you are already async.

    Example:
        >>> async def work() -> int:
        ...     return 42
        >>> run_sync(work())
        42

        Repeated calls share a loop, so a pooled client survives between them:

        >>> from windlass.providers.llm.fake import FakeLLM
        >>> llm = FakeLLM(responses=["one", "two"])
        >>> llm.complete("a").content, llm.complete("b").content
        ('one', 'two')
    """
    if _BACKGROUND.is_current_thread():
        raise RuntimeError(
            "run_sync() was called from inside the background event loop, where "
            "blocking on that same loop can never return. Await the a* coroutine "
            "directly instead."
        )
    return _BACKGROUND.submit(awaitable).result(timeout)


def iter_sync(agen: AsyncIterator[_T], *, timeout: float | None = None) -> Iterator[_T]:
    """Consume an async iterator from synchronous code.

    Pulls one item at a time, so streaming stays streaming — nothing is buffered
    ahead of the consumer.

    Every ``__anext__`` is dispatched to the *same* background event loop. That
    is not an optimisation, it is a correctness requirement: :func:`asyncio.run`
    calls ``loop.shutdown_asyncgens()`` on exit, which finalises any suspended
    async generator. Driving one with repeated ``asyncio.run`` calls therefore
    yields exactly one item and then silently stops.

    Args:
        agen: The async iterator to drain.
        timeout: Optional per-item ceiling in seconds.

    Yields:
        Each item produced by ``agen``.

    Raises:
        TimeoutError: If ``timeout`` elapses while waiting for an item.
        Exception: Anything the iterator raises, re-raised unchanged.

    Example:
        >>> async def gen():
        ...     for i in range(3):
        ...         yield i
        >>> list(iter_sync(gen()))
        [0, 1, 2]
    """
    sentinel = object()

    async def _next() -> Any:
        try:
            return await agen.__anext__()
        except StopAsyncIteration:
            return sentinel

    try:
        while True:
            item = _BACKGROUND.submit(_next()).result(timeout)
            if item is sentinel:
                return
            yield item
    finally:
        aclose = getattr(agen, "aclose", None)
        if aclose is not None:
            with contextlib.suppress(Exception):
                _BACKGROUND.submit(aclose()).result(timeout)


async def gather_bounded(
    awaitables: Sequence[Awaitable[_T]],
    *,
    limit: int = 8,
    return_exceptions: bool = False,
) -> list[_T]:
    """``asyncio.gather`` with a concurrency ceiling.

    Unbounded ``gather`` over a few thousand embedding calls will happily open a
    few thousand sockets and get you rate limited. This keeps at most ``limit``
    in flight while preserving input order in the results.

    Args:
        awaitables: The coroutines to run.
        limit: Maximum number in flight at once. ``<= 0`` means unbounded.
        return_exceptions: When True, failures land in the result list instead
            of propagating.

    Returns:
        Results in the same order as ``awaitables``.

    Example:
        >>> import asyncio
        >>> async def double(x): return x * 2
        >>> asyncio.run(gather_bounded([double(i) for i in range(4)], limit=2))
        [0, 2, 4, 6]
    """
    if not awaitables:
        return []
    if limit <= 0 or limit >= len(awaitables):
        return await asyncio.gather(*awaitables, return_exceptions=return_exceptions)  # type: ignore[return-value]

    semaphore = asyncio.Semaphore(limit)

    async def _guarded(aw: Awaitable[_T]) -> _T:
        async with semaphore:
            return await aw

    return await asyncio.gather(  # type: ignore[return-value]
        *(_guarded(aw) for aw in awaitables), return_exceptions=return_exceptions
    )


async def map_async(
    fn: Callable[[_T], Awaitable[_R]],
    items: Sequence[_T],
    *,
    limit: int = 8,
    return_exceptions: bool = False,
) -> list[_R]:
    """Apply an async function across ``items`` with bounded concurrency.

    Args:
        fn: Async callable applied to each item.
        items: Inputs.
        limit: Maximum concurrent invocations.
        return_exceptions: Collect failures instead of raising.

    Returns:
        Results in input order.

    Example:
        >>> import asyncio
        >>> async def square(x): return x * x
        >>> asyncio.run(map_async(square, [1, 2, 3], limit=2))
        [1, 4, 9]
    """
    return await gather_bounded(
        [fn(item) for item in items], limit=limit, return_exceptions=return_exceptions
    )


def batched(items: Iterable[_T], size: int) -> Iterator[list[_T]]:
    """Split an iterable into lists of at most ``size`` elements.

    Args:
        items: Any iterable; consumed lazily.
        size: Maximum batch size. Must be positive.

    Yields:
        Successive batches. The final batch may be shorter.

    Raises:
        ValueError: If ``size`` is not positive.

    Example:
        >>> list(batched(range(5), 2))
        [[0, 1], [2, 3], [4]]
    """
    if size <= 0:
        raise ValueError("batch size must be positive")
    batch: list[_T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


async def to_thread(fn: Callable[..., _R], /, *args: Any, **kwargs: Any) -> _R:
    """Run a blocking callable on the default thread pool.

    Adapters around synchronous SDKs (FAISS, sentence-transformers, pypdf) use
    this so they never block the event loop.

    Args:
        fn: The blocking callable.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Whatever ``fn`` returns.

    Example:
        >>> import asyncio
        >>> asyncio.run(to_thread(sum, [1, 2, 3]))
        6
    """
    return await asyncio.to_thread(fn, *args, **kwargs)
