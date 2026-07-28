"""Retry with exponential backoff and jitter.

Provider calls fail transiently: rate limits, 5xx, socket resets. Windlass wraps
every outbound call in :func:`retry_async` so adapters do not each reinvent a
backoff loop, and so the policy is configurable in one place
(:class:`~windlass.core.config.RetryConfig`).

Only *transient* failures are retried. A 401 or a schema validation error is
raised immediately — retrying those just wastes a minute before failing anyway.
"""

from __future__ import annotations

import asyncio
import functools
import random
import time
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, TypeVar

from windlass.core.config import RetryConfig, settings
from windlass.core.exceptions import (
    AuthenticationError,
    ProviderTimeoutError,
    RateLimitError,
    ValidationError,
    WindlassError,
)
from windlass.core.logging import get_logger

__all__ = ["backoff_delays", "is_retryable", "retry_async", "retry_sync", "with_retry"]

_T = TypeVar("_T")
_log = get_logger(__name__)

#: Errors that are never worth retrying — the next attempt would fail identically.
NON_RETRYABLE: tuple[type[BaseException], ...] = (
    AuthenticationError,
    ValidationError,
    KeyboardInterrupt,
    SystemExit,
    asyncio.CancelledError,
)

#: Errors that are always worth retrying.
RETRYABLE: tuple[type[BaseException], ...] = (
    RateLimitError,
    ProviderTimeoutError,
    ConnectionError,
    TimeoutError,
    OSError,
)


def is_retryable(exc: BaseException, extra: Iterable[type[BaseException]] = ()) -> bool:
    """Decide whether ``exc`` is worth another attempt.

    The decision order matters: an explicit non-retryable class wins over a
    retryable base class, so :class:`AuthenticationError` never retries even
    though it subclasses ``ProviderError``.

    Args:
        exc: The raised exception.
        extra: Additional exception types to treat as retryable — adapters pass
            their SDK's transient error classes here.

    Returns:
        True when a retry is warranted.

    Example:
        >>> is_retryable(RateLimitError("slow down"))
        True
        >>> is_retryable(AuthenticationError("bad key"))
        False
    """
    if isinstance(exc, NON_RETRYABLE):
        return False
    if isinstance(exc, RETRYABLE) or (tuple(extra) and isinstance(exc, tuple(extra))):
        return True
    # Duck-typed HTTP status: httpx / SDK errors expose .status_code or .response.
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status == 408 or status == 429 or 500 <= status < 600
    return False


def backoff_delays(config: RetryConfig | None = None) -> list[float]:
    """Return the delay schedule a retry loop would use.

    Useful for tests and for logging the policy at startup.

    Args:
        config: Policy to expand. Defaults to the configured global policy.

    Returns:
        One delay per retry (so ``attempts - 1`` entries), jitter excluded.

    Example:
        >>> backoff_delays(RetryConfig(attempts=4, initial_delay=1, multiplier=2))
        [1.0, 2.0, 4.0]
    """
    cfg = config or settings().retry
    delays: list[float] = []
    delay = cfg.initial_delay
    for _ in range(max(0, cfg.attempts - 1)):
        delays.append(min(delay, cfg.max_delay))
        delay *= cfg.multiplier
    return delays


def _next_delay(attempt: int, cfg: RetryConfig, exc: BaseException) -> float:
    """Compute the sleep before ``attempt`` (1-based), honouring Retry-After."""
    advertised = getattr(exc, "retry_after", None)
    if isinstance(advertised, int | float) and advertised > 0:
        base = min(float(advertised), cfg.max_delay)
    else:
        base = min(cfg.initial_delay * (cfg.multiplier ** (attempt - 1)), cfg.max_delay)
    if cfg.jitter:
        base += base * cfg.jitter * random.random()
    return base


async def retry_async(
    fn: Callable[[], Awaitable[_T]],
    *,
    config: RetryConfig | None = None,
    retry_on: Iterable[type[BaseException]] = (),
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    description: str = "provider call",
) -> _T:
    """Await ``fn`` with retries on transient failures.

    Args:
        fn: A zero-argument callable returning an awaitable. It is re-invoked on
            each attempt, so pass a lambda rather than a coroutine object —
            coroutines cannot be awaited twice.
        config: Retry policy. Defaults to the global configuration.
        retry_on: Extra exception types to treat as transient.
        on_retry: Callback invoked as ``(attempt, exception, delay)`` before each
            sleep. Handy for metrics.
        description: Label used in log messages.

    Returns:
        Whatever ``fn`` eventually returns.

    Raises:
        Exception: The last failure, once attempts are exhausted or the error is
            classified as non-retryable.

    Example:
        >>> import asyncio
        >>> calls = []
        >>> async def flaky():
        ...     calls.append(1)
        ...     if len(calls) < 2:
        ...         raise ConnectionError("reset")
        ...     return "ok"
        >>> asyncio.run(retry_async(flaky, config=RetryConfig(attempts=3, initial_delay=0)))
        'ok'
    """
    cfg = config or settings().retry
    last: BaseException | None = None

    for attempt in range(1, cfg.attempts + 1):
        try:
            return await fn()
        except BaseException as exc:
            last = exc
            if attempt >= cfg.attempts or not is_retryable(exc, retry_on):
                raise
            delay = _next_delay(attempt, cfg, exc)
            _log.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.2fs",
                description,
                attempt,
                cfg.attempts,
                exc,
                delay,
            )
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            await asyncio.sleep(delay)

    raise last if last is not None else WindlassError(f"{description} failed without an exception")


def retry_sync(
    fn: Callable[[], _T],
    *,
    config: RetryConfig | None = None,
    retry_on: Iterable[type[BaseException]] = (),
    description: str = "provider call",
) -> _T:
    """Blocking counterpart of :func:`retry_async`.

    Args:
        fn: Zero-argument callable, re-invoked per attempt.
        config: Retry policy. Defaults to the global configuration.
        retry_on: Extra transient exception types.
        description: Label used in log messages.

    Returns:
        Whatever ``fn`` eventually returns.

    Raises:
        Exception: The last failure once attempts are exhausted.

    Example:
        >>> retry_sync(lambda: 7, config=RetryConfig(attempts=1))
        7
    """
    cfg = config or settings().retry
    last: BaseException | None = None

    for attempt in range(1, cfg.attempts + 1):
        try:
            return fn()
        except BaseException as exc:
            last = exc
            if attempt >= cfg.attempts or not is_retryable(exc, retry_on):
                raise
            delay = _next_delay(attempt, cfg, exc)
            _log.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.2fs",
                description,
                attempt,
                cfg.attempts,
                exc,
                delay,
            )
            time.sleep(delay)

    raise last if last is not None else WindlassError(f"{description} failed without an exception")


def with_retry(
    *,
    config: RetryConfig | None = None,
    retry_on: Iterable[type[BaseException]] = (),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator applying the retry policy to a sync or async function.

    Args:
        config: Retry policy. Defaults to the global configuration.
        retry_on: Extra transient exception types.

    Returns:
        A decorator preserving the wrapped function's signature and sync/async
        nature.

    Example:
        >>> @with_retry(config=RetryConfig(attempts=2, initial_delay=0))
        ... def ping() -> str:
        ...     return "pong"
        >>> ping()
        'pong'
    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        label = getattr(fn, "__qualname__", repr(fn))

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await retry_async(
                    lambda: fn(*args, **kwargs),
                    config=config,
                    retry_on=retry_on,
                    description=label,
                )

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return retry_sync(
                lambda: fn(*args, **kwargs),
                config=config,
                retry_on=retry_on,
                description=label,
            )

        return sync_wrapper

    return decorate
