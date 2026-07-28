"""Logging for Windlass.

Library rule number one: **do not configure the root logger.** Windlass attaches
a :class:`logging.NullHandler` to its own logger and otherwise stays out of the
way, so an application's logging setup wins. :func:`configure_logging` is opt-in
and exists purely for scripts and for ``WINDLASS_LOG_LEVEL=DEBUG`` debugging.

Every logger is namespaced under ``windlass.*``, so one line silences the whole
framework::

    logging.getLogger("windlass").setLevel(logging.WARNING)
"""

from __future__ import annotations

import contextvars
import logging
import os
import sys
import time
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from typing import Any

__all__ = [
    "bind_context",
    "configure_logging",
    "current_context",
    "get_logger",
    "log_duration",
]

ROOT_LOGGER_NAME = "windlass"

#: Correlation fields merged into every log record emitted inside the context.
#: The default is ``None`` rather than ``{}``: a mutable default on a ContextVar
#: is shared by every context that never sets it, so one stray mutation would
#: leak correlation fields across unrelated requests.
_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "windlass_log_context", default=None
)

_root = logging.getLogger(ROOT_LOGGER_NAME)
_root.addHandler(logging.NullHandler())


class _ContextFilter(logging.Filter):
    """Copies the current correlation context onto each record."""

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _CONTEXT.get()
        if ctx:
            record.windlass_context = ctx  # type: ignore[attr-defined]
            for key, value in ctx.items():
                if not hasattr(record, key):
                    setattr(record, key, value)
        return True


class _ContextAdapter(logging.LoggerAdapter):
    """Appends correlation fields to the message when any are bound."""

    def process(
        self, msg: Any, kwargs: MutableMapping[str, Any]
    ) -> tuple[Any, MutableMapping[str, Any]]:
        ctx = _CONTEXT.get()
        if ctx:
            suffix = " ".join(f"{k}={v}" for k, v in ctx.items())
            msg = f"{msg} [{suffix}]"
        return msg, kwargs


def get_logger(name: str) -> logging.Logger:
    """Return a Windlass-namespaced logger.

    Args:
        name: Usually ``__name__``. A leading ``windlass.`` is not duplicated;
            anything else is nested under ``windlass.``.

    Returns:
        A configured :class:`logging.Logger`.

    Example:
        >>> get_logger("windlass.core.registry").name
        'windlass.core.registry'
        >>> get_logger("my_plugin").name
        'windlass.my_plugin'
    """
    if name == ROOT_LOGGER_NAME or name.startswith(f"{ROOT_LOGGER_NAME}."):
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
    if not any(isinstance(f, _ContextFilter) for f in logger.filters):
        logger.addFilter(_ContextFilter())
    return logger


def configure_logging(
    level: int | str = "INFO",
    *,
    stream: Any = None,
    fmt: str | None = None,
    force: bool = False,
) -> logging.Logger:
    """Attach a human-readable stream handler to the ``windlass`` logger.

    Intended for scripts, notebooks and the CLI. Applications with their own
    logging configuration should not call this.

    Args:
        level: Log level name or numeric value.
        stream: Target stream. Defaults to ``sys.stderr``.
        fmt: Custom :mod:`logging` format string.
        force: Replace handlers that a previous call installed.

    Returns:
        The configured ``windlass`` logger.

    Example:
        >>> logger = configure_logging("WARNING")
        >>> logger.name
        'windlass'
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level if isinstance(level, int) else level.upper())

    existing = [h for h in logger.handlers if getattr(h, "_windlass_handler", False)]
    if existing and not force:
        return logger
    for handler in existing:
        logger.removeHandler(handler)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(
        logging.Formatter(fmt or "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    )
    handler.addFilter(_ContextFilter())
    handler._windlass_handler = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.propagate = False
    return logger


@contextmanager
def bind_context(**fields: Any) -> Iterator[dict[str, Any]]:
    """Bind correlation fields for the duration of the block.

    Fields nest, are async-safe (backed by :mod:`contextvars`) and appear on
    every Windlass log record emitted inside the block.

    Args:
        **fields: Key/value pairs such as ``request_id`` or ``thread_id``.

    Yields:
        The merged context dictionary.

    Example:
        >>> with bind_context(request_id="abc"):
        ...     current_context()["request_id"]
        'abc'
    """
    token = _CONTEXT.set({**(_CONTEXT.get() or {}), **fields})
    try:
        yield _CONTEXT.get() or {}
    finally:
        _CONTEXT.reset(token)


def current_context() -> dict[str, Any]:
    """Return a copy of the currently bound correlation fields."""
    return dict(_CONTEXT.get() or {})


@contextmanager
def log_duration(
    logger: logging.Logger, message: str, level: int = logging.DEBUG
) -> Iterator[None]:
    """Log how long a block took.

    Args:
        logger: Where to emit the message.
        message: Description of the work, e.g. ``"embed 512 chunks"``.
        level: Level for the completion message.

    Yields:
        None.

    Example:
        >>> import logging
        >>> with log_duration(get_logger("windlass.demo"), "work"):
        ...     pass
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.log(level, "%s took %.1fms", message, elapsed_ms)


# Honour WINDLASS_LOG_LEVEL for zero-friction debugging of a misbehaving install.
_env_level = os.getenv("WINDLASS_LOG_LEVEL")
if _env_level:  # pragma: no cover - environment dependent
    configure_logging(_env_level)
