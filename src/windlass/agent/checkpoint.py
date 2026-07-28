"""Checkpointing — durable agent state.

An agent run is a sequence of model calls and tool executions. Checkpointing
snapshots that state after every step, which buys three things:

* **Resumability.** A run interrupted for human approval, or by a crash, picks
  up where it stopped instead of starting over (and re-paying for every token
  already spent).
* **Multi-turn conversations.** A ``thread_id`` maps to accumulated state, so
  the next message continues the same run.
* **Time travel.** Every step is retained, so you can inspect exactly what the
  agent believed at step 3 when debugging why step 4 went wrong.

Two backends ship: in-memory (fast, per-process) and SQLite (durable, shared
between processes, stdlib-only).

Example:
    >>> saver = MemoryCheckpointer()
    >>> saver.put("thread-1", {"step": 1, "messages": []})
    >>> saver.get("thread-1")["step"]
    1
"""

from __future__ import annotations

import abc
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from windlass.core.exceptions import SerializationError
from windlass.core.logging import get_logger
from windlass.core.registry import register

__all__ = ["Checkpointer", "MemoryCheckpointer", "SQLiteCheckpointer"]

_log = get_logger(__name__)


class Checkpointer(abc.ABC):
    """Stores and retrieves agent state by thread.

    Implementations must be safe to use from multiple threads, since one agent
    instance typically serves many concurrent conversations.
    """

    @abc.abstractmethod
    def put(self, thread_id: str, state: dict[str, Any]) -> None:
        """Save a state snapshot.

        Args:
            thread_id: Conversation / run identifier.
            state: JSON-serialisable state.

        Raises:
            SerializationError: When the state cannot be stored.
        """

    @abc.abstractmethod
    def get(self, thread_id: str) -> dict[str, Any] | None:
        """Return the latest snapshot for a thread, or ``None``."""

    @abc.abstractmethod
    def history(self, thread_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent snapshots, newest first.

        Args:
            thread_id: Conversation identifier.
            limit: Maximum snapshots to return.

        Returns:
            Snapshots, newest first.
        """

    @abc.abstractmethod
    def delete(self, thread_id: str) -> None:
        """Discard every snapshot for a thread."""

    def threads(self) -> list[str]:
        """Return the known thread ids."""
        return []


@register.checkpointer(
    "memory",
    aliases=("inmemory", "default"),
    description="In-process checkpoint store (no dependencies).",
)
class MemoryCheckpointer(Checkpointer):
    """Keeps checkpoints in a dict.

    Fast and dependency-free, but per-process and lost on restart. Right for
    tests, notebooks and single-process services; wrong for anything that needs
    a resumed run to survive a deploy.

    Args:
        max_history: Snapshots retained per thread. Older ones are dropped.

    Example:
        >>> saver = MemoryCheckpointer()
        >>> saver.put("t", {"step": 1})
        >>> saver.put("t", {"step": 2})
        >>> saver.get("t")["step"], len(saver.history("t"))
        (2, 2)
    """

    def __init__(self, max_history: int = 50) -> None:
        self.max_history = max_history
        self._data: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.RLock()

    def put(self, thread_id: str, state: dict[str, Any]) -> None:
        """Append a snapshot, trimming the oldest beyond ``max_history``."""
        with self._lock:
            bucket = self._data.setdefault(thread_id, [])
            bucket.append({**state, "_saved_at": time.time()})
            if len(bucket) > self.max_history:
                del bucket[: len(bucket) - self.max_history]

    def get(self, thread_id: str) -> dict[str, Any] | None:
        """Return the newest snapshot, or ``None``."""
        with self._lock:
            bucket = self._data.get(thread_id)
            return dict(bucket[-1]) if bucket else None

    def history(self, thread_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent snapshots, newest first."""
        with self._lock:
            bucket = self._data.get(thread_id, [])
            return [dict(s) for s in reversed(bucket[-limit:])]

    def delete(self, thread_id: str) -> None:
        """Discard a thread's snapshots."""
        with self._lock:
            self._data.pop(thread_id, None)

    def threads(self) -> list[str]:
        """Return the known thread ids."""
        with self._lock:
            return sorted(self._data)


@register.checkpointer(
    "sqlite",
    aliases=("file", "durable"),
    description="Durable checkpoint store backed by SQLite (stdlib only).",
)
class SQLiteCheckpointer(Checkpointer):
    """Persists checkpoints to a SQLite database.

    Survives restarts and is shared between processes on one machine — enough
    for a single-node deployment, and it needs nothing beyond the standard
    library.

    Args:
        path: Database file. Parent directories are created. ``":memory:"``
            gives a transient database.
        max_history: Snapshots retained per thread.

    Raises:
        SerializationError: When the database cannot be opened.

    Note:
        Each call opens its own connection. That costs a little per write and
        buys thread safety without a connection pool — the right trade at the
        rate an agent checkpoints.

    Example:
        >>> import tempfile, pathlib
        >>> db = pathlib.Path(tempfile.mkdtemp()) / "state.db"
        >>> saver = SQLiteCheckpointer(db)
        >>> saver.put("t", {"step": 7})
        >>> SQLiteCheckpointer(db).get("t")["step"]
        7
    """

    def __init__(
        self, path: str | Path = ".windlass/checkpoints.db", max_history: int = 50
    ) -> None:
        self.path = Path(path) if str(path) != ":memory:" else Path(":memory:")
        self.max_history = max_history
        self._lock = threading.RLock()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._shared: sqlite3.Connection | None = (
            sqlite3.connect(":memory:", check_same_thread=False)
            if str(self.path) == ":memory:"
            else None
        )
        self._init()

    def _connect(self) -> sqlite3.Connection:
        """Open (or reuse) a connection."""
        if self._shared is not None:
            return self._shared
        try:
            connection = sqlite3.connect(str(self.path), timeout=10.0)
            connection.execute("PRAGMA journal_mode=WAL")
            return connection
        except sqlite3.Error as exc:
            raise SerializationError(
                f"Could not open the checkpoint database at {self.path}: {exc}",
                hint="Check the directory exists and is writable.",
            ) from exc

    def _init(self) -> None:
        """Create the schema if it does not exist."""
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        id         INTEGER PRIMARY KEY AUTOINCREMENT,
                        thread_id  TEXT NOT NULL,
                        state      TEXT NOT NULL,
                        saved_at   REAL NOT NULL
                    )
                    """)
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_thread ON checkpoints(thread_id, id DESC)"
                )
                connection.commit()
            finally:
                if self._shared is None:
                    connection.close()

    def put(self, thread_id: str, state: dict[str, Any]) -> None:
        """Insert a snapshot and prune old ones.

        Raises:
            SerializationError: When the state is not JSON-serialisable or the
                write fails.
        """
        try:
            payload = json.dumps(state, default=str)
        except (TypeError, ValueError) as exc:
            raise SerializationError(
                f"Agent state for thread {thread_id!r} is not JSON-serialisable: {exc}",
                hint="Keep custom objects out of agent state, or store an id instead.",
            ) from exc

        with self._lock:
            connection = self._connect()
            try:
                connection.execute(
                    "INSERT INTO checkpoints (thread_id, state, saved_at) VALUES (?, ?, ?)",
                    (thread_id, payload, time.time()),
                )
                connection.execute(
                    """
                    DELETE FROM checkpoints
                    WHERE thread_id = ?
                      AND id NOT IN (
                          SELECT id FROM checkpoints
                          WHERE thread_id = ? ORDER BY id DESC LIMIT ?
                      )
                    """,
                    (thread_id, thread_id, self.max_history),
                )
                connection.commit()
            except sqlite3.Error as exc:
                raise SerializationError(f"Could not save the checkpoint: {exc}") from exc
            finally:
                if self._shared is None:
                    connection.close()

    def get(self, thread_id: str) -> dict[str, Any] | None:
        """Return the newest snapshot, or ``None``."""
        rows = self.history(thread_id, limit=1)
        return rows[0] if rows else None

    def history(self, thread_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent snapshots, newest first."""
        with self._lock:
            connection = self._connect()
            try:
                cursor = connection.execute(
                    "SELECT state FROM checkpoints WHERE thread_id = ? ORDER BY id DESC LIMIT ?",
                    (thread_id, limit),
                )
                rows = cursor.fetchall()
            except sqlite3.Error as exc:
                _log.warning("Could not read checkpoints for %s: %s", thread_id, exc)
                return []
            finally:
                if self._shared is None:
                    connection.close()

        snapshots: list[dict[str, Any]] = []
        for (payload,) in rows:
            try:
                snapshots.append(json.loads(payload))
            except json.JSONDecodeError:  # pragma: no cover - corrupt row
                _log.warning("Skipping a corrupt checkpoint for thread %s.", thread_id)
        return snapshots

    def delete(self, thread_id: str) -> None:
        """Discard a thread's snapshots."""
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
                connection.commit()
            except sqlite3.Error as exc:  # pragma: no cover
                _log.warning("Could not delete checkpoints for %s: %s", thread_id, exc)
            finally:
                if self._shared is None:
                    connection.close()

    def threads(self) -> list[str]:
        """Return the known thread ids."""
        with self._lock:
            connection = self._connect()
            try:
                cursor = connection.execute("SELECT DISTINCT thread_id FROM checkpoints")
                return sorted(row[0] for row in cursor.fetchall())
            except sqlite3.Error:  # pragma: no cover
                return []
            finally:
                if self._shared is None:
                    connection.close()

    def close(self) -> None:
        """Close the shared in-memory connection, if there is one."""
        if self._shared is not None:
            self._shared.close()
            self._shared = None
