"""Small persisted runtime state so a restart never re-trades a candle."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class RuntimeState:
    """Everything that must survive a daemon restart."""

    processed_candle: str | None = None
    pending_candle: str | None = None
    signal_attempts: int = 0
    day: str | None = None
    trades_today: int = 0
    last_deal_ticket: int = 0

    def roll_day(self, today: str) -> bool:
        """Reset the daily counters when the UTC date changes."""
        if self.day == today:
            return False
        self.day = today
        self.trades_today = 0
        return True

    def record_trade(self) -> None:
        self.trades_today += 1

    def begin_candle(self, candle: str) -> None:
        """Start (or resume) work on a candle and expose its attempt count."""
        if self.pending_candle != candle:
            self.pending_candle = candle
            self.signal_attempts = 0

    def finish_candle(self, candle: str) -> None:
        """Mark a candle as fully handled so it is never processed twice."""
        self.processed_candle = candle
        self.pending_candle = None
        self.signal_attempts = 0


class StateStore:
    """Read and write :class:`RuntimeState` as JSON."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> RuntimeState:
        if not self.path.exists():
            return RuntimeState()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return RuntimeState()
        if not isinstance(raw, dict):
            return RuntimeState()
        known = {field: raw[field] for field in RuntimeState.__dataclass_fields__ if field in raw}
        return RuntimeState(**known)

    def save(self, state: RuntimeState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
        os.replace(temporary, self.path)


@contextmanager
def single_instance(path: str | Path) -> Iterator[None]:
    """Hold an exclusive lock for the lifetime of the process.

    A second daemon fails immediately instead of trading the same account in
    parallel. The lock is released by the OS even if the process is killed.
    """
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                "another bot instance already holds the lock "
                f"({lock_path}); stop it before starting a new one"
            ) from exc
        yield
    finally:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()
