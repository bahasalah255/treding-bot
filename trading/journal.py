"""Append-only JSONL journal for every decision the demo bot makes."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator


class Journal:
    """Write one JSON object per line to a date-stamped file under ``directory``."""

    def __init__(self, directory: str | Path, echo: bool = True) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.echo = echo

    def path_for(self, day: date) -> Path:
        return self.directory / f"journal-{day.isoformat()}.jsonl"

    def write(self, event: str, **fields: Any) -> dict[str, Any]:
        """Append one event and return the record that was written."""
        now = datetime.now(timezone.utc)
        record: dict[str, Any] = {
            "time": now.isoformat(),
            "event": event,
            **fields,
        }
        line = json.dumps(record, default=str, ensure_ascii=False)
        path = self.path_for(now.date())
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if self.echo:
            print(line)
        return record


def read_events(directory: str | Path) -> list[dict[str, Any]]:
    """Read every journal file in chronological file order."""
    folder = Path(directory)
    if not folder.exists():
        return []
    events: list[dict[str, Any]] = []
    for path in sorted(folder.glob("journal-*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number} is not valid JSON") from exc
    return events


def iter_events(directory: str | Path) -> Iterator[dict[str, Any]]:
    """Yield journal events without loading every file into memory at once."""
    yield from read_events(directory)
