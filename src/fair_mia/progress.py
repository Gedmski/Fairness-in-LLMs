"""Dependency-free progress reporting with durable ETA snapshots."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TextIO


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class ProgressReporter:
    """Print throttled progress bars and persist their latest state as JSON."""

    def __init__(
        self,
        *,
        label: str,
        total: int,
        path: str | Path | None = None,
        enabled: bool = True,
        min_interval: float = 2.0,
        width: int = 28,
        stream: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.label = label
        self.total = max(0, int(total))
        self.path = Path(path) if path is not None else None
        self.enabled = enabled
        self.min_interval = max(0.0, float(min_interval))
        self.width = max(10, int(width))
        self.stream = stream or sys.stdout
        self.clock = clock
        self.completed = 0
        self.phase_name = "initializing"
        self.detail = ""
        self.started_at = datetime.now(timezone.utc)
        self.started_clock = self.clock()
        self.last_emit_clock = float("-inf")
        self._emit(force=True)

    def phase(self, name: str, detail: str = "") -> None:
        self.phase_name = name
        self.detail = detail
        self._emit(force=True)

    def update(self, *, advance: int = 1, detail: str = "") -> None:
        self.completed = min(self.total, self.completed + max(0, int(advance)))
        if detail:
            self.detail = detail
        self._emit(force=self.completed >= self.total)

    def finish(self, detail: str = "complete") -> None:
        self.completed = self.total
        self.phase_name = "complete"
        self.detail = detail
        self._emit(force=True)

    def _snapshot(self) -> dict[str, object]:
        now_clock = self.clock()
        elapsed = max(0.0, now_clock - self.started_clock)
        rate = self.completed / elapsed if self.completed > 0 and elapsed > 0 else 0.0
        remaining = max(0, self.total - self.completed)
        eta_seconds = remaining / rate if rate > 0 else None
        percent = 100.0 if self.total == 0 else 100.0 * self.completed / self.total
        return {
            "label": self.label,
            "phase": self.phase_name,
            "detail": self.detail,
            "completed": self.completed,
            "total": self.total,
            "percent": round(percent, 2),
            "elapsed_seconds": round(elapsed, 2),
            "eta_seconds": round(eta_seconds, 2) if eta_seconds is not None else None,
            "rate_per_second": round(rate, 6),
            "started_at": self.started_at.isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _emit(self, *, force: bool) -> None:
        if not self.enabled:
            return
        now = self.clock()
        if not force and now - self.last_emit_clock < self.min_interval:
            return
        self.last_emit_clock = now
        snapshot = self._snapshot()
        fraction = 1.0 if self.total == 0 else self.completed / self.total
        filled = min(self.width, int(round(self.width * fraction)))
        bar = "#" * filled + "-" * (self.width - filled)
        line = (
            f"[progress] {self.label} [{bar}] "
            f"{self.completed}/{self.total} {snapshot['percent']:6.2f}% "
            f"elapsed {format_duration(float(snapshot['elapsed_seconds']))} "
            f"ETA {format_duration(snapshot['eta_seconds'])} "
            f"phase={self.phase_name}"
        )
        if self.detail:
            line += f" | {self.detail}"
        print(line, file=self.stream, flush=True)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f"{self.path.name}.tmp")
            temporary.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(self.path)
