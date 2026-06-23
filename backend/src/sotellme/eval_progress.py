"""Live, per-persona progress and cost reporting for a simulated eval run."""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable

# Widest committed persona name ("mid-blurred-ownership") plus a little slack, so the outcome and
# cost columns line up regardless of which persona is on the line.
_NAME_WIDTH = 22

_OUTCOME = {
    "completed": ("✓", "done"),
    "terminated": ("⊘", "terminated"),
    "max_turns": ("⚠", "hit turn cap"),
}


def _stderr(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


class EvalProgress:
    """Per-persona start/finish lines with a live running cost."""

    def __init__(self, total: int, write: Callable[[str], None] = _stderr) -> None:
        self._total = total
        self._write = write
        # Personas finish on separate worker threads, so a lock keeps each line atomic; without
        # it two near-simultaneous writes interleave and garble the output.
        self._lock = threading.Lock()

    def _emit(self, line: str) -> None:
        with self._lock:
            self._write(line)

    def start(self, index: int, name: str) -> None:
        """Announce that a persona's interview has begun."""
        self._emit(f"[{index}/{self._total}] {name:<{_NAME_WIDTH}} running…")

    def finish(
        self,
        index: int,
        name: str,
        *,
        turns: int,
        finished_reason: str,
        persona_usd: float,
        total_usd: float,
    ) -> None:
        """Report a finished persona: outcome, turns, its own cost, and the run's running total."""
        glyph, label = _OUTCOME.get(finished_reason, ("·", finished_reason))
        turn_word = "turn" if turns == 1 else "turns"
        self._emit(
            f"[{index}/{self._total}] {name:<{_NAME_WIDTH}} {glyph} {label} · "
            f"{turns} {turn_word} · ${persona_usd:.2f}   (run total ${total_usd:.2f})"
        )
