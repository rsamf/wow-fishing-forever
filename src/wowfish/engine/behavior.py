"""Optional variations on the fishing loop's rhythm, each off by default. While a variation is
off it draws nothing from the random generator, so the loop runs exactly as without it."""

from __future__ import annotations

import random

from wowfish.config import BehaviorConfig


class Behavior:
    def __init__(self, cfg: BehaviorConfig, rng: random.Random) -> None:
        self.cfg = cfg
        self._rng = rng
        self._next_break: float | None = None

    def start(self, now: float) -> None:
        """Schedule the first break, counting from the start of the run."""
        self._schedule_break(now)

    def reaction_s(self) -> float:
        """How long to wait between hearing the splash and pressing Interact."""
        lo, hi = self.cfg.reaction_min_ms, self.cfg.reaction_max_ms
        if hi <= 0:
            return 0.0
        return self._rng.uniform(lo, hi) / 1000.0

    def break_s(self, now: float) -> float:
        """Seconds to rest now: 0 until the next break is due, which is then scheduled from
        the end of this one."""
        if self._next_break is None or now < self._next_break:
            return 0.0
        rest = self._rng.uniform(self.cfg.break_min_minutes, self.cfg.break_max_minutes) * 60.0
        self._schedule_break(now + rest)
        return rest

    def _schedule_break(self, after: float) -> None:
        lo, hi = self.cfg.break_every_min_minutes, self.cfg.break_every_max_minutes
        self._next_break = None if hi <= 0 else after + self._rng.uniform(lo, hi) * 60.0
