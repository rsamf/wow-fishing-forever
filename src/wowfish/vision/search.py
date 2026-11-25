"""Acquire the icon on the whole frame, then track it in a small box; fall back when lost."""

from __future__ import annotations

from typing import Literal

import numpy as np

from wowfish.vision.matcher import IconMatch, IconMatcher
from wowfish.vision.region import Region

Phase = Literal["acquire", "track"]


class IconSearch:
    def __init__(
        self,
        matcher: IconMatcher,
        threshold: float,
        *,
        margin: float = 0.10,
        lost_after: int = 3,
    ) -> None:
        self.matcher = matcher
        self.threshold = threshold
        self.margin = margin
        self.lost_after = lost_after
        self.region: Region | None = None
        self._misses = 0

    @property
    def phase(self) -> Phase:
        return "track" if self.region is not None else "acquire"

    def reset(self) -> None:
        self.region = None
        self._misses = 0

    def step(self, frame_bgr: np.ndarray, zone: Region | None = None) -> tuple[Phase, IconMatch]:
        """`zone` limits where the icon is acquired; once tracked, its own box is used."""
        phase = self.phase
        match = self.matcher.match(frame_bgr, self.region or zone)
        if match.accepted(self.threshold):
            h, w = frame_bgr.shape[:2]
            self.region = Region.around(match.x, match.y, w, h, self.margin)
            self._misses = 0
        elif self.region is not None:
            self._misses += 1
            if self._misses >= self.lost_after:
                self.reset()
        return phase, match
