"""Masked template matching of one icon template against a frame or a region of it."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from wowfish.vision.region import FULL, Region
from wowfish.vision.template import IconTemplate, resize_template

NO_MATCH = -1.0
ICON_THRESHOLD = 0.75  # the score to accept the icon at until a run has measured its own
GOLD_MIN = 0.12  # masked mean of (R - B) / 255; the gold sprite is ~0.3, the grey one ~0
# Smallest template side, in match-space pixels, that still looks like a gear. Below this the
# sprite is a warm blob and scores ~0.85 on skin, hair, and wood (seen live: 19 px at 5K -> 7 px).
MIN_TEMPLATE_PX = 12


@dataclass(frozen=True)
class IconMatch:
    score: float  # TM_CCOEFF_NORMED in [-1, 1]; NO_MATCH when the crop is too small
    x: int  # icon centre, full-frame pixels
    y: int
    size: int  # template side at full-frame scale
    gold: float  # colour gate, see GOLD_MIN

    def accepted(self, threshold: float) -> bool:
        return self.score >= threshold and self.gold >= GOLD_MIN


class IconMatcher:
    def __init__(self, template: IconTemplate, match_width: int = 1920) -> None:
        self.template = template
        self.match_width = match_width
        self._scaled: dict[int, IconTemplate] = {}

    def _template(self, size: int) -> IconTemplate:
        t = self._scaled.get(size)
        if t is None:
            t = self._scaled[size] = resize_template(self.template, size)
        return t

    def template_px(self, frame_width: int) -> int:
        """Template side in match-space pixels for a frame this wide (after any downscale)."""
        return round(self.template.size * min(1.0, self.match_width / frame_width))

    def match(
        self, frame_bgr: np.ndarray, region: Region | None = None, *, full_width: int | None = None
    ) -> IconMatch:
        """Best match in `frame_bgr` (or its `region`). Pass `full_width` when `frame_bgr` is a
        crop of a wider window so it is downscaled exactly as the whole window would be."""
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame must be HxWx3 BGR")
        h, w = frame_bgr.shape[:2]
        ref_w = full_width or w
        tsize = self.template_px(ref_w)
        if tsize < MIN_TEMPLATE_PX:
            return IconMatch(NO_MATCH, -1, -1, self.template.size, 0.0)
        x0, y0, x1, y1 = (region or FULL).to_box(w, h)
        crop = np.ascontiguousarray(frame_bgr[y0:y1, x0:x1])
        scale = min(1.0, self.match_width / ref_w)
        if scale < 1.0:
            size = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
            crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        if crop.shape[0] < tsize or crop.shape[1] < tsize:
            return IconMatch(NO_MATCH, -1, -1, self.template.size, 0.0)
        t = self._template(tsize)
        result = cv2.matchTemplate(crop, t.bgr, cv2.TM_CCOEFF_NORMED, mask=t.mask)
        result = np.nan_to_num(result, nan=NO_MATCH, posinf=NO_MATCH, neginf=NO_MATCH)
        gold_map = _gold_map(crop, t.mask)[: result.shape[0], : result.shape[1]]
        valid = gold_map >= GOLD_MIN
        ranked = np.where(valid, result, NO_MATCH) if valid.any() else result
        _, _, _, (lx, ly) = cv2.minMaxLoc(ranked)
        score = float(min(1.0, result[ly, lx]))
        gold = float(gold_map[ly, lx])
        cx = x0 + round((lx + tsize / 2) / scale)
        cy = y0 + round((ly + tsize / 2) / scale)
        return IconMatch(score, cx, cy, self.template.size, gold)


def _gold_map(crop_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Masked mean of (R - B) / 255 for every top-left template position (same indexing as
    matchTemplate's result). Water is bluish (negative), the gold sprite is ~0.3."""
    rb = (crop_bgr[:, :, 2].astype(np.float32) - crop_bgr[:, :, 0].astype(np.float32)) / 255.0
    kernel = (mask > 0).astype(np.float32)
    kernel /= max(float(kernel.sum()), 1.0)
    return cv2.filter2D(rb, -1, kernel, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
