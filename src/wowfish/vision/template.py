"""Interact-icon templates: the gold gear sprites with their alpha channel as the match mask."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from wowfish.assets import ICON_SIZES, icon_path


@dataclass(frozen=True)
class IconTemplate:
    native_size: int  # sprite the pixels came from: 29 | 42 | 53 | 79 | 103
    size: int  # side in pixels after resizing
    bgr: np.ndarray  # (size, size, 3) uint8
    mask: np.ndarray  # (size, size) uint8, 255 where the sprite is fully opaque


def load_icon_templates() -> tuple[IconTemplate, ...]:
    out: list[IconTemplate] = []
    for n in ICON_SIZES:
        rgba = cv2.imread(str(icon_path(n)), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
            raise FileNotFoundError(f"missing or non-RGBA icon asset: {icon_path(n)}")
        bgr = np.ascontiguousarray(rgba[:, :, :3])
        # Fully opaque pixels only: the sprite carries a soft dark drop shadow (alpha 1-254,
        # ~40% of its box) that the game blends into the scene. Over bright water that shadow is
        # barely visible, and matching it capped a real gear at ~0.57 so a UI ring won.
        mask = (rgba[:, :, 3] == 255).astype(np.uint8) * 255
        out.append(IconTemplate(n, n, bgr, mask))
    return tuple(out)


def resize_template(t: IconTemplate, size: int) -> IconTemplate:
    if size == t.size:
        return t
    interp = cv2.INTER_AREA if size < t.size else cv2.INTER_CUBIC
    bgr = cv2.resize(t.bgr, (size, size), interpolation=interp)
    mask = cv2.resize(t.mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return IconTemplate(t.native_size, size, bgr, mask)


def template_for_size(templates: Sequence[IconTemplate], size: int) -> IconTemplate:
    """Smallest native sprite at least `size` wide (downscaling keeps detail), else the largest."""
    by_size = sorted(templates, key=lambda t: t.native_size)
    chosen = next((t for t in by_size if t.native_size >= size), by_size[-1])
    return resize_template(chosen, size)
