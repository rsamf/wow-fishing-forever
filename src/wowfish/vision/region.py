"""A rectangle expressed as fractions of a frame, so it survives resolution changes."""

from __future__ import annotations

from dataclasses import dataclass


def clamp(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


@dataclass(frozen=True)
class Region:
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        ok_x = 0.0 <= self.left < self.right <= 1.0
        ok_y = 0.0 <= self.top < self.bottom <= 1.0
        if not (ok_x and ok_y):
            raise ValueError(f"invalid region {self}")

    def to_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Pixel box (x0, y0, x1, y1), exclusive on the right/bottom, never empty."""
        x0, y0 = int(self.left * width), int(self.top * height)
        x1 = min(width, max(x0 + 1, round(self.right * width)))
        y1 = min(height, max(y0 + 1, round(self.bottom * height)))
        return x0, y0, x1, y1

    @classmethod
    def around(cls, x: int, y: int, width: int, height: int, margin: float = 0.10) -> Region:
        """A box of ±margin (fraction of the frame) around pixel (x, y), clamped to the frame."""
        fx, fy = clamp(x / width, 0.0, 1.0), clamp(y / height, 0.0, 1.0)
        return cls(
            max(0.0, fx - margin),
            max(0.0, fy - margin),
            min(1.0, fx + margin),
            min(1.0, fy + margin),
        )


FULL = Region(0.0, 0.0, 1.0, 1.0)
