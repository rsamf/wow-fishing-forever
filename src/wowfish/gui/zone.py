"""The search zone's drag maths: corners resize, the inside pans. Everything is in fractions of
the window, so no Tk here."""

from __future__ import annotations

from wowfish.vision.region import Region

MIN_SIDE = 0.1  # smallest zone side, as a fraction of the window: the gear must still fit
CORNERS = ("nw", "ne", "sw", "se")


def _clamp(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


def resize(zone: Region, corner: str, fx: float, fy: float) -> Region:
    """`zone` with `corner` dragged to (fx, fy); the opposite corner stays where it is."""
    left, top, right, bottom = zone.left, zone.top, zone.right, zone.bottom
    if "w" in corner:
        left = _clamp(fx, 0.0, right - MIN_SIDE)
    else:
        right = _clamp(fx, left + MIN_SIDE, 1.0)
    if "n" in corner:
        top = _clamp(fy, 0.0, bottom - MIN_SIDE)
    else:
        bottom = _clamp(fy, top + MIN_SIDE, 1.0)
    return Region(left, top, right, bottom)


def pan(zone: Region, dfx: float, dfy: float) -> Region:
    """`zone` moved by (dfx, dfy), same size, stopped at the window's edges."""
    w, h = zone.right - zone.left, zone.bottom - zone.top
    left = _clamp(zone.left + dfx, 0.0, 1.0 - w)
    top = _clamp(zone.top + dfy, 0.0, 1.0 - h)
    return Region(left, top, min(1.0, left + w), min(1.0, top + h))


def grip(zone: Region, fx: float, fy: float, rx: float, ry: float) -> str | None:
    """What a press at (fx, fy) takes hold of: a corner within (rx, ry) of it, "move" inside
    the zone, or None outside."""
    for corner in CORNERS:
        cx = zone.left if "w" in corner else zone.right
        cy = zone.top if "n" in corner else zone.bottom
        if abs(fx - cx) <= rx and abs(fy - cy) <= ry:
            return corner
    if zone.left <= fx <= zone.right and zone.top <= fy <= zone.bottom:
        return "move"
    return None
