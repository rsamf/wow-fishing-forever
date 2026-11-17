"""Screen capture boundary: find the game window and grab its pixels as BGR arrays."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

import numpy as np

from wowfish.errors import NotSupported

Box = tuple[int, int, int, int]  # x, y, width, height in window pixels
GAME_TITLE = "World of Warcraft"  # only ever a guess: the user picks the window
MIN_SIDE_PX = 50  # smaller windows (tooltips, tray icons) are not offered


@dataclass(frozen=True)
class WindowRef:
    title: str
    pid: int | None
    left: int  # screen position; logical (unscaled) pixels for a Wayland window
    top: int
    width: int  # size of a grab, in pixels
    height: int
    handle: int | None = None  # native window id where the platform has one
    compositor: str | None = None  # "hyprland" for a native Wayland window, else None
    scale: float = 1.0  # the Wayland output's scale: grab pixels per logical pixel

    @property
    def box(self) -> Box:
        return self.left, self.top, self.width, self.height

    def moved(self, left: int, top: int, width: int, height: int) -> WindowRef:
        return replace(self, left=left, top=top, width=width, height=height)


@runtime_checkable
class Capturer(Protocol):
    def open(self, window: WindowRef) -> None: ...

    def grab(self, box: Box | None = None) -> np.ndarray:
        """Return the window (or `box` inside it) as HxWx3 uint8 BGR."""
        ...

    def close(self) -> None: ...


def linux_x11() -> bool:
    return sys.platform == "linux" and bool(os.environ.get("DISPLAY"))


def linux_hyprland() -> bool:
    return sys.platform == "linux" and bool(os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"))


def macos() -> bool:
    return sys.platform == "darwin"


def best_title_match(windows: Sequence[WindowRef], title_substring: str) -> WindowRef | None:
    """The window most likely meant by the title: one titled exactly so beats one that only
    mentions it (a browser tab about the game), and among equals the largest wins."""
    wanted = title_substring.lower()
    found = [w for w in windows if wanted in w.title.lower()]
    return max(
        found,
        key=lambda w: (w.title.strip().lower() == wanted, w.width * w.height),
        default=None,
    )


def _platform_windows() -> list[WindowRef]:
    """Every window the platform lists, in its own order."""
    found: list[WindowRef] = []
    if linux_hyprland():
        from wowfish.capture._wayland import list_windows_hyprland

        try:
            found = list_windows_hyprland()
        except (NotSupported, OSError, ValueError):
            if not linux_x11():
                raise
    if linux_x11():
        from wowfish.capture._x11 import list_windows_x11

        return found + list_windows_x11()
    if linux_hyprland():
        return found
    if sys.platform == "win32":
        from wowfish.capture import _win32

        return _win32.list_windows_win32()
    if macos():
        from wowfish.capture._quartz import list_windows_quartz

        return list_windows_quartz()
    return []


def list_windows() -> list[WindowRef]:
    """The windows a user can pick: titled, big enough to be an app's, and not wowfish's own."""
    own = os.getpid()
    return [
        w
        for w in _platform_windows()
        if w.pid != own and w.title.strip() and min(w.width, w.height) >= MIN_SIDE_PX
    ]


def grab_once(capturer: Capturer, window: WindowRef) -> np.ndarray:
    """One frame of the window from a capturer opened just for it."""
    capturer.open(window)
    try:
        return capturer.grab()
    finally:
        capturer.close()


def square_bgra(img: np.ndarray | None, size: int) -> np.ndarray | None:
    """An icon image as size x size BGRA; None when it is not a colour image."""
    import cv2

    if img is None or img.ndim != 3:
        return None
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def refresh_window(window: WindowRef) -> WindowRef | None:
    """The same window where it is now, looked up by its id; None only once it no longer exists.

    A window that cannot be measured right now (minimized, on another workspace), or a platform
    that cannot tell, gives the ref back unchanged.
    """
    if window.handle is None:
        return window
    try:
        if window.compositor == "hyprland":
            from wowfish.capture import _wayland

            return _wayland.refresh_hyprland(
                _wayland.hyprland_request("clients"), _wayland.hyprland_request("monitors"), window
            )
        if window.compositor:
            return window
        if linux_x11():
            from wowfish.capture._x11 import refresh_x11

            return refresh_x11(window)
        if sys.platform == "win32":
            from wowfish.capture import _win32

            return _win32.refresh_win32(window)
        if macos():
            from wowfish.capture import _quartz

            infos = _quartz.window_infos(_quartz.INCLUDING_WINDOW, window.handle)
            return _quartz.refreshed_from_infos(infos, window)
    except (NotSupported, OSError, ValueError, RuntimeError):
        return window
    return window


def get_capturer(window: WindowRef | None = None, *, own_pixels: bool = False) -> Capturer:
    """The capturer for `window`. `own_pixels` asks for one that still sees the window while
    another covers it: X11 always does; Windows and macOS switch from reading the screen to
    reading the window; a native Wayland window on Hyprland can only be read off the screen."""
    if window is not None and window.compositor:
        from wowfish.capture._wayland import GrimCapturer

        return GrimCapturer()
    if linux_x11():
        from wowfish.capture._x11 import X11WindowCapturer

        return X11WindowCapturer()
    if own_pixels and window is not None and window.handle is not None:
        from wowfish.capture._own import WindowPixelsCapturer

        if sys.platform == "win32":
            from wowfish.capture._win32 import snapshot_win32

            return WindowPixelsCapturer(snapshot_win32)
        if macos():
            from wowfish.capture._quartz import snapshot_quartz

            return WindowPixelsCapturer(snapshot_quartz)
    try:
        import mss  # noqa: F401
    except Exception:
        from wowfish.capture._stub import StubCapturer

        return StubCapturer()
    from wowfish.capture._mss import MssCapturer

    return MssCapturer()


__all__ = [
    "GAME_TITLE",
    "Box",
    "Capturer",
    "WindowRef",
    "best_title_match",
    "get_capturer",
    "linux_hyprland",
    "linux_x11",
    "list_windows",
    "macos",
    "refresh_window",
]
