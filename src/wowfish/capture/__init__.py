"""Screen capture boundary: find the game window and grab its pixels as BGR arrays."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
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


def same_window(a: WindowRef, b: WindowRef) -> bool:
    """Whether two refs name one window, wherever it has moved to since."""
    if a.handle is not None:
        return (a.handle, a.compositor) == (b.handle, b.compositor)
    return a == b


def _platform_windows() -> list[WindowRef]:
    """Every window the platform lists, in its own order."""
    if linux_hyprland():
        from wowfish.capture._wayland import list_windows_hyprland

        try:
            found = list_windows_hyprland()
        except (NotSupported, OSError, ValueError):
            if not linux_x11():
                raise
            found = []
        if linux_x11():
            from wowfish.capture._x11 import list_windows_x11

            found += list_windows_x11()
        return found
    if linux_x11():
        from wowfish.capture._x11 import list_windows_x11

        return list_windows_x11()
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


def find_window(title_substring: str = GAME_TITLE) -> WindowRef | None:
    """The largest viewable window whose title contains the text: the picker's first guess.

    On Hyprland the compositor picks it from all its windows, native Wayland and Xwayland alike;
    an Xwayland winner is then looked up through X11 by its pid. Elsewhere X11, then Windows,
    then macOS (in points, which mss grabs as is); else None.
    """
    pid: int | None = None
    if linux_hyprland():
        from wowfish.capture._wayland import find_window_hyprland

        try:
            found = find_window_hyprland(title_substring)
        except (NotSupported, OSError, ValueError):
            if not linux_x11():
                raise
            # Hyprland's IPC did not answer; X11 alone still finds an Xwayland game
        else:
            if found is None or found.compositor:
                return found
            pid = found.pid
    if linux_x11():
        from wowfish.capture._x11 import find_window_x11

        return find_window_x11(title_substring, pid)
    if sys.platform == "win32":
        from wowfish.capture import _win32

        return _win32.find_window_win32(title_substring)
    if macos():
        from wowfish.capture._quartz import find_window_quartz

        return find_window_quartz(title_substring)
    return None


def get_capturer(window: WindowRef | None = None) -> Capturer:
    if window is not None and window.compositor:
        from wowfish.capture._wayland import GrimCapturer

        return GrimCapturer()
    if linux_x11():
        from wowfish.capture._x11 import X11WindowCapturer

        return X11WindowCapturer()
    try:
        import mss  # noqa: F401
    except Exception:
        from wowfish.capture._stub import StubCapturer

        return StubCapturer()
    from wowfish.capture._mss import MssCapturer

    return MssCapturer()


__all__ = [
    "GAME_TITLE",
    "MIN_SIDE_PX",
    "Box",
    "Capturer",
    "WindowRef",
    "find_window",
    "get_capturer",
    "linux_hyprland",
    "linux_x11",
    "list_windows",
    "macos",
    "same_window",
]
