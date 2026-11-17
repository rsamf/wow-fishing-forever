"""What the window picker offers: every open window, the title guess first, each with the name
of the app that owns it."""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from wowfish.capture import (
    GAME_TITLE,
    WindowRef,
    best_title_match,
    linux_x11,
    list_windows,
    same_window,
)


@dataclass(frozen=True)
class WindowChoice:
    window: WindowRef
    app: str  # the owning process's short name; "" when unknown
    guess: bool  # the window the title search would have taken


def order_choices(
    windows: Sequence[WindowRef], guess: WindowRef | None, app_of: Callable[[WindowRef], str]
) -> list[WindowChoice]:
    """The guess first and only once, then the rest from the largest down."""
    if guess is not None:  # the copy from the list is the complete one (its pid filled in)
        guess = next((w for w in windows if same_window(guess, w)), guess)
    rest = [w for w in windows if guess is None or not same_window(guess, w)]
    rest.sort(key=lambda w: w.width * w.height, reverse=True)
    first = [WindowChoice(guess, app_of(guess), True)] if guess is not None else []
    return first + [WindowChoice(w, app_of(w), False) for w in rest]


def _app_name_win32(pid: int) -> str:
    from ctypes import wintypes

    k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD),
    ]  # fmt: skip
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ""
    try:
        size = wintypes.DWORD(1024)
        buf = ctypes.create_unicode_buffer(size.value)
        if not k32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return ""
        return buf.value.replace("/", "\\").rsplit("\\", 1)[-1]
    finally:
        k32.CloseHandle(handle)


def _app_name_macos(pid: int) -> str:
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
    buf = ctypes.create_string_buffer(256)
    n = libproc.proc_name(ctypes.c_int(pid), buf, ctypes.c_uint32(len(buf)))
    return buf.value.decode("utf-8", "replace") if n > 0 else ""


def app_name(window: WindowRef) -> str:
    """The short name of the process that owns the window, or "" when it cannot be told."""
    if window.pid is None:
        return ""
    try:
        if sys.platform == "linux":
            with open(f"/proc/{window.pid}/comm", encoding="utf-8", errors="replace") as f:
                return f.read().strip()
        if sys.platform == "win32":
            return _app_name_win32(window.pid)
        if sys.platform == "darwin":
            return _app_name_macos(window.pid)
    except Exception:
        return ""
    return ""


def window_choices(title: str = GAME_TITLE) -> list[WindowChoice]:
    """Every window the user can pick, the title guess first. Platform errors propagate."""
    windows = list_windows()  # listed once: the guess is one of these
    return order_choices(windows, best_title_match(windows, title), app_name)


def _snapshot(window: WindowRef) -> np.ndarray | None:
    if window.compositor == "hyprland":
        from wowfish.capture._wayland import snapshot_hyprland

        return snapshot_hyprland(window)
    if window.handle is None:
        return None
    if linux_x11():
        from wowfish.capture._x11 import X11WindowCapturer

        cap = X11WindowCapturer()
        cap.open(window)
        try:
            return cap.grab()
        finally:
            cap.close()
    if sys.platform == "win32":
        from wowfish.capture._win32 import snapshot_win32

        return snapshot_win32(window.handle)
    if sys.platform == "darwin":
        from wowfish.capture._quartz import snapshot_quartz

        return snapshot_quartz(window.handle)
    return None


def window_snapshot(window: WindowRef, width: int = 240) -> np.ndarray | None:
    """What the window shows now, as BGR at most `width` px wide; None when it cannot be had
    (minimized, on another workspace, no permission), which the picker shows as a placeholder."""
    import cv2

    try:
        frame = _snapshot(window)
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        if w > width:
            size = (width, max(1, round(h * width / w)))
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        return frame
    except Exception:
        return None


def window_icon(window: WindowRef, size: int = 24) -> np.ndarray | None:
    """The window's (or its app's) icon as size x size BGRA; None when there is none to show."""
    try:
        if window.compositor == "hyprland":
            from wowfish.capture._wayland import icon_hyprland

            return icon_hyprland(window, size)
        if linux_x11() and window.handle is not None:
            from wowfish.capture._x11 import icon_x11

            return icon_x11(window.handle, size)
        if sys.platform == "win32" and window.handle is not None:
            from wowfish.capture._win32 import icon_win32

            return icon_win32(window.handle, size)
        if sys.platform == "darwin" and window.pid is not None:
            from wowfish.capture._quartz import icon_quartz

            return icon_quartz(window.pid, size)
    except Exception:
        return None
    return None
