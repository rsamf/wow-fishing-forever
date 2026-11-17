"""What the window picker offers: every open window, the title guess first, each with the name
of the app that owns it."""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from wowfish.capture import GAME_TITLE, WindowRef, find_window, list_windows, same_window


@dataclass(frozen=True)
class WindowChoice:
    window: WindowRef
    app: str  # the owning process's short name; "" when unknown
    guess: bool  # the window the title search would have taken


def order_choices(
    windows: Sequence[WindowRef], guess: WindowRef | None, app_of: Callable[[WindowRef], str]
) -> list[WindowChoice]:
    """The guess first and only once, then the rest from the largest down."""
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
    return order_choices(list_windows(), find_window(title), app_name)
