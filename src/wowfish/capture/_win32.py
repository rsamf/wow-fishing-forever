"""Windows window lookup through user32: the largest visible window whose title matches."""

from __future__ import annotations

import contextlib
import ctypes
import sys
from typing import Any

from wowfish.capture import Box, WindowRef

DWMWA_CLOAKED = 14
GW_OWNER = 4
GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x80


def client_rect_to_box(left: int, top: int, right: int, bottom: int, sx: int, sy: int) -> Box:
    """GetClientRect (0, 0, right, bottom) plus the client origin on screen (sx, sy) -> a Box."""
    return sx + left, sy + top, right - left, bottom - top


def set_dpi_aware() -> None:
    """Ask for per-monitor DPI awareness so client rectangles are in physical pixels."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # PER_MONITOR_V2
    except Exception:
        with contextlib.suppress(Exception):
            ctypes.windll.shcore.SetProcessDpiAwareness(2)


def list_windows_win32() -> list[WindowRef]:
    """The windows Alt-Tab would show: visible, titled, not cloaked, not a tool or owned window."""
    if sys.platform != "win32":
        return []
    from ctypes import wintypes

    set_dpi_aware()
    user32: Any = ctypes.windll.user32
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindow.argtypes = [wintypes.HWND, ctypes.c_uint]
    user32.GetWindow.restype = wintypes.HWND
    user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    found: list[WindowRef] = []
    proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cloaked(hwnd: int) -> bool:  # e.g. a UWP app on another virtual desktop
        value = wintypes.DWORD(0)
        try:
            ctypes.windll.dwmapi.DwmGetWindowAttribute(
                wintypes.HWND(hwnd), DWMWA_CLOAKED, ctypes.byref(value), ctypes.sizeof(value)
            )
        except Exception:
            return False
        return bool(value.value)

    def visit(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd) or user32.GetWindow(hwnd, GW_OWNER):
            return True
        if user32.GetWindowLongW(hwnd, GWL_EXSTYLE) & WS_EX_TOOLWINDOW or cloaked(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        rect = wintypes.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
            return True
        origin = wintypes.POINT(0, 0)
        user32.ClientToScreen(hwnd, ctypes.byref(origin))
        x, y, w, h = client_rect_to_box(
            rect.left, rect.top, rect.right, rect.bottom, origin.x, origin.y
        )
        pid = wintypes.DWORD(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if w > 0 and h > 0:
            found.append(WindowRef(buf.value, int(pid.value) or None, x, y, w, h, handle=int(hwnd)))
        return True

    user32.EnumWindows(proto(visit), 0)
    return found


def find_window_win32(title_substring: str) -> WindowRef | None:
    wanted = title_substring.lower()
    found = [r for r in list_windows_win32() if wanted in r.title.lower()]
    return max(found, key=lambda r: r.width * r.height, default=None)
