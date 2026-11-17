"""Windows window lookup through user32: the largest visible window whose title matches."""

from __future__ import annotations

import contextlib
import ctypes
import sys
from functools import cache
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


def refresh_win32(window: WindowRef) -> WindowRef | None:
    """The window's current client area; None once the handle is no longer a window."""
    if sys.platform != "win32" or window.handle is None:
        return window
    from ctypes import wintypes

    user32: Any = ctypes.windll.user32
    for fn in (user32.IsWindow, user32.IsIconic):
        fn.argtypes = [wintypes.HWND]
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    hwnd = window.handle
    if not user32.IsWindow(hwnd):
        return None
    rect = wintypes.RECT()
    if user32.IsIconic(hwnd) or not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return window
    origin = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(origin))
    x, y, w, h = client_rect_to_box(
        rect.left, rect.top, rect.right, rect.bottom, origin.x, origin.y
    )
    if w <= 0 or h <= 0:
        return window
    return window.moved(x, y, w, h)


class _BitmapInfoHeader(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.c_uint32),
        ("biWidth", ctypes.c_int32),
        ("biHeight", ctypes.c_int32),
        ("biPlanes", ctypes.c_uint16),
        ("biBitCount", ctypes.c_uint16),
        ("biCompression", ctypes.c_uint32),
        ("biSizeImage", ctypes.c_uint32),
        ("biXPelsPerMeter", ctypes.c_int32),
        ("biYPelsPerMeter", ctypes.c_int32),
        ("biClrUsed", ctypes.c_uint32),
        ("biClrImportant", ctypes.c_uint32),
    ]


def _top_down_header(width: int, height: int) -> _BitmapInfoHeader:
    return _BitmapInfoHeader(ctypes.sizeof(_BitmapInfoHeader), width, -height, 1, 32, 0)


@cache
def _gdi() -> tuple[Any, Any]:
    from ctypes import wintypes

    user32: Any = ctypes.windll.user32
    gdi32: Any = ctypes.windll.gdi32
    vp = ctypes.c_void_p
    for fn, args, res in (
        (user32.GetDC, [wintypes.HWND], vp),
        (user32.ReleaseDC, [wintypes.HWND, vp], ctypes.c_int),
        (user32.PrintWindow, [wintypes.HWND, vp, ctypes.c_uint], wintypes.BOOL),
        (user32.IsIconic, [wintypes.HWND], wintypes.BOOL),
        (user32.GetClientRect, [wintypes.HWND, ctypes.POINTER(wintypes.RECT)], wintypes.BOOL),
        (user32.DrawIconEx, [vp, ctypes.c_int, ctypes.c_int, vp, ctypes.c_int, ctypes.c_int,
                             ctypes.c_uint, vp, ctypes.c_uint], wintypes.BOOL),
        (user32.SendMessageTimeoutW, [wintypes.HWND, ctypes.c_uint, vp, vp, ctypes.c_uint,
                                      ctypes.c_uint, ctypes.POINTER(vp)], vp),
        (user32.GetClassLongPtrW, [wintypes.HWND, ctypes.c_int], vp),
        (gdi32.CreateCompatibleDC, [vp], vp),
        (gdi32.CreateCompatibleBitmap, [vp, ctypes.c_int, ctypes.c_int], vp),
        (gdi32.CreateDIBSection, [vp, vp, ctypes.c_uint, ctypes.POINTER(vp), vp, ctypes.c_uint32],
         vp),
        (gdi32.SelectObject, [vp, vp], vp),
        (gdi32.GetDIBits, [vp, vp, ctypes.c_uint, ctypes.c_uint, vp, vp, ctypes.c_uint],
         ctypes.c_int),
        (gdi32.DeleteObject, [vp], wintypes.BOOL),
        (gdi32.DeleteDC, [vp], wintypes.BOOL),
    ):  # fmt: skip
        fn.argtypes, fn.restype = args, res
    return user32, gdi32


def snapshot_win32(hwnd: int) -> Any:
    """The window's client area as BGR, drawn by the window itself (PrintWindow), so it shows
    even when covered; None when minimized or when the window will not draw."""
    if sys.platform != "win32":
        return None
    from ctypes import wintypes

    import numpy as np

    user32, gdi32 = _gdi()
    rect = wintypes.RECT()
    if user32.IsIconic(hwnd) or not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    w, h = rect.right - rect.left, rect.bottom - rect.top
    if w <= 0 or h <= 0:
        return None
    window_dc = user32.GetDC(hwnd)
    mem_dc = gdi32.CreateCompatibleDC(window_dc)
    bitmap = gdi32.CreateCompatibleBitmap(window_dc, w, h)
    old = gdi32.SelectObject(mem_dc, bitmap)
    try:
        if not user32.PrintWindow(hwnd, mem_dc, 1 | 2):  # PW_CLIENTONLY | PW_RENDERFULLCONTENT
            return None
        gdi32.SelectObject(mem_dc, old)
        header = _top_down_header(w, h)
        buf = ctypes.create_string_buffer(w * h * 4)
        if not gdi32.GetDIBits(mem_dc, bitmap, 0, h, buf, ctypes.byref(header), 0):
            return None
        return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()
    finally:
        gdi32.SelectObject(mem_dc, old)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(hwnd, window_dc)


def icon_win32(hwnd: int, size: int) -> Any:
    """The window's icon as size x size BGRA; None when it has none."""
    if sys.platform != "win32":
        return None
    import numpy as np

    user32, gdi32 = _gdi()
    icon = ctypes.c_void_p(0)
    for which in (1, 2):  # WM_GETICON: ICON_BIG, then ICON_SMALL2
        user32.SendMessageTimeoutW(hwnd, 0x7F, which, 0, 0x2, 200, ctypes.byref(icon))
        if icon.value:
            break
    handle = icon.value or user32.GetClassLongPtrW(hwnd, -14)  # GCLP_HICON
    if not handle:
        return None
    header = _top_down_header(size, size)
    bits = ctypes.c_void_p()
    mem_dc = gdi32.CreateCompatibleDC(None)
    bitmap = gdi32.CreateDIBSection(mem_dc, ctypes.byref(header), 0, ctypes.byref(bits), None, 0)
    if not bitmap or not bits.value:
        gdi32.DeleteDC(mem_dc)
        return None
    old = gdi32.SelectObject(mem_dc, bitmap)
    try:
        if not user32.DrawIconEx(mem_dc, 0, 0, handle, size, size, 0, None, 3):  # DI_NORMAL
            return None
        raw = ctypes.string_at(bits.value, size * size * 4)
        bgra = np.frombuffer(raw, dtype=np.uint8).reshape(size, size, 4).copy()
        if not bgra[:, :, 3].any():  # an icon without an alpha channel: opaque
            bgra[:, :, 3] = 255
        return bgra
    finally:
        gdi32.SelectObject(mem_dc, old)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(mem_dc)
