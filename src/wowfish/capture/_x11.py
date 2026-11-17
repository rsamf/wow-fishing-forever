"""Linux: find the game's X11/Xwayland window and read its pixels with XGetImage through ctypes.

python-xlib's own GetImage path copies 60 MB per 5K frame in pure Python (1.6 s); the ctypes call
does the same request in ~150 ms, and a 60 % region in ~30 ms. Root-window grabs do not work under
rootless Xwayland, so this always reads the game window itself.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from collections.abc import Sequence
from typing import Any

import numpy as np

from wowfish.capture import Box, WindowRef, best_title_match
from wowfish.errors import NotSupported

_ZPIXMAP = 2
_ALL_PLANES = 0xFFFFFFFF


class _XImage(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("xoffset", ctypes.c_int),
        ("format", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_ubyte)),
        ("byte_order", ctypes.c_int),
        ("bitmap_unit", ctypes.c_int),
        ("bitmap_bit_order", ctypes.c_int),
        ("bitmap_pad", ctypes.c_int),
        ("depth", ctypes.c_int),
        ("bytes_per_line", ctypes.c_int),
        ("bits_per_pixel", ctypes.c_int),
    ]


class _XErrorEvent(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("display", ctypes.c_void_p),
        ("resourceid", ctypes.c_ulong),
        ("serial", ctypes.c_ulong),
        ("error_code", ctypes.c_ubyte),
        ("request_code", ctypes.c_ubyte),
        ("minor_code", ctypes.c_ubyte),
    ]


_ERROR_HANDLER_TYPE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_XErrorEvent))
_last_error: list[int] = []


@_ERROR_HANDLER_TYPE
def _record_error(_display: Any, event: Any) -> int:
    # libX11's default handler prints and exits the whole process; we want a Python exception.
    _last_error.append(int(event.contents.error_code))
    return 0


_lib_cache: Any = None


def _libx11() -> Any:
    global _lib_cache
    if _lib_cache is not None:
        return _lib_cache
    path = ctypes.util.find_library("X11")
    if not path:
        raise NotSupported("X11 capture (libX11 not found)")
    lib = ctypes.CDLL(path)
    lib.XSetErrorHandler.restype = ctypes.c_void_p
    lib.XSetErrorHandler.argtypes = [_ERROR_HANDLER_TYPE]
    lib.XSetErrorHandler(_record_error)
    lib.XOpenDisplay.restype = ctypes.c_void_p
    lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
    lib.XCloseDisplay.argtypes = [ctypes.c_void_p]
    lib.XGetImage.restype = ctypes.c_void_p
    lib.XGetImage.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_ulong,
        ctypes.c_int,
    ]
    lib.XDestroyImage.argtypes = [ctypes.c_void_p]
    _lib_cache = lib
    return lib


def _xres_pid(d: Any, window: int) -> int | None:
    """The pid the X server itself records for the window's client (X-Resource extension)."""
    try:
        from Xlib.ext import res

        reply = d.res_query_client_ids([{"client": window, "mask": res.LocalClientPIDMask}])
    except Exception:  # no X-Resource on this server
        return None
    return next((int(i.value[0]) for i in reply.ids if i.value), None)


def _titled_windows(d: Any) -> list[WindowRef]:
    """Every viewable window with a title, minus window-manager frames around a titled client."""
    from Xlib import X

    root = d.screen().root
    name_atom = d.intern_atom("_NET_WM_NAME")
    pid_atom = d.intern_atom("_NET_WM_PID")
    matches: list[tuple[WindowRef, list[int]]] = []  # (ref, ancestor ids)

    def title_of(w: Any) -> str:
        try:
            prop = w.get_full_property(name_atom, 0)
            if prop and prop.value:
                return bytes(prop.value).decode("utf-8", "replace")
            name = w.get_wm_name()
            if isinstance(name, str):
                return name
            return (name or b"").decode("utf-8", "replace")
        except Exception:
            return ""

    def visit(w: Any, ancestors: list[int]) -> None:
        try:
            children = w.query_tree().children
        except Exception:  # the window went away mid-walk
            return
        for child in children:
            title = title_of(child)
            if title:
                try:
                    attrs = child.get_attributes()
                    geo = child.get_geometry()
                    # InputOnly windows draw nothing (e.g. mutter's guard window)
                    if attrs.map_state == X.IsViewable and attrs.win_class != X.InputOnly:
                        origin = root.translate_coords(child, 0, 0)
                        pid_prop = child.get_full_property(pid_atom, 0)
                        wm_pid = int(pid_prop.value[0]) if pid_prop and pid_prop.value else None
                        ref = WindowRef(
                            title, wm_pid, origin.x, origin.y, geo.width, geo.height, child.id
                        )
                        matches.append((ref, ancestors))
                except Exception:
                    pass
            visit(child, [*ancestors, child.id])

    visit(root, [])
    # A window-manager frame repeats the client's title; prefer the client (the descendant).
    ancestor_ids = {a for _, ancestors in matches for a in ancestors}
    return [ref for ref, _ in matches if ref.handle not in ancestor_ids]


def list_windows_x11() -> list[WindowRef]:
    """Every viewable titled client window. A window without _NET_WM_PID gets the pid the X
    server records, so wowfish's own Tk window is recognised."""
    from dataclasses import replace

    from Xlib import display

    d = display.Display()
    try:
        return [
            r if r.pid is not None else replace(r, pid=_xres_pid(d, r.handle or 0))
            for r in _titled_windows(d)
        ]
    finally:
        d.close()


def find_window_x11(title_substring: str, pid: int | None = None) -> WindowRef | None:
    """The largest viewable match, among those of process `pid` when any of them is.

    Hyprland reports a pid from _NET_WM_PID or from the X server's record, and the two differ
    when the game runs in a container with its own pid namespace, so either one counts.
    """
    from Xlib import display

    d = display.Display()
    try:
        wanted = title_substring.lower()
        clients = [r for r in _titled_windows(d) if wanted in r.title.lower()]
        if pid is not None:
            owned = [r for r in clients if pid in (r.pid, _xres_pid(d, r.handle or 0))]
            clients = owned or clients
        return best_title_match(clients, title_substring)
    finally:
        d.close()


def refresh_x11(window: WindowRef) -> WindowRef | None:
    """The window's current place and size; None once the X server no longer knows its id."""
    from dataclasses import replace

    from Xlib import X, display
    from Xlib.error import BadDrawable, BadWindow

    d = display.Display()
    try:
        w = d.create_resource_object("window", window.handle)
        try:
            attrs = w.get_attributes()
            if attrs.map_state != X.IsViewable:
                return window  # minimized or on another workspace: still open
            geo = w.get_geometry()
            origin = d.screen().root.translate_coords(w, 0, 0)
        except (BadWindow, BadDrawable):
            return None
        return replace(window, left=origin.x, top=origin.y, width=geo.width, height=geo.height)
    finally:
        d.close()


def icon_from_net_wm_icon(values: Sequence[int], size: int) -> np.ndarray | None:
    """A size x size BGRA icon from a _NET_WM_ICON list (width, height, then that many ARGB
    pixels, repeated): the smallest icon at least `size` wide, else the largest."""
    import cv2

    icons: list[tuple[int, int, int]] = []  # (width, height, offset of the pixels)
    i = 0
    while i + 2 <= len(values):
        w, h = int(values[i]), int(values[i + 1])
        if w <= 0 or h <= 0 or i + 2 + w * h > len(values):
            break
        icons.append((w, h, i + 2))
        i += 2 + w * h
    if not icons:
        return None
    big_enough = [icon for icon in icons if icon[0] >= size]
    w, h, at = min(big_enough) if big_enough else max(icons)
    argb = np.asarray(values[at : at + w * h], dtype=np.uint64).astype(np.uint32).reshape(h, w)
    bgra = np.dstack([(argb >> shift) & 0xFF for shift in (0, 8, 16, 24)]).astype(np.uint8)
    if (w, h) != (size, size):
        bgra = cv2.resize(bgra, (size, size), interpolation=cv2.INTER_AREA)
    return bgra


def icon_x11(handle: int, size: int) -> np.ndarray | None:
    from Xlib import X, display

    d = display.Display()
    try:
        w = d.create_resource_object("window", handle)
        prop = w.get_full_property(d.intern_atom("_NET_WM_ICON"), X.AnyPropertyType, 1 << 22)
        return icon_from_net_wm_icon(list(prop.value), size) if prop else None
    finally:
        d.close()


class X11WindowCapturer:
    def __init__(self) -> None:
        self._lib: Any = None
        self._dpy: int | None = None
        self._window: WindowRef | None = None

    def open(self, window: WindowRef) -> None:
        if window.handle is None:
            raise ValueError("X11 capture needs WindowRef.handle (use find_window)")
        self._lib = _libx11()
        self._dpy = self._lib.XOpenDisplay(None)
        if not self._dpy:
            raise NotSupported("X11 capture (cannot open the display)")
        self._window = window

    def grab(self, box: Box | None = None) -> np.ndarray:
        if self._window is None or self._dpy is None:
            raise RuntimeError("capturer is not open")
        x, y, w, h = box or (0, 0, self._window.width, self._window.height)
        handle = self._window.handle
        _last_error.clear()
        ptr = self._lib.XGetImage(self._dpy, handle, x, y, w, h, _ALL_PLANES, _ZPIXMAP)
        if not ptr:
            code = f" (X error {_last_error[-1]})" if _last_error else ""
            raise RuntimeError(f"XGetImage failed{code}: window closed, or box outside the window")
        try:
            img = ctypes.cast(ptr, ctypes.POINTER(_XImage)).contents
            if img.bits_per_pixel != 32:
                raise RuntimeError(f"unsupported pixel format: {img.bits_per_pixel} bpp")
            raw = ctypes.string_at(img.data, img.bytes_per_line * img.height)
            stride = img.bytes_per_line // 4
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(img.height, stride, 4)
            return np.ascontiguousarray(arr[:, :w, :3])  # BGRX -> BGR
        finally:
            self._lib.XDestroyImage(ptr)

    def close(self) -> None:
        if self._dpy:
            self._lib.XCloseDisplay(self._dpy)
        self._dpy = None
        self._window = None
