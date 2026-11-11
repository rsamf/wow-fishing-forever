"""macOS: find a window in CoreGraphics' window list and check the Screen Recording permission.

ctypes only (no pyobjc), loaded on first use so this module imports on every OS. Window bounds
are global points with a top-left origin, the space mss grabs at nominal resolution, so a
WindowRef from here has scale 1.0 and MssCapturer grabs it as is.
"""

from __future__ import annotations

import ctypes
import os
import sys
from functools import cache
from typing import Any

from wowfish.capture import WindowRef
from wowfish.errors import NotSupported

CORE_GRAPHICS = "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
CORE_FOUNDATION = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
APPLICATION_SERVICES = (
    "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
)

ON_SCREEN_ONLY = 1 << 0  # kCGWindowListOptionOnScreenOnly
EXCLUDE_DESKTOP = 1 << 4  # kCGWindowListExcludeDesktopElements
UTF8 = 0x08000100  # kCFStringEncodingUTF8
SINT64 = 4  # kCFNumberSInt64Type
DOUBLE = 13  # kCFNumberDoubleType

# TERM_PROGRAM of the usual hosts. macOS grants privacy permissions to the app that started
# wowfish (a double-clicked executable runs in Terminal), so that is the app to name.
_HOSTS = {
    "Apple_Terminal": "Terminal",
    "iTerm.app": "iTerm",
    "vscode": "Visual Studio Code",
    "ghostty": "Ghostty",
}


class CGPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


class CGSize(ctypes.Structure):
    _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]


class CGRect(ctypes.Structure):
    _fields_ = [("origin", CGPoint), ("size", CGSize)]


def host_app() -> str:
    return _HOSTS.get(os.environ.get("TERM_PROGRAM", ""), "the app you started wowfish from")


@cache
def frameworks() -> tuple[Any, Any]:
    """(CoreGraphics, CoreFoundation) with the signatures wowfish calls declared."""
    if sys.platform != "darwin":
        raise NotSupported("Quartz outside macOS")
    try:
        cg = ctypes.CDLL(CORE_GRAPHICS)
        cf = ctypes.CDLL(CORE_FOUNDATION)
    except OSError as e:
        raise NotSupported(f"CoreGraphics ({e})") from e
    vp, idx = ctypes.c_void_p, ctypes.c_long  # CFIndex is a signed long
    for fn, args, res in (
        (cf.CFArrayGetCount, [vp], idx),
        (cf.CFArrayGetValueAtIndex, [vp, idx], vp),
        (cf.CFDictionaryGetValue, [vp, vp], vp),
        (cf.CFDictionaryCreate, [vp, vp, vp, idx, vp, vp], vp),
        (cf.CFStringGetLength, [vp], idx),
        (cf.CFStringGetMaximumSizeForEncoding, [idx, ctypes.c_uint32], idx),
        (cf.CFStringGetCString, [vp, ctypes.c_char_p, idx, ctypes.c_uint32], ctypes.c_bool),
        (cf.CFNumberGetValue, [vp, idx, vp], ctypes.c_bool),
        (cf.CFRelease, [vp], None),
        (cg.CGWindowListCopyWindowInfo, [ctypes.c_uint32, ctypes.c_uint32], vp),
        (cg.CGRectMakeWithDictionaryRepresentation, [vp, ctypes.POINTER(CGRect)], ctypes.c_bool),
        (cg.CGEventSourceCreate, [ctypes.c_int32], vp),
        (cg.CGEventCreateKeyboardEvent, [vp, ctypes.c_uint16, ctypes.c_bool], vp),
        (cg.CGEventSetFlags, [vp, ctypes.c_uint64], None),
        (cg.CGEventPost, [ctypes.c_uint32, vp], None),
    ):
        fn.argtypes, fn.restype = args, res
    return cg, cf


def constant(lib: Any, name: str) -> int:
    """The value of an exported CFStringRef/CFBooleanRef global such as kCGWindowName."""
    return ctypes.c_void_p.in_dll(lib, name).value or 0


def _string(cf: Any, ref: int | None) -> str | None:
    if not ref:
        return None
    size = cf.CFStringGetMaximumSizeForEncoding(cf.CFStringGetLength(ref), UTF8) + 1
    buf = ctypes.create_string_buffer(size)
    if not cf.CFStringGetCString(ref, buf, size, UTF8):
        return None
    return buf.value.decode("utf-8", "replace")


def _number(cf: Any, ref: int | None, kind: int = SINT64) -> float | None:
    if not ref:
        return None
    out: ctypes.c_int64 | ctypes.c_double = (
        ctypes.c_double() if kind == DOUBLE else ctypes.c_int64()
    )
    cf.CFNumberGetValue(ref, kind, ctypes.byref(out))
    return out.value


def window_infos() -> list[dict[str, Any]]:
    """Every on-screen window, front to back, as plain dicts (see pick_quartz_window).

    Without the Screen Recording permission macOS leaves out other apps' window names.
    """
    cg, cf = frameworks()
    key = {
        k: constant(cg, f"kCGWindow{k}")
        for k in ("Number", "Name", "OwnerName", "OwnerPID", "Layer", "Alpha", "Bounds")
    }
    array = cg.CGWindowListCopyWindowInfo(ON_SCREEN_ONLY | EXCLUDE_DESKTOP, 0)
    if not array:
        return []
    infos: list[dict[str, Any]] = []
    try:
        for i in range(cf.CFArrayGetCount(array)):
            d = cf.CFArrayGetValueAtIndex(array, i)
            rect = CGRect()
            bounds = cf.CFDictionaryGetValue(d, key["Bounds"])
            if not bounds or not cg.CGRectMakeWithDictionaryRepresentation(
                bounds, ctypes.byref(rect)
            ):
                continue
            pid = _number(cf, cf.CFDictionaryGetValue(d, key["OwnerPID"]))
            number = _number(cf, cf.CFDictionaryGetValue(d, key["Number"]))
            alpha = _number(cf, cf.CFDictionaryGetValue(d, key["Alpha"]), DOUBLE)
            infos.append(
                {
                    "number": None if number is None else int(number),
                    "name": _string(cf, cf.CFDictionaryGetValue(d, key["Name"])),
                    "owner": _string(cf, cf.CFDictionaryGetValue(d, key["OwnerName"])),
                    "pid": None if pid is None else int(pid),
                    "layer": int(_number(cf, cf.CFDictionaryGetValue(d, key["Layer"])) or 0),
                    "alpha": 1.0 if alpha is None else alpha,
                    "bounds": (rect.origin.x, rect.origin.y, rect.size.width, rect.size.height),
                }
            )
    finally:
        cf.CFRelease(array)
    return infos


def _visible(info: dict[str, Any]) -> bool:
    """A normal app window (layer 0) that shows something; the menu bar and Dock sit higher."""
    _, _, w, h = info["bounds"]
    return info.get("layer") == 0 and info.get("alpha", 1.0) > 0 and w >= 1 and h >= 1


def pick_quartz_window(infos: list[dict[str, Any]], title_substring: str) -> WindowRef | None:
    """The largest visible window whose title contains the text (any case).

    When macOS hides the title (no Screen Recording permission) the app's name stands in for
    it, so the game is still found and doctor can say what is missing.
    """
    want = title_substring.lower()
    best: WindowRef | None = None
    for info in infos:
        if not _visible(info):
            continue
        name, owner = info.get("name") or "", info.get("owner") or ""
        if want in name.lower():
            title = name
        elif not name and want in owner.lower():
            title = owner
        else:
            continue
        x, y, w, h = info["bounds"]
        ref = WindowRef(
            title,
            info.get("pid"),
            round(x),
            round(y),
            round(w),
            round(h),
            handle=info.get("number"),
        )
        if best is None or ref.width * ref.height > best.width * best.height:
            best = ref
    return best


def frontmost_pid(infos: list[dict[str, Any]]) -> int | None:
    """The process that owns the front-most visible app window (the list runs front to back)."""
    return next((info.get("pid") for info in infos if _visible(info)), None)


def find_window_quartz(title_substring: str) -> WindowRef | None:
    return pick_quartz_window(window_infos(), title_substring)


def screen_capture_available(request: bool = False) -> tuple[bool, str]:
    """Whether macOS lets this process see other apps' windows; `request` shows its prompt."""
    cg, _ = frameworks()
    fn = getattr(
        cg, "CGRequestScreenCaptureAccess" if request else "CGPreflightScreenCaptureAccess", None
    )
    if fn is None:  # before macOS 10.15 there was no such permission
        return True, "screen capture (no permission needed on this macOS)"
    fn.argtypes, fn.restype = [], ctypes.c_bool
    if fn():
        return True, "Screen Recording allowed"
    return False, (
        f"macOS Screen Recording is off for {host_app()}: turn it on in System Settings > "
        "Privacy & Security > Screen & System Audio Recording, then quit and reopen it"
    )
