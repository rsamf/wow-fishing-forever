"""macOS: key presses as Quartz keyboard events, and whether the game is the front app.

Posting events needs the Accessibility permission. Without it macOS drops them silently, so
press_key checks first and says where to turn it on.
"""

from __future__ import annotations

import ctypes
import sys
import time
from functools import cache
from typing import Any

from wowfish.capture import WindowRef, _quartz
from wowfish.errors import InputUnavailable, NotSupported, PermissionMissing
from wowfish.input.keys import MAC_KEYCODES, is_supported_key, normalize_key

HID_EVENT_TAP = 0  # kCGHIDEventTap: as if from the keyboard
HID_SYSTEM_STATE = 1  # kCGEventSourceStateHIDSystemState
NON_COALESCED = 0x100  # kCGEventFlagMaskNonCoalesced, the only flag a plain key press carries
FOCUS_REQUEST_TIMEOUT_S = 0.25  # an app that is not responding must not stall the engine


@cache
def _app_services() -> Any:
    if sys.platform != "darwin":
        raise InputUnavailable("macOS input outside macOS")
    try:
        lib = ctypes.CDLL(_quartz.APPLICATION_SERVICES)
    except OSError as e:
        raise InputUnavailable(f"ApplicationServices ({e})") from e
    vp = ctypes.c_void_p
    for fn, args, res in (
        (lib.AXIsProcessTrusted, [], ctypes.c_bool),
        (lib.AXIsProcessTrustedWithOptions, [vp], ctypes.c_bool),
        (lib.AXUIElementCreateApplication, [ctypes.c_int], vp),
        (lib.AXUIElementSetMessagingTimeout, [vp, ctypes.c_float], ctypes.c_int32),
        (lib.AXUIElementSetAttributeValue, [vp, vp, vp], ctypes.c_int32),
    ):
        fn.argtypes, fn.restype = args, res
    return lib


def _trusted_with_prompt(lib: Any) -> bool:
    _, cf = _quartz.frameworks()
    keys = (ctypes.c_void_p * 1)(_quartz.constant(lib, "kAXTrustedCheckOptionPrompt"))
    values = (ctypes.c_void_p * 1)(_quartz.constant(cf, "kCFBooleanTrue"))
    callbacks = (
        ctypes.addressof(ctypes.c_char.in_dll(cf, "kCFTypeDictionaryKeyCallBacks")),
        ctypes.addressof(ctypes.c_char.in_dll(cf, "kCFTypeDictionaryValueCallBacks")),
    )
    options = cf.CFDictionaryCreate(
        None,
        ctypes.cast(keys, ctypes.c_void_p),
        ctypes.cast(values, ctypes.c_void_p),
        1,
        *callbacks,
    )
    try:
        return bool(lib.AXIsProcessTrustedWithOptions(options))
    finally:
        if options:
            cf.CFRelease(options)


def accessibility_available(request: bool = False) -> tuple[bool, str]:
    """Whether macOS lets this process post key presses; `request` shows its prompt."""
    lib = _app_services()
    if _trusted_with_prompt(lib) if request else lib.AXIsProcessTrusted():
        return True, "Quartz key events (Accessibility allowed)"
    return False, (
        f"macOS Accessibility is off for {_quartz.host_app()}: turn it on in System Settings > "
        "Privacy & Security > Accessibility"
    )


class QuartzInputDevice:
    def __init__(self) -> None:
        try:
            self._cg, self._cf = _quartz.frameworks()
        except NotSupported as e:
            raise InputUnavailable(f"Quartz key events ({e})") from e
        self._source = self._cg.CGEventSourceCreate(HID_SYSTEM_STATE)

    def close(self) -> None:
        if self._source:
            self._cf.CFRelease(self._source)
            self._source = None

    def _post(self, code: int, down: bool) -> None:
        event = self._cg.CGEventCreateKeyboardEvent(self._source, code, down)
        if not event:
            raise InputUnavailable("Quartz key events (CGEventCreateKeyboardEvent failed)")
        try:
            self._cg.CGEventSetFlags(event, NON_COALESCED)  # no Shift/Cmd held by the user
            self._cg.CGEventPost(HID_EVENT_TAP, event)
        finally:
            self._cf.CFRelease(event)

    def press_key(self, key: str, hold_s: float = 0.06) -> None:
        name = normalize_key(key)
        if not is_supported_key(name):
            raise InputUnavailable(f"key {key!r} (not a supported key name)")
        ok, detail = accessibility_available()
        if not ok:
            raise PermissionMissing(detail)
        code = MAC_KEYCODES[name]
        self._post(code, True)
        time.sleep(hold_s)
        self._post(code, False)

    def is_game_focused(self, window: WindowRef) -> bool:
        # Quartz posts to the active app. A failed lookup must not let a key through.
        return window.pid is not None and self.current_focus() == window.pid

    # -- focus ------------------------------------------------------------------------------

    def _activate(self, pid: object | None) -> None:
        if not isinstance(pid, int) or pid <= 0:
            return
        lib = _app_services()
        app = lib.AXUIElementCreateApplication(pid)
        if not app:
            return
        frontmost = None
        try:
            # macOS 14+ ignores NSApplicationActivateIgnoringOtherApps. Use the Accessibility
            # grant we already have to request focus even when wowfish itself is in the background.
            frontmost = self._cf.CFStringCreateWithCString(None, b"AXFrontmost", _quartz.UTF8)
            if frontmost:
                lib.AXUIElementSetMessagingTimeout(app, FOCUS_REQUEST_TIMEOUT_S)
                lib.AXUIElementSetAttributeValue(
                    app, frontmost, _quartz.constant(self._cf, "kCFBooleanTrue")
                )  # the engine checks whether focus actually moved
        finally:
            if frontmost:
                self._cf.CFRelease(frontmost)
            self._cf.CFRelease(app)

    def current_focus(self) -> int | None:
        try:
            return _quartz.frontmost_pid()
        except OSError:
            return None

    def focus_window(self, window: WindowRef) -> None:
        self._activate(window.pid)

    def restore_focus(self, token: object | None) -> None:
        self._activate(token)
