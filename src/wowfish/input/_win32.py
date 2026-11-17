"""SendInput key presses with scan codes (games read those), and a foreground-window check."""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

from wowfish.capture import WindowRef
from wowfish.errors import InputUnavailable
from wowfish.input.keys import VK_CODES, is_supported_key, normalize_key

INPUT_KEYBOARD = 1
KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_KEYUP = 0x0002
VK_MENU = 0x12  # Alt
ULONG_PTR = ctypes.c_size_t


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT), ("pad", ctypes.c_byte * 32)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


def _user32() -> ctypes.WinDLL:  # type: ignore[name-defined]
    if not hasattr(ctypes, "WinDLL"):
        raise InputUnavailable("Windows input on this platform")
    return ctypes.WinDLL("user32", use_last_error=True)


class Win32InputDevice:
    def __init__(self) -> None:
        self._u = _user32()
        self._u.GetForegroundWindow.restype = wintypes.HWND  # unsigned: handles can have bit 31 set
        self._u.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
        self._u.MapVirtualKeyW.restype = wintypes.UINT
        self._u.SendInput.argtypes = [wintypes.UINT, ctypes.c_void_p, ctypes.c_int]
        self._u.SendInput.restype = wintypes.UINT

    def close(self) -> None:
        pass

    def _send(self, scan: int, up: bool) -> None:
        flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0)
        inp = _INPUT(type=INPUT_KEYBOARD)
        inp.u.ki = _KEYBDINPUT(0, scan, flags, 0, 0)
        if self._u.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT)) != 1:
            err = ctypes.get_last_error()
            raise InputUnavailable(f"SendInput (error {err})")

    def press_key(self, key: str, hold_s: float = 0.06) -> None:
        name = normalize_key(key)
        if not is_supported_key(name):
            raise InputUnavailable(f"key {key!r} (not a supported key name)")
        scan = self._u.MapVirtualKeyW(VK_CODES[name], 0)  # MAPVK_VK_TO_VSC
        if not scan:
            raise InputUnavailable(f"key {key!r} (no scan code)")
        self._send(scan, up=False)
        time.sleep(hold_s)
        self._send(scan, up=True)

    def is_game_focused(self, window: WindowRef) -> bool | None:
        if window.handle is None:
            return None
        return (self._u.GetForegroundWindow() or 0) == int(window.handle)

    # -- focus ------------------------------------------------------------------------------

    def _set_foreground(self, hwnd: int) -> None:
        """SetForegroundWindow, and once more after an Alt tap when Windows turns it down: it
        only lets a process that just received input change the foreground window."""
        u = self._u
        u.SetForegroundWindow.argtypes = [wintypes.HWND]
        u.SetForegroundWindow(hwnd)
        if int(u.GetForegroundWindow() or 0) == int(hwnd):
            return
        for flags in (0, KEYEVENTF_KEYUP):
            inp = _INPUT(type=INPUT_KEYBOARD)
            inp.u.ki = _KEYBDINPUT(VK_MENU, 0, flags, 0, 0)
            u.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))
        u.SetForegroundWindow(hwnd)

    def current_focus(self) -> object | None:
        return int(self._u.GetForegroundWindow() or 0) or None

    def focus_window(self, window: WindowRef) -> None:
        if window.handle is not None:
            self._set_foreground(window.handle)

    def restore_focus(self, token: object | None) -> None:
        if isinstance(token, int):
            self._set_foreground(token)
