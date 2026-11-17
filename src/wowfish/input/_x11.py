"""XTest key presses through python-xlib, and a focus check against the game window."""

from __future__ import annotations

import contextlib
import os
import time
from typing import Any

from wowfish.capture import WindowRef, linux_hyprland
from wowfish.errors import InputUnavailable
from wowfish.input._wayland import (
    hyprland_current_focus,
    hyprland_focus_window,
    hyprland_restore_focus,
)
from wowfish.input.keys import X11_NAMES, is_supported_key, normalize_key

ALT_TAB_STEP_S = 0.05  # between the four key events of an Alt+Tab


def _open_display() -> Any:
    from Xlib import display

    try:
        return display.Display()
    except Exception as e:  # DisplayConnectionError and friends
        raise InputUnavailable(f"X11 input (cannot open display: {e})") from e


def xtest_available() -> tuple[bool, str]:
    """Whether the XTEST extension answers, and which server this is."""
    try:
        d = _open_display()
    except InputUnavailable as e:
        return False, str(e)
    try:
        ok = d.query_extension("XTEST") is not None
    finally:
        d.close()
    server = "Xwayland" if os.environ.get("WAYLAND_DISPLAY") else "X11"
    if not ok:
        return False, f"XTEST extension missing on {server}"
    note = (
        " (GNOME asks for Remote Desktop permission on the first key press; click Allow)"
        if server == "Xwayland"
        else ""
    )
    return True, f"XTest on {server}{note}"


class X11InputDevice:
    def __init__(self) -> None:
        self._d = _open_display()

    def close(self) -> None:
        if self._d is not None:
            self._d.close()
            self._d = None

    def press_key(self, key: str, hold_s: float = 0.06) -> None:
        from Xlib import XK, X
        from Xlib.ext import xtest

        name = normalize_key(key)
        if not is_supported_key(name):
            raise InputUnavailable(f"key {key!r} (not a supported key name)")
        keysym = XK.string_to_keysym(X11_NAMES[name])
        keycode = self._d.keysym_to_keycode(keysym)
        if not keycode:
            raise InputUnavailable(f"key {key!r} (no keycode on this keyboard layout)")
        xtest.fake_input(self._d, X.KeyPress, keycode)
        self._d.sync()
        time.sleep(hold_s)
        xtest.fake_input(self._d, X.KeyRelease, keycode)
        self._d.sync()

    def is_game_focused(self, window: WindowRef) -> bool | None:
        if window.handle is None:
            return None
        try:
            focus = self._d.get_input_focus().focus
        except Exception:
            return None
        if not hasattr(focus, "id"):
            return False  # PointerRoot / None: nothing focused
        w = focus
        for _ in range(64):
            if w.id == window.handle:
                return True
            try:
                parent = w.query_tree().parent
            except Exception:
                return False
            if not hasattr(parent, "id") or parent.id in (0, w.id):
                return False  # reached the root (python-xlib gives an int there)
            w = parent
        return False

    # -- focus ------------------------------------------------------------------------------

    def _root_ids(self, atom_name: str) -> list[int]:
        from Xlib import X

        try:
            root = self._d.screen().root
            prop = root.get_full_property(self._d.intern_atom(atom_name), X.AnyPropertyType)
        except Exception:
            return []
        return [int(v) for v in prop.value] if prop is not None else []

    def _activate(self, window_id: int) -> None:
        """Ask the window manager to focus a window (_NET_ACTIVE_WINDOW, as a pager would: an
        application's own request can be turned down as focus stealing)."""
        from Xlib import X, protocol

        root = self._d.screen().root
        atom = self._d.intern_atom("_NET_ACTIVE_WINDOW")
        event = protocol.event.ClientMessage(
            window=self._d.create_resource_object("window", window_id),
            client_type=atom,
            data=(32, [2, X.CurrentTime, 0, 0, 0]),
        )
        root.send_event(event, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
        self._d.sync()

    def _alt_tab(self) -> None:
        from Xlib import XK, X
        from Xlib.ext import xtest

        alt = self._d.keysym_to_keycode(XK.string_to_keysym("Alt_L"))
        tab = self._d.keysym_to_keycode(XK.string_to_keysym("Tab"))
        for kind, code in (
            (X.KeyPress, alt),
            (X.KeyPress, tab),
            (X.KeyRelease, tab),
            (X.KeyRelease, alt),
        ):
            xtest.fake_input(self._d, kind, code)
            self._d.sync()
            time.sleep(ALT_TAB_STEP_S)

    def current_focus(self) -> object | None:
        if linux_hyprland():
            return hyprland_current_focus()
        return next(iter(self._root_ids("_NET_ACTIVE_WINDOW")), None)

    def focus_window(self, window: WindowRef) -> None:
        if linux_hyprland():
            hyprland_focus_window(window)
        elif window.handle is not None:
            with contextlib.suppress(Exception):
                self._activate(window.handle)

    def restore_focus(self, token: object | None) -> None:
        """Back to the window that had focus. Under Xwayland a native Wayland app cannot be
        named from here (the window manager reports a placeholder), so Alt+Tab goes back to it:
        it was the window in front just before the game."""
        if linux_hyprland():
            hyprland_restore_focus(token)
            return
        if not isinstance(token, int):
            return
        try:
            if token in self._root_ids("_NET_CLIENT_LIST"):
                self._activate(token)
            elif os.environ.get("WAYLAND_DISPLAY"):
                self._alt_tab()
        except Exception:
            pass
