"""Keyboard boundary: press one key into the game and tell whether the game has focus."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from wowfish.capture import WindowRef, linux_x11


@runtime_checkable
class InputDevice(Protocol):
    def press_key(self, key: str, hold_s: float = 0.06) -> None:
        """Press and release a config key name (see input.keys). Raises InputUnavailable."""
        ...

    def is_game_focused(self, window: WindowRef) -> bool | None:
        """True/False when the backend can tell which window has keyboard focus, else None."""
        ...


def get_input_device(window: WindowRef | None = None) -> InputDevice:
    """Key presses for `window`: a native Wayland window needs the compositor's virtual keyboard."""
    import sys

    if window is not None and window.compositor:
        from wowfish.input._wayland import WaylandInputDevice

        return WaylandInputDevice()
    if linux_x11():
        from wowfish.input._x11 import X11InputDevice

        return X11InputDevice()
    if sys.platform == "win32":
        from wowfish.input._win32 import Win32InputDevice

        return Win32InputDevice()
    from wowfish.input._stub import StubInputDevice

    return StubInputDevice()


__all__ = ["InputDevice", "get_input_device"]
