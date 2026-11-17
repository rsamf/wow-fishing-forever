from __future__ import annotations

from wowfish.capture import WindowRef
from wowfish.errors import InputUnavailable


class StubInputDevice:
    def press_key(self, key: str, hold_s: float = 0.06) -> None:
        raise InputUnavailable("input injection on this platform")

    def is_game_focused(self, window: WindowRef) -> bool | None:
        return None

    def current_focus(self) -> object | None:
        return None

    def focus_window(self, window: WindowRef) -> None:
        pass

    def restore_focus(self, token: object | None) -> None:
        pass
