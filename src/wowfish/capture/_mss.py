"""Windows/macOS: grab a window area with mss (window discovery arrives with the runtime)."""

from __future__ import annotations

from typing import Any

import numpy as np

from wowfish.capture import Box, WindowRef


def _mss_instance() -> Any:
    import mss

    factory = getattr(mss, "MSS", None) or mss.mss
    return factory()


class MssCapturer:
    def __init__(self) -> None:
        self._sct: Any = None
        self._window: WindowRef | None = None

    def open(self, window: WindowRef) -> None:
        self._sct = _mss_instance()
        self._window = window

    def grab(self, box: Box | None = None) -> np.ndarray:
        if self._window is None:
            raise RuntimeError("capturer is not open")
        x, y, w, h = box or (0, 0, self._window.width, self._window.height)
        area = {"left": self._window.left + x, "top": self._window.top + y, "width": w, "height": h}
        shot = self._sct.grab(area)
        return np.ascontiguousarray(np.asarray(shot)[:, :, :3])  # BGRA -> BGR

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
        self._sct = None
        self._window = None
