"""Read a window's own pixels rather than the screen area it sits on, so the game is still seen
while another window covers it (Windows PrintWindow, macOS CGWindowListCreateImage)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from wowfish.capture import Box, WindowRef


class WindowPixelsCapturer:
    def __init__(self, snapshot: Callable[[int], np.ndarray | None]) -> None:
        self._snapshot = snapshot  # window handle -> the whole window as BGR, or None
        self._window: WindowRef | None = None

    def open(self, window: WindowRef) -> None:
        if window.handle is None:
            raise ValueError("reading a window's own pixels needs WindowRef.handle")
        self._window = window

    def grab(self, box: Box | None = None) -> np.ndarray:
        if self._window is None or self._window.handle is None:
            raise RuntimeError("capturer is not open")
        frame = self._snapshot(self._window.handle)
        if frame is None:
            raise RuntimeError(
                f"cannot read {self._window.title!r}; is it minimized? Keep the game window open"
            )
        if box is None:
            return frame
        x, y, w, h = box
        return np.ascontiguousarray(frame[y : y + h, x : x + w])

    def close(self) -> None:
        self._window = None
