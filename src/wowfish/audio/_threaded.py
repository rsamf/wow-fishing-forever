"""A reader thread in front of a blocking AudioSource so the frame loop never waits on audio.

Real devices block in `read(n)` until n frames exist, and soundcard's WASAPI capture buffer is
only one block long, so a loop that also grabs frames would fall behind or lose audio. Here the
thread reads block after block into a deque; `read(n)` hands back whatever is there (up to n)
without blocking, possibly nothing.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any

import numpy as np


class ThreadedSource:
    def __init__(self, inner: Any, *, block: int = 1024, max_s: float = 4.0) -> None:
        self.inner = inner
        self.sample_rate = int(inner.sample_rate)
        self.name = getattr(inner, "name", type(inner).__name__)
        self.block = block
        self._max_frames = int(max_s * self.sample_rate)
        self._chunks: deque[np.ndarray] = deque()
        self._frames = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="wowfish-audio", daemon=True)

    def __enter__(self) -> ThreadedSource:
        self.inner.__enter__()
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(2.0)
        self.inner.__exit__(*exc)

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                samples = np.asarray(self.inner.read(self.block), dtype=np.float32)
                if not len(samples):
                    continue
                with self._lock:
                    self._chunks.append(samples)
                    self._frames += len(samples)
                    while self._frames > self._max_frames and len(self._chunks) > 1:
                        self._frames -= len(self._chunks.popleft())
                if getattr(self.inner, "exhausted", False):
                    break
        except BaseException as e:  # surface on the next read instead of dying silently
            self.error = e

    def clear(self) -> int:
        """Drop everything buffered so far; return how many frames were dropped."""
        with self._lock:
            dropped = self._frames
            self._chunks.clear()
            self._frames = 0
        return dropped

    def read(self, n_frames: int) -> np.ndarray:
        """Up to n_frames of buffered audio, immediately; empty when nothing has arrived."""
        if self.error is not None:
            raise RuntimeError(f"audio capture stopped: {self.error}") from self.error
        out: list[np.ndarray] = []
        got = 0
        with self._lock:
            while self._chunks and got < n_frames:
                chunk = self._chunks.popleft()
                take = min(len(chunk), n_frames - got)
                out.append(chunk[:take])
                got += take
                self._frames -= take
                if take < len(chunk):
                    self._chunks.appendleft(chunk[take:])
        if not out:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(out)
