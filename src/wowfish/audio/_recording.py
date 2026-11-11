"""An AudioSource that saves everything read through it to a WAV when it closes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from wowfish.audio.wav import write_wav


class RecordingSource:
    def __init__(self, inner: Any, path: Path) -> None:
        self.inner = inner
        self.path = Path(path)
        self.sample_rate = int(inner.sample_rate)
        self.name = getattr(inner, "name", type(inner).__name__)
        self._chunks: list[np.ndarray] = []

    def __enter__(self) -> RecordingSource:
        self.inner.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            self.inner.__exit__(*exc)
        finally:
            if self._chunks:
                write_wav(self.path, np.concatenate(self._chunks), self.sample_rate)

    def clear(self) -> int:
        """Drop the inner source's backlog; the recording keeps only what the engine heard."""
        clear = getattr(self.inner, "clear", None)
        return clear() if clear is not None else 0

    def read(self, n_frames: int) -> np.ndarray:
        samples = self.inner.read(n_frames)
        if len(samples):
            self._chunks.append(np.asarray(samples, dtype=np.float32))
        return samples
