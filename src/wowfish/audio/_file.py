"""Replay a WAV file as an AudioSource, for tests and `listen --from`."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wowfish.audio.wav import read_wav, resample


class WavFileSource:
    def __init__(self, path: Path | str, *, target_rate: int = 22050) -> None:
        y, rate = read_wav(path)
        self.samples = resample(y, rate, target_rate)
        self.sample_rate = target_rate
        self._pos = 0

    def __enter__(self) -> WavFileSource:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    @property
    def exhausted(self) -> bool:
        return self._pos >= len(self.samples)

    def read(self, n_frames: int) -> np.ndarray:
        out = np.zeros(n_frames, dtype=np.float32)
        part = self.samples[self._pos : self._pos + n_frames]
        out[: len(part)] = part
        self._pos += n_frames
        return out
