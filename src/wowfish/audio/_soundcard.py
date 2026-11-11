"""AudioSource over a `soundcard` loopback microphone (WASAPI loopback, Pulse/PipeWire monitor)."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np


class SoundcardSource:
    def __init__(self, mic: Any, sample_rate: int = 22050, blocksize: int = 1024) -> None:
        self.mic = mic
        self.name = str(mic.name)
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self._rec: Any = None

    def __enter__(self) -> SoundcardSource:
        # CoreAudio only takes a buffer inside the device's range (BlackHole: 15-512 frames) and
        # soundcard raises rather than clamp. The buffer only sets latency; reads are any length.
        blocksize = None if sys.platform == "darwin" else self.blocksize
        self._rec = self.mic.recorder(samplerate=self.sample_rate, channels=1, blocksize=blocksize)
        self._rec.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._rec is not None:
            self._rec.__exit__(*exc)
            self._rec = None

    def read(self, n_frames: int) -> np.ndarray:
        data = np.asarray(self._rec.record(numframes=n_frames), dtype=np.float32)
        if data.ndim == 2:
            data = data[:, 0] if data.shape[1] == 1 else data.mean(axis=1)
        return np.ascontiguousarray(data, dtype=np.float32)
