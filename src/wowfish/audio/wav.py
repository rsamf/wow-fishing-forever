"""WAV files through the standard library only: no libsndfile in the shipped binary."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def read_wav(path: Path | str) -> tuple[np.ndarray, int]:
    """Return (float32 mono in [-1, 1], sample_rate). Multi-channel input is averaged."""
    with wave.open(str(path), "rb") as w:
        channels, width = w.getnchannels(), w.getsampwidth()
        rate, n = w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 2:
        y = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif width == 1:
        y = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        y = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        high = b[:, 2].astype(np.int8).astype(np.int32) << 16
        y = (b[:, 0] | (b[:, 1] << 8) | high).astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"unsupported WAV sample width {width} bytes in {path}")
    if channels > 1:
        y = y.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(y, dtype=np.float32), rate


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear resampling. Adequate here: the features stop at 8 kHz on broadband noise."""
    if src_rate == dst_rate:
        return samples
    n_out = round(len(samples) * dst_rate / src_rate)
    x_old = np.arange(len(samples), dtype=np.float64)
    x_new = np.linspace(0, len(samples) - 1, n_out, dtype=np.float64)
    return np.interp(x_new, x_old, samples).astype(np.float32)
