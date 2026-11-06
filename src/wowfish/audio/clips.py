"""The bundled reference clips and their fingerprints."""

from __future__ import annotations

import numpy as np

from wowfish.assets import CLIP_NAMES, clip_path
from wowfish.audio.features import DEFAULT_MEL, MelParams
from wowfish.audio.fingerprint import Fingerprint, build_fingerprint
from wowfish.audio.wav import read_wav, resample

ONSET_S = 0.8


def load_reference_clips(rate: int = 22050) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in CLIP_NAMES:
        y, r = read_wav(clip_path(name))
        out[name] = resample(y, r, rate)
    return out


def build_fingerprints(
    p: MelParams = DEFAULT_MEL, onset_s: float = ONSET_S
) -> tuple[Fingerprint, ...]:
    clips = load_reference_clips(p.sample_rate)
    return tuple(build_fingerprint(name, clips[name], p, onset_s) for name in CLIP_NAMES)
