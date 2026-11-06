"""Offline scoring of a recording against every clip, for `verify-bite` and the setup wizard."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from wowfish.audio.detector import ClipScore
from wowfish.audio.features import DEFAULT_MEL, MelParams, log_mel
from wowfish.audio.fingerprint import Fingerprint, score_sequence
from wowfish.audio.wav import resample


def score_recording(
    samples: np.ndarray,
    rate: int,
    fingerprints: Sequence[Fingerprint],
    p: MelParams = DEFAULT_MEL,
) -> dict[str, np.ndarray]:
    y = resample(np.asarray(samples, dtype=np.float32), rate, p.sample_rate)
    feats = log_mel(y, p)
    return {fp.name: score_sequence(feats, fp) for fp in fingerprints}


def _onset_time(frame: int, fp: Fingerprint, p: MelParams) -> float:
    return (frame * p.hop + p.n_fft) / p.sample_rate - fp.length * p.hop / p.sample_rate


def clip_peaks(
    scores: dict[str, np.ndarray], fingerprints: Sequence[Fingerprint], p: MelParams
) -> list[ClipScore]:
    out = []
    for fp in fingerprints:
        s = scores[fp.name]
        i = int(np.argmax(s))
        out.append(ClipScore(fp.name, float(s[i]), _onset_time(i, fp, p)))
    return out


def noise_floor(
    scores_for_clip: np.ndarray, peak_index: int, p: MelParams, exclude_s: float = 2.0
) -> float:
    """Highest score outside ±exclude_s of the peak: what the clip scores on everything else."""
    half = int(exclude_s / p.frame_s)
    masked = scores_for_clip.copy()
    masked[max(0, peak_index - half) : peak_index + half + 1] = -1.0
    return float(masked.max()) if len(masked) else -1.0


FLOOR_MARGIN = 0.05  # suggested threshold sits this far above what everything else scores
MIN_BITE = 0.65  # a peak below this is never counted as a bite (real bites: 0.72-0.94)
MIN_GAP = 0.08  # the weakest bite must clear the next peak below it by this much
PEEL_MIN = 0.45  # peaks below this are not worth listing
MAX_PEAKS = 64


@dataclass(frozen=True)
class BiteReport:
    bites: tuple[ClipScore, ...]  # peaks counted as bites, strongest first
    floor: float  # highest score outside the bites: what everything else scores
    threshold: float  # suggested bite_threshold

    @property
    def stands_out(self) -> bool:
        return bool(self.bites)


def bite_report(
    scores: dict[str, np.ndarray],
    fingerprints: Sequence[Fingerprint],
    p: MelParams,
    *,
    separation_s: float = 2.0,
) -> BiteReport:
    """Peel peaks off the best-of-all-clips score and split them into bites and the rest.

    Any clip can be the bite (the game plays one of several splash sounds), so this works on
    the same best-of-clips score the detector compares with the threshold.

    A recording of a real session holds several bites of different strengths (0.72-0.94 seen)
    above a dense continuum of other game sounds (<= 0.63), so one bite must not count as another
    bite's noise. Candidates are the peaks >= MIN_BITE; the group shrinks from the bottom until
    its weakest member clears the next peak by MIN_GAP. The threshold is the floor + FLOOR_MARGIN.
    """
    members = list(fingerprints)
    stack = np.stack([scores[fp.name] for fp in members])  # (clips, t)
    best = stack.max(axis=0)
    which = stack.argmax(axis=0)
    half = int(separation_s / p.frame_s)
    work = best.copy()
    peaks: list[ClipScore] = []
    while len(peaks) < MAX_PEAKS:
        i = int(np.argmax(work))
        if work[i] < PEEL_MIN:
            break
        fp = members[int(which[i])]
        peaks.append(ClipScore(fp.name, float(best[i]), _onset_time(i, fp, p)))
        work[max(0, i - half) : i + half + 1] = -1.0
    k = sum(1 for c in peaks if c.score >= MIN_BITE)
    while k > 0:
        below = peaks[k].score if k < len(peaks) else float(work.max())
        if peaks[k - 1].score - below >= MIN_GAP:
            break
        k -= 1
    bites = tuple(peaks[:k])
    rest = best.copy()
    for c in bites:
        i = int(np.argmin(np.abs(np.arange(len(best)) * p.frame_s - c.t)))
        rest[max(0, i - half) : i + half + 1] = -1.0
    floor = float(rest.max()) if len(rest) else -1.0
    threshold = float(min(0.9, max(0.5, round(floor + FLOOR_MARGIN, 2))))
    return BiteReport(bites, floor, threshold)
