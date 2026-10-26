"""Streaming bite detector: feed audio chunks, get a BiteEvent when a trigger onset is heard."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from wowfish.audio import AudioChunk
from wowfish.audio.features import DEFAULT_MEL, MelParams, log_mel
from wowfish.audio.fingerprint import Fingerprint, score_sequence


@dataclass(frozen=True)
class ClipScore:
    name: str
    score: float
    t: float  # stream seconds of the window's onset start


@dataclass(frozen=True)
class BiteEvent:
    clip: str
    score: float
    t: float  # stream seconds of the estimated onset start


PEAK_HOLD_S = 0.02  # after the first frame over the threshold, wait this long for the peak


class BiteDetector:
    def __init__(
        self,
        fingerprints: Sequence[Fingerprint],
        p: MelParams = DEFAULT_MEL,
        *,
        trigger_clips: Sequence[str],
        threshold: float = 0.66,
        refractory_s: float = 3.0,
        min_consecutive: int = 1,
    ) -> None:
        self.fingerprints = tuple(fingerprints)
        self.p = p
        self.trigger = frozenset(trigger_clips)
        self.threshold = threshold
        self.refractory_s = refractory_s
        self.min_consecutive = min_consecutive
        names = [fp.name for fp in self.fingerprints]
        self._max_len = max(fp.length for fp in self.fingerprints)
        self._pending = np.zeros(0, dtype=np.float32)
        self._feats = np.zeros((0, p.n_mels), dtype=np.float32)
        self._frames_total = 0
        self._run = dict.fromkeys(names, 0)
        # armed[name] = (peak score, onset time of the peak, frames since the peak) while a
        # trigger clip is over the threshold and the event has not been emitted yet
        self._armed: dict[str, tuple[float, float, int]] = {}
        self._hold_frames = max(1, round(PEAK_HOLD_S / p.frame_s))
        self._scores = {name: ClipScore(name, -1.0, 0.0) for name in names}
        self._level_dbfs = -120.0
        self._quiet_until = -math.inf

    def reset(self) -> None:
        """Forget buffered audio, runs, and the refractory period (e.g. right after a cast)."""
        self._pending = np.zeros(0, dtype=np.float32)
        self._feats = np.zeros((0, self.p.n_mels), dtype=np.float32)
        for name in self._run:
            self._run[name] = 0
        self._armed.clear()
        self._quiet_until = -math.inf

    @property
    def last_scores(self) -> tuple[ClipScore, ...]:
        return tuple(self._scores[fp.name] for fp in self.fingerprints)

    @property
    def level_dbfs(self) -> float:
        return self._level_dbfs

    def feed(self, chunk: AudioChunk) -> list[BiteEvent]:
        """Consume a chunk; return every bite heard in it, in time order (usually none or one)."""
        p = self.p
        if chunk.sample_rate != p.sample_rate:
            raise ValueError(
                f"chunk sample rate {chunk.sample_rate} != {p.sample_rate}; "
                "resample in the source, not per chunk"
            )
        samples = np.asarray(chunk.samples, dtype=np.float32)
        if len(samples):
            rms = float(np.sqrt(np.mean(samples * samples)))
            self._level_dbfs = 20.0 * math.log10(rms + 1e-9)
        buf = np.concatenate([self._pending, samples])
        n_new = p.frames_for(len(buf))
        if n_new == 0:
            self._pending = buf
            return []
        new = log_mel(buf[: (n_new - 1) * p.hop + p.n_fft], p)
        self._pending = buf[n_new * p.hop :]
        feats = np.concatenate([self._feats, new])
        self._frames_total += n_new

        # scores[name][j] for the j-th new frame (NaN where fewer than L frames exist yet)
        per_clip: dict[str, np.ndarray] = {}
        for fp in self.fingerprints:
            avail = min(n_new, len(feats) - fp.length + 1)
            col = np.full(n_new, np.nan, dtype=np.float32)
            if avail > 0:
                col[n_new - avail :] = score_sequence(feats[-(avail + fp.length - 1) :], fp)[
                    fp.length - 1 :
                ]
            per_clip[fp.name] = col

        events: list[BiteEvent] = []
        for j in range(n_new):
            frame = self._frames_total - n_new + j  # global index of the window's last frame
            t_end = (frame * p.hop + p.n_fft) / p.sample_rate
            for fp in self.fingerprints:
                name = fp.name
                score = float(per_clip[name][j])
                if math.isnan(score):
                    continue
                t_onset = t_end - fp.length * p.hop / p.sample_rate
                self._scores[name] = ClipScore(name, score, t_onset)
                armed = self._armed.get(name)
                if score < self.threshold:
                    self._run[name] = 0
                    if armed is not None:  # the run ended: report its peak
                        self._emit(events, name, armed, t_end)
                    continue
                self._run[name] += 1
                if armed is not None:
                    peak, peak_onset, since = armed
                    armed = (score, t_onset, 0) if score > peak else (peak, peak_onset, since + 1)
                    self._armed[name] = armed
                    if armed[2] >= self._hold_frames:
                        self._emit(events, name, armed, t_end)
                elif (
                    name in self.trigger
                    and self._run[name] == self.min_consecutive
                    and t_end >= self._quiet_until
                ):
                    self._armed[name] = (score, t_onset, 0)
        self._feats = feats[-(2 * self._max_len) :]
        return events

    def _emit(
        self, events: list[BiteEvent], name: str, armed: tuple[float, float, int], t_end: float
    ) -> None:
        peak, peak_onset, _ = armed
        events.append(BiteEvent(name, peak, peak_onset))
        self._quiet_until = t_end + self.refractory_s
        del self._armed[name]
