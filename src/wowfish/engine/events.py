"""Everything the engine does or plans goes out as one of these. `t` is seconds since start."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wowfish.engine.state import State


@dataclass(frozen=True)
class Event:
    t: float


@dataclass(frozen=True)
class StateChanged(Event):
    state: State
    did: str
    next: str


@dataclass(frozen=True)
class IconFound(Event):
    thumbnail: np.ndarray  # the full frame fed to the matcher, <= 480 px wide, BGR
    box: tuple[int, int, int, int]  # x, y, w, h in thumbnail pixels
    score: float
    size: int  # icon side in window pixels
    where: tuple[int, int]  # icon centre in window pixels


@dataclass(frozen=True)
class Calibrated(Event):
    size: int  # icon side in window pixels, measured on this frame
    score: float


@dataclass(frozen=True)
class IconLost(Event):
    pass


@dataclass(frozen=True)
class BiteHeard(Event):
    clip: str
    score: float


@dataclass(frozen=True)
class KeyPressed(Event):
    key: str
    why: str  # "cast" | "bite"


@dataclass(frozen=True)
class Caught(Event):
    pass


@dataclass(frozen=True)
class Timeout(Event):
    what: str  # "icon" | "bite" | "loot"
    after_s: float
    detail: str = ""  # for "bite": the best match heard while waiting, so a miss explains itself


@dataclass(frozen=True)
class Listening(Event):
    """Periodic while waiting for the splash: the best match so far and the bar it must clear."""

    best_clip: str  # "" when nothing has scored yet
    best_score: float  # -1.0 when nothing has scored yet
    threshold: float  # the bite_threshold the detector is comparing against


@dataclass(frozen=True)
class Scores(Event):
    """Every SCORES_EVERY_S while waiting for the splash, for the GUI's score graph (not logged)."""

    peaks: dict[str, float]  # each clip's highest score in this slice; -1.0 = silent
    threshold: float  # the bite_threshold the detector is comparing against


@dataclass(frozen=True)
class Paused(Event):
    reason: str


@dataclass(frozen=True)
class Resumed(Event):
    state: State  # the state the engine went back to


@dataclass(frozen=True)
class Stats(Event):
    casts: int = 0
    catches: int = 0
    misses: int = 0
    bites_heard: int = 0
    elapsed_s: float = 0.0

    def as_line(self) -> str:
        m, s = divmod(int(self.elapsed_s), 60)
        return (
            f"casts {self.casts}, catches {self.catches}, misses {self.misses}, "
            f"bites {self.bites_heard}, {m:02d}:{s:02d}"
        )


@dataclass(frozen=True)
class Error(Event):
    message: str


@dataclass(frozen=True)
class Stopped(Event):
    reason: str  # "user" | "session limit" | "error"
