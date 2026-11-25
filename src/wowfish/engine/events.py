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
    score: float
    size: int  # icon side in window pixels
    where: tuple[int, int]  # icon centre in window pixels


@dataclass(frozen=True)
class Calibrated(Event):
    size: int  # icon side in window pixels, measured on this frame
    score: float


@dataclass(frozen=True)
class Preview(Event):
    """The game window as the bot sees it, every `preview_s`, for the GUI's live picture (not
    logged)."""

    frame: np.ndarray  # the whole window, <= PREVIEW_WIDTH px wide, BGR
    # x, y, w, h in `frame` pixels: the gear being followed, or, while one is looked for, the
    # best match so far; None when there is neither
    box: tuple[int, int, int, int] | None
    score: float | None = None  # the best match's score; None when `box` is the followed gear
    scan: float | None = None  # how far a scan for the gear's size is, 0..1; None = none running


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
    why: str  # "cast" | "bite" | "jump" | "quit" | "lure"


@dataclass(frozen=True)
class Caught(Event):
    pass


@dataclass(frozen=True)
class Timeout(Event):
    what: str  # "icon" | "bite" | "loot"
    after_s: float
    # so a miss explains itself. "bite": the best match heard while waiting; "icon": a scan that
    # outlasted the timeout
    detail: str = ""


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
