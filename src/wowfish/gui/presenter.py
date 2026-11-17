"""Turns engine events into the words the console prints and the GUI shows, and owns the
engine thread. No Tk here."""

from __future__ import annotations

import queue
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from loguru import logger

from wowfish.audio import AudioUnavailable
from wowfish.config import Config, load_config, save_config
from wowfish.engine import events as ev
from wowfish.engine.state import State
from wowfish.engine.thread import EngineThread
from wowfish.gui import image
from wowfish.input import keys


def describe(event: ev.Event) -> str:
    """One line per event; the console and the GUI say the same thing."""
    match event:
        case ev.StateChanged(did=did, next=nxt):
            return f"{did}. {nxt}."
        case ev.IconFound(score=score, where=(x, y)):
            return f"Gear icon detected at ({x}, {y}), score {score:.2f}"
        case ev.Calibrated(size=size, score=score):
            return f"Measured the gear icon at {size} px (score {score:.2f})"
        case ev.IconLost():
            return "Lost sight of the gear icon"
        case ev.BiteHeard(clip=clip, score=score):
            return f"Heard the splash ({score:.2f}, {clip})"
        case ev.KeyPressed(key=key, why="cast"):
            return f"Pressed {key} to cast"
        case ev.KeyPressed(key=key, why="jump"):
            return f"Pressed {key} to jump"
        case ev.KeyPressed(key=key, why="quit"):
            return f"Pressed {key} to quit the game"
        case ev.KeyPressed(key=key, why="lure"):
            return f"Pressed {key} to put the lure on"
        case ev.KeyPressed(key=key):
            return f"Pressed {key} to interact"
        case ev.Caught():
            return "Caught something; the bobber is gone"
        case ev.Timeout(what="icon", after_s=s):
            return f"No gear icon within {s:g} s; recasting"
        case ev.Timeout(what="bite", after_s=s, detail=detail) if detail:
            return f"No splash within {s:g} s; {detail}; recasting"
        case ev.Timeout(what="bite", after_s=s):
            return f"No splash within {s:g} s; recasting"
        case ev.Listening(best_clip=clip, best_score=score, threshold=threshold):
            which = f" ({clip})" if clip else ""  # no clip: everything so far was silence
            return f"Best match so far {max(score, 0.0):.2f}{which}; threshold {threshold:.2f}"
        case ev.Timeout(after_s=s):
            return f"Bobber still there {s:g} s after Interact; recasting"
        case ev.Paused(reason=reason):
            return f"Paused: {reason}"
        case ev.Resumed():
            return "Resumed"
        case ev.Stats():
            return event.as_line()
        case ev.Error(message=message):
            return f"Error: {message}"
        case ev.Stopped(reason=reason):
            return f"Stopped ({reason})"
    return type(event).__name__


# -- what the GUI is told besides engine events ----------------------------------------------------


@dataclass(frozen=True)
class WorkerFailed:
    what: str
    message: str


@dataclass
class ScoreTrace:
    """One cast's per-clip scores for the Scores graph, from the bobber found to bite/timeout."""

    t0: float = 0.0  # engine time the bobber was found; the graph's x is t - t0
    points: list[tuple[float, dict[str, float]]] = field(default_factory=list)  # (t, peaks)
    threshold: float | None = None  # what the running detector compares against
    bite: tuple[float, str, float] | None = None  # (t, clip, score) when the bot heard it
    timed_out: bool = False
    version: int = 0  # bumped on every change, so the graph redraws only when needed


@dataclass
class View:
    state: str
    did: str
    next: str
    stats: str
    thumbnail_png: bytes | None
    running: bool
    new_lines: list[str] = field(default_factory=list)
    log_events: list[object] = field(default_factory=list)
    trace: ScoreTrace = field(default_factory=ScoreTrace)


class Presenter:
    def __init__(
        self,
        *,
        build_engine: Callable[..., Any] | None = None,
        list_devices: Callable[[], list[Any]] | None = None,
    ) -> None:
        from wowfish.audio.devices import audio_follows_window, list_loopback_devices
        from wowfish.engine.engine import build_engine as real_build

        self._build_engine = build_engine or real_build
        self._list_devices = list_devices or list_loopback_devices
        self.audio_follows_window: bool = audio_follows_window()

        self.cfg: Config = load_config()[0]
        self._results: queue.Queue[object] = queue.Queue()
        self._thread: EngineThread | None = None
        self.state = State.IDLE.name
        self.did = "Not started"
        self.next = "Press Start when the game is at the water"
        self.stats = ev.Stats(0.0).as_line()
        self.thumbnail_png: bytes | None = None
        self._before_pause: tuple[str, str, str] | None = None
        self.trace = ScoreTrace()

    # -- config -----------------------------------------------------------------------------

    def reload(self) -> None:
        self.cfg = load_config()[0]

    def save_keys(
        self, cast_key: str, interact_key: str, quit_key: str = "", lure_key: str = ""
    ) -> None:
        cast, interact = keys.normalize_key(cast_key), keys.normalize_key(interact_key)
        quit_, lure = keys.normalize_key(quit_key), keys.normalize_key(lure_key)  # "" = none
        for name in (cast, interact, *(k for k in (quit_, lure) if k)):
            if not keys.is_supported_key(name):
                raise ValueError(f"unsupported key {name!r}: use a letter, digit, F1-F12, or space")
        self.reload()  # pick up what is in the file now
        game = replace(
            self.cfg.game, cast_key=cast, interact_key=interact, quit_key=quit_, lure_key=lure
        )
        save_config(replace(self.cfg, game=game))
        self.reload()

    def save_session_minutes(self, minutes: int) -> None:
        if minutes < 0:
            raise ValueError("the time limit must be 0 (forever) or more minutes")
        self.reload()  # pick up what is in the file now
        if minutes != self.cfg.game.session_minutes:  # saving rewrites the file; skip if unchanged
            save_config(replace(self.cfg, game=replace(self.cfg.game, session_minutes=minutes)))
            self.reload()

    def save_background(self, on: bool) -> None:
        """Whether the bot fishes while another window has focus; used from the next Start."""
        self.reload()  # pick up what is in the file now
        if on != self.cfg.run.background:
            save_config(replace(self.cfg, run=replace(self.cfg.run, background=on)))
            self.reload()

    def audio_devices(self) -> list[Any]:
        """macOS cannot capture one app, so there it is the input devices, BlackHole first
        (default). Elsewhere the bot hears the picked window's app and there is nothing to choose.
        """
        if self.audio_follows_window:
            return []
        try:
            return list(self._list_devices())
        except AudioUnavailable:
            return []

    def default_device(self) -> Any | None:
        """The device to preselect: a virtual one, or none at all. Never the microphone."""
        return next((d for d in self.audio_devices() if d.is_default), None)

    def save_audio_device(self, device: str) -> None:
        """macOS: the input device the game's sound is routed into ("" = BlackHole or the like)."""
        save_config(replace(self.cfg, audio=replace(self.cfg.audio, device=device)))
        self.reload()

    def save_bite_threshold(self, value: str) -> float:
        """Check and save the bite threshold typed on the Setup tab; the bot uses it from the
        next Start. Raises ValueError with a plain message when it is not a number in (0, 1]."""
        try:
            threshold = float(value)
        except ValueError:
            raise ValueError(f"{value.strip() or 'empty'} is not a number") from None
        if not 0.0 < threshold <= 1.0:
            raise ValueError("must be more than 0 and at most 1")
        self.reload()  # pick up what is in the file now
        save_config(replace(self.cfg, audio=replace(self.cfg.audio, bite_threshold=threshold)))
        self.reload()
        return threshold

    def preflight(self, request: bool = False) -> list[str]:
        """What rules out even looking at windows: the platform, or a missing macOS permission.

        `request` also asks macOS for missing permissions, which shows its prompt.
        """
        from wowfish.engine.engine import missing_permissions, unsupported_platform

        problem = unsupported_platform()
        return ([problem] if problem else []) + missing_permissions(request)

    def prerequisites(self, request: bool = False) -> list[str]:
        """What blocks Start before a window is picked, in plain words; empty when the bot can
        run. Start passes `request`; the check when the window opens does not."""
        from wowfish.audio.devices import mac_device_name

        reasons = self.preflight(request)
        if not self.audio_follows_window:
            try:
                mac_device_name(self.cfg.audio.device)
            except AudioUnavailable as e:
                reasons.append(str(e))
        return reasons

    # -- the bot ----------------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, window: Any, minutes: float | None = None) -> None:
        if self.running:
            return
        self.reload()
        try:
            session_s = None if minutes is None else minutes * 60
            engine = self._build_engine(self.cfg, window, session_s=session_s)
        except Exception as e:
            self._results.put(WorkerFailed("start", str(e)))
            return
        self._thread = EngineThread(engine)
        self._thread.start()
        self.state = State.IDLE.name
        self.did = "Started"
        self.next = "Casting"

    def stop(self) -> None:
        if self._thread is not None:
            self._thread.stop()

    def shutdown(self) -> None:
        if self._thread is not None:
            self._thread.stop(timeout=3.0)
            self._thread = None

    def poll(self) -> View:
        lines: list[str] = []
        items: list[object] = []
        if self._thread is not None:
            q = self._thread.events
            while True:
                try:
                    event = q.get_nowait()
                except queue.Empty:
                    break
                if isinstance(event, ev.Scores):  # the graph's feed: not a log line
                    self._add_scores(event)
                    continue
                items.append(event)
                lines.append(describe(event))
                logger.debug(lines[-1])
                self._apply(event)
            if not self._thread.is_alive() and q.empty():
                self._thread = None
        while True:
            try:
                item = self._results.get_nowait()
            except queue.Empty:
                break
            items.append(item)
            lines.append(self._describe_result(item))
        return View(
            self.state, self.did, self.next, self.stats, self.thumbnail_png, self.running,
            lines, items, self.trace,
        )  # fmt: skip

    def _add_scores(self, event: ev.Scores) -> None:
        self.trace.points.append((event.t, event.peaks))
        self.trace.threshold = event.threshold
        self.trace.version += 1

    def _apply(self, event: ev.Event) -> None:
        match event:
            case ev.StateChanged(state=state, did=did, next=nxt):
                self.state, self.did, self.next = state.name, did, nxt
            case ev.IconFound(thumbnail=thumb, box=box):
                self.thumbnail_png = image.bgr_to_png_bytes(image.with_box(thumb, box))
                self.trace = ScoreTrace(
                    t0=event.t, threshold=self.trace.threshold, version=self.trace.version + 1
                )
            case ev.BiteHeard(clip=clip, score=score):
                self.trace.bite = (event.t, clip, score)
                self.trace.version += 1
            case ev.Timeout(what="bite"):
                self.trace.timed_out = True
                self.trace.version += 1
            case ev.Stats():
                self.stats = event.as_line()
            case ev.Paused(reason=reason):
                self._before_pause = (self.state, self.did, self.next)
                self.state, self.did, self.next = State.PAUSED.name, f"Paused: {reason}", "Waiting"
            case ev.Resumed(state=state):
                if self._before_pause is not None and self._before_pause[0] == state.name:
                    self.state, self.did, self.next = self._before_pause
                else:
                    self.state, self.did, self.next = state.name, "Resumed", "Carrying on"
                self._before_pause = None
            case ev.Stopped(reason=reason):
                self.state, self.did, self.next = State.STOPPED.name, f"Stopped ({reason})", "Idle"
            case ev.Error(message=message):
                self.did = f"Error: {message}"

    def _describe_result(self, item: object) -> str:
        match item:
            case WorkerFailed(what=what, message=message):
                return f"Cannot {what}: {message}"
        return str(item)
