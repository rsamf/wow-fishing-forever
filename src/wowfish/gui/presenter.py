"""Turns engine events into the words the console prints and the GUI shows, and owns the
engine thread and the setup workers (meter, calibration grab, bite recording). No Tk here."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from loguru import logger

from wowfish.audio import AudioChunk, AudioUnavailable
from wowfish.config import Config, load_config, save_config
from wowfish.engine import events as ev
from wowfish.engine.state import State
from wowfish.engine.thread import EngineThread
from wowfish.gui import image
from wowfish.input import keys

BLOCK = 1024
CALIBRATE_WATCH_S = 30.0  # how long the calibration keeps grabbing frames for the gear icon
CALIBRATE_REST_S = 0.1  # between those grabs, so the watch does not spin a CPU core


def describe(event: ev.Event) -> str:
    """One line per event; the console and the GUI say the same thing."""
    match event:
        case ev.StateChanged(did=did, next=nxt):
            return f"{did}. {nxt}."
        case ev.IconFound(score=score, where=(x, y)):
            return f"Found the bobber at ({x}, {y}), score {score:.2f}"
        case ev.Calibrated(size=size, score=score):
            return f"Measured the gear icon at {size} px (score {score:.2f})"
        case ev.IconLost():
            return "Lost sight of the bobber"
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


# -- results of the setup workers ---------------------------------------------------------------


@dataclass(frozen=True)
class CalibrateDone:
    calibration: Any  # vision.calibrate.Calibration
    thumbnail_png: bytes


@dataclass(frozen=True)
class VerifyDone:
    report: Any  # audio.verify.BiteReport


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
    level_dbfs: float
    last_scores: dict[str, float]
    new_lines: list[str] = field(default_factory=list)
    log_events: list[object] = field(default_factory=list)
    trace: ScoreTrace = field(default_factory=ScoreTrace)
    listening: bool = False  # the meter or the bite recording is reading audio
    verify_status: str = ""  # what the bite recording is doing; empty when idle
    calibrate_status: str = ""  # what the calibration is doing; empty when idle


class _Meter:
    def __init__(self, source: Any, detector: Any) -> None:
        self.source = source
        self.detector = detector
        self.level_dbfs = -120.0
        self.scores: dict[str, float] = {}
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="wowfish-meter", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.thread.join(2.0)

    def _run(self) -> None:
        frames = 0
        with self.source:
            while not self._stop.is_set():
                samples = self.source.read(BLOCK)
                frames += len(samples)
                self.detector.feed(AudioChunk(samples, self.source.sample_rate, frames / 22050))
                self.level_dbfs = self.detector.level_dbfs
                self.scores = {s.name: s.score for s in self.detector.last_scores}
                if getattr(self.source, "exhausted", False):
                    break


class Presenter:
    def __init__(
        self,
        *,
        build_engine: Callable[..., Any] | None = None,
        list_devices: Callable[[], list[Any]] | None = None,
        open_game_audio: Callable[..., Any] | None = None,
        get_capturer: Callable[..., Any] | None = None,
    ) -> None:
        from wowfish.audio.devices import audio_follows_window, list_loopback_devices
        from wowfish.audio.devices import open_game_audio as real_open
        from wowfish.capture import get_capturer as real_capturer
        from wowfish.engine.engine import build_engine as real_build

        self._build_engine = build_engine or real_build
        self._list_devices = list_devices or list_loopback_devices
        self._open_game_audio = open_game_audio or real_open
        self.audio_follows_window: bool = audio_follows_window()
        self._get_capturer = get_capturer or real_capturer

        self.cfg: Config = load_config()[0]
        self._results: queue.Queue[object] = queue.Queue()
        self._thread: EngineThread | None = None
        self._meter: _Meter | None = None
        self.state = State.IDLE.name
        self.did = "Not started"
        self.next = "Press Start when the game is at the water"
        self.stats = ev.Stats(0.0).as_line()
        self.thumbnail_png: bytes | None = None
        self._before_pause: tuple[str, str, str] | None = None
        self.last_calibration: CalibrateDone | None = None
        self.trace = ScoreTrace()
        # Written by the setup workers, read by the GUI tick.
        self.verify_status = ""
        self.calibrate_status = ""
        self._record_level: float | None = None

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
        self.reload()  # a run may have saved its icon measurement since; keep it
        game = replace(
            self.cfg.game, cast_key=cast, interact_key=interact, quit_key=quit_, lure_key=lure
        )
        save_config(replace(self.cfg, game=game))
        self.reload()

    def save_session_minutes(self, minutes: int) -> None:
        if minutes < 0:
            raise ValueError("the time limit must be 0 (forever) or more minutes")
        self.reload()  # a run may have saved its icon measurement since; keep it
        if minutes != self.cfg.game.session_minutes:  # saving rewrites the file; skip if unchanged
            save_config(replace(self.cfg, game=replace(self.cfg.game, session_minutes=minutes)))
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
        found = self.audio_devices()
        return next((d for d in found if d.is_default), found[0] if found else None)

    def save_audio_device(self, device: str) -> None:
        """macOS: the input device the game's sound is routed into ("" = BlackHole or the like)."""
        save_config(replace(self.cfg, audio=replace(self.cfg.audio, device=device)))
        self.reload()

    def _open_audio(self, window: Any, device: str | None) -> Any:
        """The picked window's app; on macOS the input device (the saved one when None)."""
        return self._open_game_audio(window, self.cfg.audio.device if device is None else device)

    def save_calibration(self) -> None:
        if self.last_calibration is None:
            raise ValueError("calibrate first")
        cal = self.last_calibration.calibration
        vision = replace(
            self.cfg.vision,
            icon_size_px=cal.icon_size_px,
            icon_native_size=cal.native_size,
            icon_threshold=cal.suggested_threshold(),
        )
        save_config(replace(self.cfg, vision=vision))
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
        self.reload()  # a running bot may have saved a calibration since this was loaded
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

    def window_problems(self, window: Any) -> list[str]:
        """What blocks Start in the window the user picked."""
        from wowfish.vision.matcher import MIN_TEMPLATE_PX

        size = self.cfg.vision.icon_size_px
        if size > 0:
            px = round(size * min(1.0, self.cfg.vision.match_width / window.width))
            if px < MIN_TEMPLATE_PX:
                return ["the calibrated icon is too small for this window; recalibrate"]
        return []

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
        self.stop_meter()
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
            self.level_dbfs, self.last_scores, lines, items, self.trace,
            listening=self.meter_running or self._record_level is not None,
            verify_status=self.verify_status,
            calibrate_status=self.calibrate_status,
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
            case CalibrateDone(calibration=cal):
                self.last_calibration = item
                ok = "ok" if cal.valid else "no gold icon found; cast first and try again"
                return f"Calibration: {cal.icon_size_px} px, score {cal.score:.2f} ({ok})"
            case VerifyDone(report=r):
                if r.stands_out:
                    lo = min(b.score for b in r.bites)
                    hi = max(b.score for b in r.bites)
                    return (
                        f"Bite check: {len(r.bites)} bite(s) scoring {lo:.2f}-{hi:.2f}, "
                        f"everything else at most {r.floor:.2f}; suggested threshold "
                        f"{r.threshold:.2f}"
                    )
                return (
                    f"Bite check: no bite stood out (everything at most {r.floor:.2f}); try again"
                )
            case WorkerFailed(what=what, message=message):
                return f"Cannot {what}: {message}"
        return str(item)

    # -- setup workers ----------------------------------------------------------------------

    def grab_and_calibrate(self, window: Any) -> None:
        """Watch the window the user picked for the gear icon and measure it."""
        threading.Thread(
            target=self._calibrate_worker, args=(window,), name="wowfish-calibrate", daemon=True
        ).start()

    def _calibrate_worker(self, window: Any) -> None:
        from wowfish.vision.calibrate import calibrate_icon_scale
        from wowfish.vision.template import load_icon_templates

        result: object
        try:
            cap = self._get_capturer(window)
            cap.open(window)
            templates = load_icon_templates()
            deadline = time.monotonic() + CALIBRATE_WATCH_S
            try:
                while True:  # until a frame shows the gear, so there is time to go and cast
                    left = max(0.0, deadline - time.monotonic())
                    self.calibrate_status = (
                        f"Watching {window.title} for the gold gear icon ({left:.0f} s left). "
                        "Cast now and keep the game on screen."
                    )
                    frame = cap.grab()
                    cal = calibrate_icon_scale(
                        frame, templates, match_width=self.cfg.vision.match_width
                    )
                    if cal.valid or time.monotonic() >= deadline:
                        break
                    time.sleep(CALIBRATE_REST_S)
            finally:
                cap.close()
            thumb, scale = image.thumbnail(frame)
            side = max(2, round(cal.icon_size_px * scale))
            box = (round(cal.x * scale - side / 2), round(cal.y * scale - side / 2), side, side)
            png = image.bgr_to_png_bytes(image.with_box(thumb, box) if cal.valid else thumb)
            result = CalibrateDone(cal, png)
        except Exception as e:
            result = WorkerFailed("calibrate", str(e))
        finally:
            self.calibrate_status = ""  # before the result, so the result's words stay up
        self._results.put(result)

    def record_and_verify(
        self, seconds: float = 60.0, window: Any = None, device: str | None = None
    ) -> None:
        threading.Thread(
            target=self._verify_worker,
            args=(seconds, window, device),
            name="wowfish-verify",
            daemon=True,
        ).start()

    def _verify_worker(self, seconds: float, window: Any, device: str | None) -> None:
        from wowfish.audio.clips import build_fingerprints
        from wowfish.audio.features import DEFAULT_MEL
        from wowfish.audio.verify import bite_report, score_recording

        result: object
        try:
            self.verify_status = "Opening the game's audio..."
            source = self._open_audio(window, device)
            chunks: list[np.ndarray] = []
            frames = 0
            with source:
                total = int(seconds * source.sample_rate)
                while frames < total:
                    samples = source.read(BLOCK)
                    chunks.append(samples)
                    frames += len(samples)
                    if len(samples):
                        rms = float(np.sqrt(np.mean(np.square(samples))))
                        self._record_level = 20.0 * np.log10(rms + 1e-9)
                    left = (total - frames) / source.sample_rate
                    self.verify_status = (
                        f"Recording: {left:.0f} s left. "
                        "Fish by hand in the game until a fish bites."
                    )
                    if getattr(source, "exhausted", False):
                        break
            self._record_level = None
            self.verify_status = "Recorded; looking for the splash..."
            y = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
            fps = build_fingerprints()
            scores = score_recording(y, source.sample_rate, fps, DEFAULT_MEL)
            report = bite_report(scores, fps, DEFAULT_MEL)
            result = VerifyDone(report)
        except Exception as e:
            result = WorkerFailed("verify the bite", str(e))
        finally:
            self._record_level = None
            self.verify_status = ""  # before the result, so the result's words stay up
        self._results.put(result)

    @property
    def level_dbfs(self) -> float:
        if self._record_level is not None:
            return self._record_level
        return self._meter.level_dbfs if self._meter is not None else -120.0

    @property
    def last_scores(self) -> dict[str, float]:
        return dict(self._meter.scores) if self._meter is not None else {}

    @property
    def meter_running(self) -> bool:
        return self._meter is not None and self._meter.thread.is_alive()

    def start_meter(self, window: Any = None, device: str | None = None) -> None:
        from wowfish.audio.detector import detector_for

        self.stop_meter()
        try:
            source = self._open_audio(window, device)
        except Exception as e:
            self._results.put(WorkerFailed("open the audio device", str(e)))
            return
        detector = detector_for(self.cfg.audio)
        self._meter = _Meter(source, detector)
        self._meter.start()

    def stop_meter(self) -> None:
        if self._meter is not None:
            self._meter.stop()
            self._meter = None
