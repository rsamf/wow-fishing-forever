"""Turns engine events into the words the console prints and the GUI shows, and owns the
engine thread and the setup workers (meter, calibration grab, bite recording). No Tk here."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from wowfish import paths
from wowfish.audio import AudioChunk, AudioUnavailable
from wowfish.config import Config, load_config, save_config
from wowfish.engine import events as ev
from wowfish.engine.state import State
from wowfish.engine.thread import EngineThread
from wowfish.gui import image
from wowfish.input import keys

GAME_TITLE = "World of Warcraft"
BLOCK = 1024


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
    recording: Path


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
        open_capture: Callable[..., Any] | None = None,
        find_window: Callable[..., Any] | None = None,
        get_capturer: Callable[[], Any] | None = None,
    ) -> None:
        from wowfish.audio.devices import GAME, GAME_CHOICE, LoopbackDevice, list_loopback_devices
        from wowfish.audio.devices import open_capture as real_open
        from wowfish.capture import find_window as real_find
        from wowfish.capture import get_capturer as real_capturer
        from wowfish.engine.engine import build_engine as real_build

        self._build_engine = build_engine or real_build
        self._list_devices = list_devices or list_loopback_devices
        self._open_capture = open_capture or real_open
        self._game = GAME
        self._game_choice = GAME_CHOICE
        self._device_type = LoopbackDevice
        self._find_window = find_window or real_find
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

    # -- config -----------------------------------------------------------------------------

    def reload(self) -> None:
        self.cfg = load_config()[0]

    def save_keys(self, cast_key: str, interact_key: str) -> None:
        cast, interact = keys.normalize_key(cast_key), keys.normalize_key(interact_key)
        for name in (cast, interact):
            if not keys.is_supported_key(name):
                raise ValueError(f"unsupported key {name!r}: use a letter, digit, F1-F12, or space")
        save_config(
            replace(self.cfg, game=replace(self.cfg.game, cast_key=cast, interact_key=interact))
        )
        self.reload()

    def audio_devices(self) -> list[Any]:
        """The game's own audio first (default), then the loopback monitors."""
        try:
            monitors = self._list_devices()
        except AudioUnavailable:
            monitors = []
        return [self._game_choice] + [self._device_type(d.id, d.name, False) for d in monitors]

    def default_device(self) -> Any | None:
        found = self.audio_devices()
        return next((d for d in found if d.is_default), found[0] if found else None)

    def save_audio_device(self, selector: str) -> None:
        if selector == self._game:
            audio = replace(self.cfg.audio, capture="game")
        else:
            audio = replace(self.cfg.audio, capture="device", device=selector)
        save_config(replace(self.cfg, audio=audio))
        self.reload()

    def _open_audio(self, selector: str | None) -> Any:
        selector = selector if selector is not None else self.cfg.audio.selector
        window = self._find_window(GAME_TITLE) if selector == self._game else None
        return self._open_capture(selector, pid=window.pid if window else None, title=GAME_TITLE)

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

    def prerequisites(self) -> list[str]:
        """What blocks Start, in plain words; empty when the bot can run."""
        from wowfish.engine.engine import unsupported_platform
        from wowfish.vision.matcher import MIN_TEMPLATE_PX

        reasons: list[str] = []
        problem = unsupported_platform()
        if problem:
            reasons.append(problem)
        window = self._find_window("World of Warcraft")
        if window is None and not problem:
            reasons.append("World of Warcraft window not found; is the game running?")
        size = self.cfg.vision.icon_size_px
        if size > 0 and window is not None:
            px = round(size * min(1.0, self.cfg.vision.match_width / window.width))
            if px < MIN_TEMPLATE_PX:
                reasons.append("the calibrated icon is too small for this window; recalibrate")
        if not self.audio_devices():
            reasons.append("no loopback audio device found")
        return reasons

    # -- the bot ----------------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, minutes: float | None = None) -> None:
        if self.running:
            return
        self.reload()
        try:
            session_s = None if minutes is None else minutes * 60
            engine = self._build_engine(self.cfg, session_s=session_s)
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
            case VerifyDone(report=r, recording=path):
                if r.stands_out:
                    lo = min(b.score for b in r.bites)
                    hi = max(b.score for b in r.bites)
                    return (
                        f"Bite check: {len(r.bites)} bite(s) scoring {lo:.2f}-{hi:.2f}, "
                        f"everything else at most {r.floor:.2f}; suggested threshold "
                        f"{r.threshold:.2f} ({path.name})"
                    )
                return (
                    f"Bite check: no bite stood out (everything at most {r.floor:.2f}); try again"
                )
            case WorkerFailed(what=what, message=message):
                return f"Cannot {what}: {message}"
        return str(item)

    # -- setup workers ----------------------------------------------------------------------

    def grab_and_calibrate(self) -> None:
        threading.Thread(
            target=self._calibrate_worker, name="wowfish-calibrate", daemon=True
        ).start()

    def _calibrate_worker(self) -> None:
        from wowfish.vision.calibrate import calibrate_icon_scale
        from wowfish.vision.template import load_icon_templates

        try:
            window = self._find_window("World of Warcraft")
            if window is None:
                raise RuntimeError("World of Warcraft window not found; is the game running?")
            cap = self._get_capturer()
            cap.open(window)
            try:
                frame = cap.grab()
            finally:
                cap.close()
            cal = calibrate_icon_scale(
                frame, load_icon_templates(), match_width=self.cfg.vision.match_width
            )
            thumb, scale = image.thumbnail(frame)
            side = max(2, round(cal.icon_size_px * scale))
            box = (round(cal.x * scale - side / 2), round(cal.y * scale - side / 2), side, side)
            png = image.bgr_to_png_bytes(image.with_box(thumb, box) if cal.valid else thumb)
            self._results.put(CalibrateDone(cal, png))
        except Exception as e:
            self._results.put(WorkerFailed("calibrate", str(e)))

    def record_and_verify(self, seconds: float = 60.0, selector: str | None = None) -> None:
        threading.Thread(
            target=self._verify_worker, args=(seconds, selector), name="wowfish-verify", daemon=True
        ).start()

    def _verify_worker(self, seconds: float, selector: str | None) -> None:
        from wowfish.audio.clips import build_fingerprints
        from wowfish.audio.features import DEFAULT_MEL
        from wowfish.audio.verify import bite_report, score_recording
        from wowfish.audio.wav import write_wav

        try:
            source = self._open_audio(selector)
            chunks: list[np.ndarray] = []
            frames = 0
            with source:
                while frames < seconds * source.sample_rate:
                    samples = source.read(BLOCK)
                    chunks.append(samples)
                    frames += len(samples)
                    if getattr(source, "exhausted", False):
                        break
            y = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
            folder = paths.config_file().parent / "recordings"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / time.strftime("bite-%Y%m%d-%H%M%S.wav")
            write_wav(path, y, source.sample_rate)
            fps = build_fingerprints()
            scores = score_recording(y, source.sample_rate, fps, DEFAULT_MEL)
            report = bite_report(scores, fps, DEFAULT_MEL)
            self._results.put(VerifyDone(report, path))
        except Exception as e:
            self._results.put(WorkerFailed("verify the bite", str(e)))

    @property
    def level_dbfs(self) -> float:
        return self._meter.level_dbfs if self._meter is not None else -120.0

    @property
    def last_scores(self) -> dict[str, float]:
        return dict(self._meter.scores) if self._meter is not None else {}

    @property
    def meter_running(self) -> bool:
        return self._meter is not None and self._meter.thread.is_alive()

    def start_meter(self, selector: str | None = None) -> None:
        from wowfish.audio.detector import detector_for

        self.stop_meter()
        try:
            source = self._open_audio(selector)
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
