"""The fishing loop. One `step()` per iteration; every collaborator is injectable."""

from __future__ import annotations

import math
import queue
import random
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from wowfish.audio import AudioChunk
from wowfish.audio._threaded import ThreadedSource
from wowfish.capture import Box, WindowRef, find_window, get_capturer, linux_x11
from wowfish.config import Config
from wowfish.engine import events as ev
from wowfish.engine.state import State
from wowfish.vision.matcher import IconMatch
from wowfish.vision.region import Region
from wowfish.vision.search import IconSearch

UNFOCUSED = "World of Warcraft is not the focused window; click into the game"
THUMB_WIDTH = 480
AUDIO_BLOCK = 1024
AUDIO_MAX_S = 2.0  # never read more than this at once; the device buffer may drop older audio
STATS_EVERY_S = 10.0
LISTEN_EVERY_S = 2.0  # while waiting for the splash, say what is heard this often
SCORES_EVERY_S = 0.1  # ...and send each clip's peak score this often, for the GUI's graph
SLEEP_SLICE_S = 0.1
RESWEEP_AFTER_TIMEOUTS = 2  # icon timeouts in a row before the next cast measures the icon again


class FishingEngine:
    def __init__(
        self,
        cfg: Config,
        window: WindowRef,
        *,
        capturer: Any,
        input_device: Any,
        audio_source: Any,
        detector: Any,
        matcher: Any = None,
        matcher_factory: Callable[[int], Any] | None = None,
        calibrate: Callable[..., Any] | None = None,
        on_calibrated: Callable[[Any], None] | None = None,
        find_window: Callable[..., WindowRef | None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        session_s: float | None = None,
        box_interval_s: float = 0.25,
        lost_after: int = 3,
    ) -> None:
        self.cfg = cfg
        self.window = window
        self.capturer = capturer
        self.input = input_device
        self.audio = audio_source
        self.detector = detector
        self._matcher_factory = matcher_factory or (lambda size: matcher)
        self._calibrate = calibrate or _real_calibrate
        self._on_calibrated = on_calibrated or (lambda cal: None)
        self._find_window = find_window
        self.threshold = cfg.vision.icon_threshold
        # The saved size (if any) is only a first guess: every run measures the icon on its
        # first cast, since resolution and UI scale change the icon's size on screen.
        self.matcher = matcher if matcher is not None else None
        if self.matcher is None and cfg.vision.icon_size_px > 0:
            self.matcher = self._matcher_factory(cfg.vision.icon_size_px)
        self.search = IconSearch(self.matcher, self.threshold) if self.matcher else None
        self._sweeping = True
        self._icon_timeouts = 0
        self._clock = clock
        self._sleep = sleep
        self._rng = rng or random.Random()
        self.session_s = cfg.game.session_minutes * 60.0 if session_s is None else session_s
        self.box_interval_s = box_interval_s
        self.lost_after = lost_after

        self.events: queue.Queue[ev.Event] = queue.Queue()
        self.state = State.IDLE
        self.stats = ev.Stats(0.0)
        self.started: float | None = None
        self._stop = False
        self._resume_state = State.IDLE
        self._t_cast = 0.0
        self._t_interact = 0.0
        self._icon_xy: tuple[int, int] | None = None
        self._icon_size = 0
        self._misses = 0
        self._last_box_grab = -math.inf
        self._last_audio: float | None = None
        self._audio_frames = 0
        self._last_stats = 0.0
        self._last_listen = 0.0
        self._last_scores_sent = 0.0
        self._best_heard: tuple[str, float] = ("", -1.0)  # best clip score since the icon was found

    # -- public ------------------------------------------------------------------------------

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        self.capturer.open(self.window)
        try:
            with self.audio:
                while self.state is not State.STOPPED:
                    self.step()
        finally:
            self.capturer.close()

    def step(self) -> None:
        if self.state is State.STOPPED:
            return
        now = self._clock()
        if self.started is None:
            self.started = now
            self._last_audio = now
        if self._stop:
            self._finish("user")
            return
        if self.session_s != 0 and now - self.started >= self.session_s:
            self._finish("session limit")
            return
        try:
            self._step(now)
        except Exception as e:  # a collaborator broke; say so and stop rather than spin
            self._emit(ev.Error(self._t(), f"{type(e).__name__}: {e}"))
            self._finish("error")

    # -- the loop body -----------------------------------------------------------------------

    def _step(self, now: float) -> None:
        focused = self.input.is_game_focused(self.window)
        if focused is False:
            if self.state is not State.PAUSED:
                self._resume_state = self.state
                self.state = State.PAUSED
                self._emit(ev.Paused(self._t(), UNFOCUSED))
            self._drain_audio(now)
            self._sleep(SLEEP_SLICE_S)
            return
        if self.state is State.PAUSED:
            self.state = self._resume_state
            self._emit(ev.Resumed(self._t(), self.state))

        chunks = self._drain_audio(now)
        bites = []
        if self.state is State.WAIT_BITE:
            for chunk in chunks:
                bites.extend(self.detector.feed(chunk))
            self._note_scores()
            self._send_scores(now, flush=bool(bites))  # flush so the graph gets the bite's peak

        if self.state is State.IDLE:
            self._cast(now)
        elif self.state is State.CASTING:
            self._acquire(now)
        elif self.state is State.WAIT_BITE:
            self._wait_bite(now, bites)
        elif self.state is State.INTERACT:
            self._after_interact(now)
        self._maybe_stats(now)

    def _cast(self, now: float) -> None:
        if self._find_window is not None:
            found = self._find_window("World of Warcraft")
            if found is not None and (found.width, found.height) != (
                self.window.width,
                self.window.height,
            ):
                self._sweeping = True  # the window changed size: measure the icon again
            if found is not None:
                self.window = found
        key = self.cfg.game.cast_key
        self._press(key, "cast")
        self.stats = ev.Stats(
            self._t(), self.stats.casts + 1, self.stats.catches, self.stats.misses,
            self.stats.bites_heard, self._elapsed(),
        )  # fmt: skip
        self._t_cast = now
        if self.search is not None:
            self.search.reset()
        limit = self.cfg.run.icon_timeout_s
        self._change(
            State.CASTING,
            f"Pressed {key} to cast",
            f"Looking for the gear icon over the bobber for up to {limit:g} s",
        )

    def _acquire(self, now: float) -> None:
        frame = self.capturer.grab()
        if self.search is not None:  # fast path: the saved or last measured size
            _, m = self.search.step(frame)
            if m.accepted(self.threshold):
                self._icon_timeouts = 0
                self._sweeping = False
                self._found(frame, m)
                return
        if self._sweeping or self.search is None:
            cal = self._calibrate(frame, match_width=self.cfg.vision.match_width)
            if cal.valid:
                self._use_calibration(cal)
                self._icon_timeouts = 0
                self._found(frame, IconMatch(cal.score, cal.x, cal.y, cal.icon_size_px, cal.gold))
                return
        if now - self._t_cast > self.cfg.run.icon_timeout_s:
            self._emit(ev.Timeout(self._t(), "icon", self.cfg.run.icon_timeout_s))
            self._icon_timeouts += 1
            if self._icon_timeouts >= RESWEEP_AFTER_TIMEOUTS:
                self._sweeping = True
            self._miss("No gear icon over the bobber")

    def _use_calibration(self, cal: Any) -> None:
        self.matcher = self._matcher_factory(cal.icon_size_px)
        self.threshold = cal.suggested_threshold()
        self.search = IconSearch(self.matcher, self.threshold)
        self._sweeping = False
        self._emit(ev.Calibrated(self._t(), int(cal.icon_size_px), float(cal.score)))
        self._on_calibrated(cal)

    def _found(self, frame: np.ndarray, m: Any) -> None:
        self._icon_xy = (m.x, m.y)
        self._icon_size = m.size
        self._misses = 0
        self._last_box_grab = self._clock()  # the full frame just confirmed it; look again later
        thumb, box = _thumbnail(frame, m.x, m.y, m.size)
        self._emit(ev.IconFound(self._t(), thumb, box, float(m.score), int(m.size), (m.x, m.y)))
        self.detector.reset()
        self._best_heard = ("", -1.0)
        self._last_listen = self._last_scores_sent = self._clock()
        self._change(
            State.WAIT_BITE,
            f"Found the bobber at ({m.x}, {m.y})",
            f"Waiting for a splash; will press {self.cfg.game.interact_key} when I hear one",
        )

    def _wait_bite(self, now: float, bites: list[Any]) -> None:
        if bites:
            bite = bites[0]
            self.stats = ev.Stats(
                self._t(), self.stats.casts, self.stats.catches, self.stats.misses,
                self.stats.bites_heard + 1, self._elapsed(),
            )  # fmt: skip
            self._emit(ev.BiteHeard(self._t(), bite.clip, float(bite.score)))
            if self.input.is_game_focused(self.window) is False:
                self._resume_state = State.WAIT_BITE
                self.state = State.PAUSED
                self._emit(ev.Paused(self._t(), UNFOCUSED))
                return
            key = self.cfg.game.interact_key
            self._press(key, "bite")
            self._t_interact = now
            self._last_box_grab = -math.inf
            self._change(
                State.INTERACT,
                f"Heard the splash ({bite.score:.2f}), pressed {key}",
                "Waiting for the bobber to go away so Auto Loot can finish",
            )
            return
        seen = self._track(now)
        if seen is False and self._misses >= self.lost_after:
            self._emit(ev.IconLost(self._t()))
            self._miss("Lost sight of the bobber")
            return
        if now - self._t_cast > self.cfg.run.cast_timeout_s:
            clip, score = self._best_heard
            detail = f"best match {score:.2f} ({clip})" if clip else "nothing matched"
            self._emit(ev.Timeout(self._t(), "bite", self.cfg.run.cast_timeout_s, detail))
            self._miss("No splash heard")

    def _after_interact(self, now: float) -> None:
        seen = self._track(now)
        if seen is False:
            self.stats = ev.Stats(
                self._t(), self.stats.casts, self.stats.catches + 1, self.stats.misses,
                self.stats.bites_heard, self._elapsed(),
            )  # fmt: skip
            self._emit(ev.Caught(self._t()))
            self._pause_then_idle("Caught something; the bobber is gone")
            return
        if now - self._t_interact > self.cfg.run.loot_timeout_s:
            self._emit(ev.Timeout(self._t(), "loot", self.cfg.run.loot_timeout_s))
            self._miss("The bobber did not go away after Interact")

    # -- helpers -----------------------------------------------------------------------------

    def _track(self, now: float) -> bool | None:
        """Grab the box around the icon (at most every box_interval_s) and look for it there.

        True = seen, False = not seen (misses counted), None = not time to look yet.
        """
        if now - self._last_box_grab < self.box_interval_s or self._icon_xy is None:
            return None
        self._last_box_grab = now
        x, y = self._icon_xy
        w, h = self.window.width, self.window.height
        x0, y0, x1, y1 = Region.around(x, y, w, h, 0.10).to_box(w, h)
        box: Box = (x0, y0, x1 - x0, y1 - y0)
        crop = self.capturer.grab(box)
        m = self.matcher.match(crop, full_width=w)
        if m.accepted(self.threshold):
            self._icon_xy = (x0 + m.x, y0 + m.y)
            self._misses = 0
            return True
        self._misses += 1
        return False

    def _press(self, key: str, why: str) -> None:
        self.input.press_key(key, self.cfg.run.key_hold_ms / 1000.0)
        self._emit(ev.KeyPressed(self._t(), key, why))

    def _miss(self, did: str) -> None:
        self.stats = ev.Stats(
            self._t(), self.stats.casts, self.stats.catches, self.stats.misses + 1,
            self.stats.bites_heard, self._elapsed(),
        )  # fmt: skip
        self._pause_then_idle(did)

    def _pause_then_idle(self, did: str) -> None:
        pause = self._rng.uniform(self.cfg.run.pause_min_s, self.cfg.run.pause_max_s)
        self._change(State.IDLE, did, f"Casting again in {pause:.1f} s")
        self._icon_xy = None
        remaining = pause
        while remaining > 0 and not self._stop:
            step = min(SLEEP_SLICE_S, remaining)
            self._sleep(step)
            remaining -= step

    def _drain_audio(self, now: float) -> list[AudioChunk]:
        rate = self.audio.sample_rate
        last = self._last_audio if self._last_audio is not None else now
        n = int((now - last) * rate)
        n = max(AUDIO_BLOCK, min(n, int(AUDIO_MAX_S * rate)))
        self._last_audio = now
        samples = self.audio.read(n)
        self._audio_frames += len(samples)
        return [AudioChunk(samples, rate, self._audio_frames / rate)]

    def _change(self, state: State, did: str, nxt: str) -> None:
        self.state = state
        self._emit(ev.StateChanged(self._t(), state, did, nxt))
        self._emit_stats()

    def _note_scores(self) -> None:
        best = getattr(self.detector, "best_heard", None)
        if best is not None and best.score > self._best_heard[1]:
            self._best_heard = (best.name, float(best.score))

    def _bite_threshold(self) -> float:
        return float(getattr(self.detector, "threshold", self.cfg.audio.bite_threshold))

    def _send_scores(self, now: float, *, flush: bool) -> None:
        take = getattr(self.detector, "take_peaks", None)
        if take is None or not (flush or now - self._last_scores_sent >= SCORES_EVERY_S):
            return
        self._last_scores_sent = now
        self._emit(ev.Scores(self._t(), take(), self._bite_threshold()))

    def _maybe_stats(self, now: float) -> None:
        if self.state is not State.WAIT_BITE:
            return
        if now - self._last_listen >= LISTEN_EVERY_S:
            self._last_listen = now
            clip, score = self._best_heard
            self._emit(ev.Listening(self._t(), clip, score, self._bite_threshold()))
        if now - self._last_stats >= STATS_EVERY_S:
            self._emit_stats()

    def _emit_stats(self) -> None:
        self._last_stats = self._clock()
        self.stats = ev.Stats(
            self._t(), self.stats.casts, self.stats.catches, self.stats.misses,
            self.stats.bites_heard, self._elapsed(),
        )  # fmt: skip
        self._emit(self.stats)

    def _finish(self, reason: str) -> None:
        if self.state is State.STOPPED:
            return
        self._emit_stats()
        self.state = State.STOPPED
        self._emit(ev.Stopped(self._t(), reason))

    def _emit(self, event: ev.Event) -> None:
        self.events.put(event)

    def _elapsed(self) -> float:
        return 0.0 if self.started is None else self._clock() - self.started

    def _t(self) -> float:
        return self._elapsed()


def _thumbnail(frame: np.ndarray, x: int, y: int, size: int) -> tuple[np.ndarray, Box]:
    h, w = frame.shape[:2]
    scale = min(1.0, THUMB_WIDTH / w)
    if scale < 1.0:
        thumb = cv2.resize(
            frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
        )
    else:
        thumb = frame.copy()
    side = max(2, round(size * scale))
    bx = max(0, round(x * scale - side / 2))
    by = max(0, round(y * scale - side / 2))
    return thumb, (bx, by, side, side)


def unsupported_platform() -> str:
    """Why the bot cannot run here, or "" when it can (X11/Xwayland on Linux, or Windows)."""
    import os
    import sys

    if sys.platform == "win32" or linux_x11():
        return ""
    if sys.platform == "linux" and not os.environ.get("DISPLAY"):
        return "DISPLAY is not set; the bot needs X11 or Xwayland (log into a session with one)"
    return "the bot is not supported yet on this platform (Linux X11/Xwayland and Windows only)"


def build_engine(
    cfg: Config,
    *,
    session_s: float | None = None,
    title: str = "World of Warcraft",
    record_audio: Path | None = None,
) -> FishingEngine:
    """Wire the real capturer, input device, audio source, detector, and matcher for `cfg`.

    `record_audio` saves everything the engine hears to that WAV when it stops (for debugging
    missed splashes offline). Raises RuntimeError with a one-line hint when a prerequisite is
    missing.
    """
    from wowfish.audio._recording import RecordingSource
    from wowfish.audio.detector import detector_for
    from wowfish.audio.devices import open_capture
    from wowfish.input import get_input_device
    from wowfish.vision.matcher import MIN_TEMPLATE_PX, IconMatcher
    from wowfish.vision.template import load_icon_templates, template_for_size

    platform_problem = unsupported_platform()
    if platform_problem:
        raise RuntimeError(platform_problem)
    window = find_window(title)
    if window is None:
        raise RuntimeError(f"no window titled {title!r}; is the game running?")
    templates = load_icon_templates()

    def matcher_factory(size: int) -> IconMatcher:
        return IconMatcher(template_for_size(templates, size), cfg.vision.match_width)

    size = cfg.vision.icon_size_px
    if size > 0 and matcher_factory(size).template_px(window.width) < MIN_TEMPLATE_PX:
        size = 0  # the saved size is useless for this window; the first cast measures it
    cfg = replace(cfg, vision=replace(cfg.vision, icon_size_px=size))

    def on_calibrated(cal: Any) -> None:
        from wowfish.config import load_config, save_config

        current, _ = load_config()
        vision = replace(
            current.vision,
            icon_size_px=cal.icon_size_px,
            icon_native_size=cal.native_size,
            icon_threshold=cal.suggested_threshold(),
        )
        save_config(replace(current, vision=vision))

    detector = detector_for(cfg.audio)
    audio: Any = ThreadedSource(open_capture(cfg.audio.selector, pid=window.pid, title=title))
    if record_audio is not None:
        audio = RecordingSource(audio, record_audio)
    return FishingEngine(
        cfg,
        window,
        capturer=get_capturer(),
        input_device=get_input_device(),
        audio_source=audio,
        detector=detector,
        matcher_factory=matcher_factory,
        on_calibrated=on_calibrated,
        find_window=lambda t=title: find_window(t),
        session_s=session_s,
    )


def _real_calibrate(frame: np.ndarray, *, match_width: int = 1920) -> Any:
    from wowfish.vision.calibrate import calibrate_icon_scale
    from wowfish.vision.template import load_icon_templates

    return calibrate_icon_scale(frame, load_icon_templates(), match_width=match_width)
