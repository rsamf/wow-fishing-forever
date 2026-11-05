"""Ten checks that protect what the bot actually relies on: the shipped assets, the default
config, logging in the console-less GUI executable, that the two detectors say yes to the real
thing and no to nothing, that the game window is picked correctly from the compositor's
replies and from macOS's window list, that macOS listens to BlackHole, and that the fishing
loop varies its rhythm only when the config asks it to.

Everything here runs on bundled files and synthetic signals. No screen, audio device, window
system, or Tk is touched, so a failure means the code or the assets changed, not the machine.
"""

from __future__ import annotations

import random
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from loguru import logger

from wowfish import capture
from wowfish.assets import CLIP_NAMES, ICON_SIZES
from wowfish.audio import AudioChunk, devices
from wowfish.audio.clips import build_fingerprints, load_reference_clips
from wowfish.audio.detector import BiteDetector, BiteEvent, detector_for
from wowfish.audio.features import DEFAULT_MEL
from wowfish.capture import WindowRef, _quartz, _wayland, _x11
from wowfish.capture._quartz import frontmost_pid, pick_quartz_window
from wowfish.capture._wayland import pick_hyprland_window
from wowfish.cli import _setup_logging
from wowfish.config import (
    AudioConfig,
    BehaviorConfig,
    Config,
    GameConfig,
    RunConfig,
    VisionConfig,
    ensure_config_file,
    load_config,
    save_config,
)
from wowfish.engine.engine import FishingEngine, unsupported_platform
from wowfish.errors import PermissionMissing
from wowfish.gui.presenter import Presenter, describe
from wowfish.input import _quartz as input_quartz
from wowfish.input import get_input_device
from wowfish.input._wayland import keycodes_from_keymap
from wowfish.input.keys import MAC_KEYCODES, SUPPORTED_KEYS
from wowfish.vision.matcher import IconMatch, IconMatcher
from wowfish.vision.template import load_icon_templates, template_for_size

REPO_ROOT = Path(__file__).resolve().parents[1]
RATE = DEFAULT_MEL.sample_rate


def test_bundled_assets_load():
    """Every sprite and clip the code expects is shipped and readable; the release check agrees."""
    check = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_release.py")],
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stderr
    assert tuple(t.native_size for t in load_icon_templates()) == ICON_SIZES
    assert tuple(fp.name for fp in build_fingerprints()) == CLIP_NAMES


def test_default_config_matches_code_defaults(tmp_path):
    """The shipped default_config file and the dataclass defaults describe the same settings,
    and a config survives a save/load round trip unchanged."""
    path = tmp_path / "config.toml"
    assert ensure_config_file(path) is True
    assert ensure_config_file(path) is False  # never overwrites an existing file

    from_file, warnings = load_config(path)
    assert warnings == []
    assert from_file == Config(game=GameConfig(), vision=VisionConfig(), audio=AudioConfig())

    save_config(from_file, path)
    assert load_config(path) == (from_file, [])


def test_logging_works_without_stderr(tmp_path, monkeypatch):
    """wowfish-gui.exe is built without a console, so on Windows Python sets sys.stderr to None.
    Logging setup must not crash there, and the log file must still be written."""
    monkeypatch.setenv("WOWFISH_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "stderr", None)
    try:
        log_path = _setup_logging()
        logger.info("started without a console")
    finally:
        logger.remove()  # closes the file sink so tmp_path can be deleted on Windows
    assert "started without a console" in log_path.read_text()


def _hear(stream: np.ndarray) -> tuple[list[BiteEvent], BiteDetector]:
    """Run a stream through the detector the bot builds from the default config, in modest
    chunks like the live loop does."""
    det = detector_for(AudioConfig())
    events: list[BiteEvent] = []
    chunk = 2048
    for i in range(0, len(stream), chunk):
        part = stream[i : i + chunk]
        events += det.feed(AudioChunk(part, RATE, (i + len(part)) / RATE))
    return events, det


def test_bite_detector_hears_each_clip_once():
    """Each of the four bundled splashes, padded with silence, yields exactly one bite naming
    that clip, above the shipped threshold, timed inside the clip, and the "best match" the
    status line reports and the peak the Scores graph plots are that same peak. Two splashes
    well apart yield two."""
    threshold = AudioConfig().bite_threshold
    gap = np.zeros(RATE, dtype=np.float32)  # 1 s
    for name, clip in load_reference_clips(RATE).items():
        events, det = _hear(np.concatenate([gap, clip, gap]))
        assert [e.clip for e in events] == [name], name
        assert events[0].score >= threshold, (name, events[0].score)
        assert 1.0 - 0.1 <= events[0].t <= 1.0 + len(clip) / RATE, (name, events[0].t)
        best = det.best_heard
        assert (best.name, best.score) == (name, events[0].score), (name, best)
        assert det.take_peaks()[name] == events[0].score, name  # what the Scores graph plots
        assert det.take_peaks()[name] == -1.0, name  # taking the peaks clears them

    clip = load_reference_clips(RATE)[CLIP_NAMES[0]]
    apart = np.zeros(4 * RATE, dtype=np.float32)  # longer than the default refractory period
    assert len(_hear(np.concatenate([gap, clip, apart, clip, gap]))[0]) == 2


def test_bite_detector_stays_quiet_on_silence_and_noise():
    """Nothing fires on silence or broadband noise, and the scores stay well under threshold."""
    threshold = AudioConfig().bite_threshold
    rng = np.random.default_rng(0)
    silence = np.zeros(3 * RATE, dtype=np.float32)
    noise = (rng.standard_normal(3 * RATE) * 0.1).astype(np.float32)
    for stream in (silence, noise):
        det = detector_for(AudioConfig())
        events: list[BiteEvent] = []
        for i in range(0, len(stream), 2048):
            part = stream[i : i + 2048]
            events += det.feed(AudioChunk(part, RATE, (i + len(part)) / RATE))
        assert events == []
        assert max(s.score for s in det.last_scores) < threshold - 0.2


def test_icon_matcher_finds_sprite_and_rejects_water():
    """Every sprite pasted onto noisy blue water is found at the right spot, at 1080p and on a
    4K-wide frame that the matcher downscales. The same water without a sprite is rejected."""
    threshold = VisionConfig().icon_threshold
    templates = load_icon_templates()
    rng = np.random.default_rng(0)
    for frame_w, frame_h in ((1920, 1080), (3840, 2160)):
        water = np.full((frame_h, frame_w, 3), (90, 60, 30), np.uint8)  # BGR: bluish
        water += rng.integers(0, 20, water.shape, dtype=np.uint8)
        for size in ICON_SIZES:
            t = template_for_size(templates, size)
            x, y = frame_w // 3, frame_h // 3
            frame = water.copy()
            frame[y : y + size, x : x + size][t.mask > 0] = t.bgr[t.mask > 0]

            hit = IconMatcher(t).match(frame)
            assert hit.accepted(threshold), (frame_w, size, hit)
            assert abs(hit.x - (x + size / 2)) <= 2 and abs(hit.y - (y + size / 2)) <= 2

            miss = IconMatcher(t).match(water)
            assert not miss.accepted(threshold), (frame_w, size, miss)


KEYMAP = """xkb_keymap {
xkb_keycodes "evdev" {
	minimum = 8;
	maximum = 255;
	<AE01> = 10;
	<AC04> = 41;
	<RTRN> = 36;
	<FK01> = 67;
	alias <LatF> = <AC04>;
};
xkb_types "complete" { type "ONE_LEVEL" { modifiers= none; }; };
xkb_symbols "pc+fr" {
%s};
};
"""
NAMED_SYMBOLS = """\tkey <AE01> {\t[ ampersand, 1 ] };
\tkey <LatF> {\t[ f, F ] };
\tkey <RTRN> {\t[ Return ] };
\tkey <FK01> {
\t\ttype= "CTRL+ALT",
\t\tsymbols[Group1]= [ F1, F1, F1, F1, XF86Switch_VT_1 ]
\t};
"""
HEX_SYMBOLS = """\tkey <AE01> {\t[ 0x26, 0x31 ] };
\tkey <LatF> {\t[ 0x66, 0x46 ] };
\tkey <RTRN> {\t[ 0xff0d ] };
\tkey <FK01> {
\t\ttype= "CTRL+ALT",
\t\tsymbols[1]= [ 0xffbe, 0xffbe, 0xffbe, 0xffbe, 0x1008fe01 ]
\t};
"""


def test_wayland_window_and_keys_from_compositor_replies(monkeypatch):
    """Hyprland reports logical geometry for native and Xwayland windows alike; the window comes
    back sized in its output's pixels, and the largest title match wins whichever kind it is. An
    Xwayland winner is then looked up through X11 by its pid. Each key resolves to its evdev
    keycode (xkb code - 8) whether the keymap names keysyms (xkbcommon) or writes them in hex
    (Hyprland), through aliases and from a shifted level (the digits on AZERTY)."""

    def client(address, title, at, size, monitor, pid=42, xwayland=False):
        return {"address": address, "title": title, "pid": pid, "at": at, "size": size,
                "monitor": monitor, "mapped": True, "hidden": False,
                "xwayland": xwayland}  # fmt: skip

    clients = [
        client("0x1f", "World of Warcraft - Wowhead", [0, 0], [800, 600], 0),
        client("0x2a", "World of Warcraft", [2560, 0], [3200, 1800], 1),
        client("0x3b", "Discord", [0, 0], [3840, 2160], 0),
        client("0x4c", "World of Warcraft", [0, 0], [3000, 1700], 0, 7, xwayland=True),
    ]
    monitors = [{"id": 0, "scale": 1.0}, {"id": 1, "scale": 1.2}, {"id": 2, "scale": 2.0}]
    w = pick_hyprland_window(clients, monitors, "world of warcraft")
    assert w is not None
    assert (w.title, w.pid, w.left, w.top, w.width, w.height) == (
        "World of Warcraft", 42, 2560, 0, 3840, 2160,
    )  # fmt: skip
    assert (w.handle, w.compositor, w.scale) == (0x2A, "hyprland", 1.2)
    assert pick_hyprland_window(clients, monitors, "Hearthstone") is None

    clients.append(client("0x5d", "World of Warcraft", [0, 0], [2560, 1440], 2, 7, xwayland=True))
    x = pick_hyprland_window(clients, monitors, "world of warcraft")
    assert x is not None
    assert (x.pid, x.width, x.height, x.handle, x.compositor) == (7, 5120, 2880, None, None)

    asked: list[int | None] = []
    monkeypatch.setattr(capture, "linux_hyprland", lambda: True)
    monkeypatch.setattr(capture, "linux_x11", lambda: True)
    monkeypatch.setattr(_x11, "find_window_x11", lambda _, pid: asked.append(pid))
    for winner, x11_asked in ((w, []), (x, [7])):
        monkeypatch.setattr(_wayland, "find_window_hyprland", lambda _, found=winner: found)
        capture.find_window()
        assert asked == x11_asked

    for symbols in (NAMED_SYMBOLS, HEX_SYMBOLS):
        codes = keycodes_from_keymap(KEYMAP % symbols)
        assert codes == {"1": 2, "f": 33, "enter": 28, "f1": 59}, symbols


def test_macos_window_focus_and_keys(monkeypatch):
    """CoreGraphics lists windows front to back, in points. The largest visible title match wins;
    when macOS withholds titles (Screen Recording off) the app's name stands in. The front-most
    visible app window owns the keyboard, whatever floats above it. On macOS the window lookup,
    the platform check, and the input device all go through Quartz, every supported key has its
    own keycode, and a key press without the Accessibility permission says so instead of being
    dropped."""

    def win(number, name, owner, pid, bounds, layer=0, alpha=1.0):
        return {"number": number, "name": name, "owner": owner, "pid": pid, "layer": layer,
                "alpha": alpha, "bounds": bounds}  # fmt: skip

    infos = [
        win(1, "", "Window Server", 90, (0, 0, 1512, 33), layer=25),  # the menu bar
        win(2, None, "Terminal", 50, (100, 100, 800, 500)),
        win(3, "World of Warcraft", "World of Warcraft", 77, (0, 33, 1512, 949)),
        win(4, "World of Warcraft - Wowhead", "Safari", 60, (0, 0, 600, 400)),
        win(5, "World of Warcraft", "World of Warcraft", 77, (0, 0, 4000, 3000), alpha=0.0),
    ]
    w = pick_quartz_window(infos, "world of warcraft")
    assert w is not None
    assert (w.title, w.pid, w.left, w.top, w.width, w.height, w.handle, w.scale) == (
        "World of Warcraft", 77, 0, 33, 1512, 949, 3, 1.0,
    )  # fmt: skip
    hidden = pick_quartz_window([{**i, "name": None} for i in infos], "World of Warcraft")
    assert hidden is not None and (hidden.title, hidden.handle) == ("World of Warcraft", 3)
    assert pick_quartz_window(infos, "Hearthstone") is None
    assert (frontmost_pid(infos), frontmost_pid(infos[2:])) == (50, 77)

    assert set(MAC_KEYCODES) == set(SUPPORTED_KEYS)
    assert len(set(MAC_KEYCODES.values())) == len(MAC_KEYCODES)
    assert (MAC_KEYCODES["1"], MAC_KEYCODES["f"], MAC_KEYCODES["f1"]) == (0x12, 0x03, 0x7A)

    posted: list[tuple[int, bool]] = []
    cg = SimpleNamespace(
        CGEventSourceCreate=lambda state: 1,
        CGEventCreateKeyboardEvent=lambda source, code, down: (code, down),
        CGEventSetFlags=lambda event, flags: None,
        CGEventPost=lambda tap, event: posted.append(event),
    )
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(_quartz, "window_infos", lambda: infos)
    monkeypatch.setattr(_quartz, "frameworks", lambda: (cg, SimpleNamespace(CFRelease=id)))
    assert capture.find_window() == w
    assert unsupported_platform() == ""
    keyboard = get_input_device(w)
    assert isinstance(keyboard, input_quartz.QuartzInputDevice)
    assert keyboard.is_game_focused(w) is False  # Terminal is in front
    monkeypatch.setattr(input_quartz, "accessibility_available", lambda: (False, "Accessibility"))
    try:
        keyboard.press_key("f", 0)
    except PermissionMissing as e:
        assert str(e) == "Accessibility"
    else:
        raise AssertionError("pressed a key without the Accessibility permission")
    monkeypatch.setattr(input_quartz, "accessibility_available", lambda: (True, ""))
    keyboard.press_key("F", 0)
    assert posted == [(0x03, True), (0x03, False)]


def test_macos_listens_to_blackhole(monkeypatch, tmp_path):
    """macOS has no loopback capture, so the bot listens to an input device: BlackHole is listed
    first and is the default, under its name (CoreAudio ids change across reboots), and an exact
    id or name wins over one it is part of. It records with CoreAudio's own buffer size, since a
    device such as BlackHole refuses the 1024 frames used elsewhere. The GUI offers no "game
    only" choice there, and Start names the missing permission and the audio setting until both
    are fixed."""
    asked: list[dict] = []

    def mic(id_, name):
        return SimpleNamespace(id=id_, name=name, isloopback=False,
                               recorder=lambda **kw: asked.append(kw) or nullcontext())  # fmt: skip

    mics = [mic(4, "MacBook Pro Microphone"), mic(42, "BlackHole 16ch"), mic(43, "BlackHole 2ch")]
    soundcard = SimpleNamespace(all_microphones=lambda: list(mics), default_speaker=lambda: None)
    monkeypatch.setenv("WOWFISH_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(devices, "_soundcard", lambda: soundcard)
    found = [(d.id, d.is_default) for d in devices.list_loopback_devices()]
    assert found == [("BlackHole 16ch", True), ("BlackHole 2ch", True),
                     ("MacBook Pro Microphone", False)]  # fmt: skip
    assert devices.open_loopback("BlackHole 2ch").mic is mics[2]
    assert devices.open_loopback("blackhole").mic is mics[1]
    assert devices.open_loopback("4").mic is mics[0]  # not BlackHole 16ch, whose id is 42
    with devices.open_loopback("blackhole"):
        assert asked == [{"samplerate": 22050, "channels": 1, "blocksize": None}]

    monkeypatch.setattr(_quartz, "screen_capture_available", lambda request: (True, ""))
    monkeypatch.setattr(input_quartz, "accessibility_available", lambda request: (False, "AX off"))
    p = Presenter(find_window=lambda _: None)
    assert [d.id for d in p.audio_devices()] == [m.name for m in mics[1:] + mics[:1]]
    assert p.prerequisites()[:2] == ["AX off", devices.MAC_NO_GAME_AUDIO]
    p.save_audio_device(p.default_device().id)
    assert (p.cfg.audio.capture, p.cfg.audio.device) == ("device", "BlackHole 16ch")
    assert devices.MAC_NO_GAME_AUDIO not in p.prerequisites()


class _Pond:
    """A game in which every cast lands a bobber that bites at once and Interact clears it.
    Fills in for the clock, capturer, keyboard, matcher, audio, and detector; time moves only
    when the engine sleeps."""

    def __init__(self) -> None:
        self.t = 0.0
        self.sample_rate = RATE
        self.bobber = False
        self.pressed: list[tuple[float, str, float]] = []  # time, key, hold
        self.bites: list[float] = []

    def clock(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s

    def open(self, window) -> None:
        pass

    def close(self) -> None:
        pass

    def grab(self, box=None) -> np.ndarray:
        return np.zeros((10, 10, 3), np.uint8)

    def match(self, frame, region=None, full_width=None) -> IconMatch:
        return IconMatch(0.9 if self.bobber else 0.0, 5, 5, 4, 1.0)

    def press_key(self, key: str, hold_s: float) -> None:
        self.pressed.append((self.t, key, hold_s))
        if key in ("1", "f"):
            self.bobber = key == "1"

    def is_game_focused(self, window) -> bool:
        return True

    def read(self, n: int) -> np.ndarray:
        return np.zeros(n, np.float32)

    def feed(self, chunk) -> list[BiteEvent]:
        if not self.bobber:
            return []
        self.bites.append(self.t)
        return [BiteEvent("fishingbobber_1", 0.9, 0.0)]

    def reset(self) -> None:
        pass


def _fish(behavior: BehaviorConfig, minutes: float) -> tuple[_Pond, list[str]]:
    pond = _Pond()
    cfg = Config(game=GameConfig(), vision=VisionConfig(), audio=AudioConfig(), behavior=behavior)
    engine = FishingEngine(
        cfg, WindowRef("World of Warcraft", None, 0, 0, 10, 10),
        capturer=pond, input_device=pond, audio_source=pond, detector=pond, matcher=pond,
        clock=pond.clock, sleep=pond.sleep, rng=random.Random(7), session_s=0,
    )  # fmt: skip
    while pond.t < minutes * 60:
        engine.step()
    return pond, [describe(e) for e in engine.events.queue]


def test_fishing_loop_varies_its_rhythm_only_when_asked():
    """Out of the box every press holds run.key_hold_ms and nothing but cast and Interact is
    pressed. Each [behavior] setting, once filled in, shows in the presses:
    - breaks: after the fishing stretch the loop rests for the whole break
    - reaction: Interact waits out the reaction time after the splash
    - skipped bites: some splashes get no Interact
    """
    pond, _ = _fish(BehaviorConfig(), 1)
    assert {k for _, k, _ in pond.pressed} == {"1", "f"}
    assert {h for *_, h in pond.pressed} == {RunConfig().key_hold_ms / 1000}

    behavior = BehaviorConfig(
        break_every_min_minutes=1,
        break_every_max_minutes=1,
        break_min_minutes=2,
        break_max_minutes=2,
        reaction_min_ms=200,
        reaction_max_ms=400,
        skip_bite_chance=0.3,
    )
    pond, lines = _fish(behavior, 5)
    gaps = [b[0] - a[0] for a, b in zip(pond.pressed, pond.pressed[1:], strict=False)]
    assert max(gaps) >= 120
    assert any("Taking a 2.0 min break" in line for line in lines)
    for t in (t for t, key, _ in pond.pressed if key == "f"):
        heard = max(b for b in pond.bites if b <= t)
        assert 0.2 - 1e-9 <= t - heard <= 0.4 + 1e-9
    assert 0 < [k for _, k, _ in pond.pressed].count("f") < len(pond.bites)
    assert any("Let this bite go on purpose" in line for line in lines)
