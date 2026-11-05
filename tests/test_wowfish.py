"""Seven checks that protect what the bot actually relies on: the shipped assets, the default
config, logging in the console-less GUI executable, that the two detectors say yes to the real
thing and no to nothing, and that a native Wayland game window is read correctly from the
compositor's replies.

Everything here runs on bundled files and synthetic signals. No screen, audio device, window
system, or Tk is touched, so a failure means the code or the assets changed, not the machine.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from loguru import logger

from wowfish.assets import CLIP_NAMES, ICON_SIZES
from wowfish.audio import AudioChunk
from wowfish.audio.clips import build_fingerprints, load_reference_clips
from wowfish.audio.detector import BiteDetector, BiteEvent, detector_for
from wowfish.audio.features import DEFAULT_MEL
from wowfish.capture._wayland import pick_hyprland_window
from wowfish.cli import _setup_logging
from wowfish.config import (
    AudioConfig,
    Config,
    GameConfig,
    VisionConfig,
    ensure_config_file,
    load_config,
    save_config,
)
from wowfish.input._wayland import keycodes_from_keymap
from wowfish.vision.matcher import IconMatcher
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


def test_wayland_window_and_keys_from_compositor_replies():
    """Hyprland reports logical geometry; the window comes back sized in its output's pixels,
    and the largest title match wins. Each key resolves to its evdev keycode (xkb code - 8)
    whether the keymap names keysyms (xkbcommon) or writes them in hex (Hyprland), through
    aliases and from a shifted level (the digits on AZERTY)."""

    def client(address, title, at, size, monitor):
        return {"address": address, "title": title, "pid": 42, "at": at, "size": size,
                "monitor": monitor, "mapped": True, "hidden": False}  # fmt: skip

    clients = [
        client("0x1f", "World of Warcraft - Wowhead", [0, 0], [800, 600], 0),
        client("0x2a", "World of Warcraft", [2560, 0], [3200, 1800], 1),
        client("0x3b", "Discord", [0, 0], [3840, 2160], 0),
    ]
    monitors = [{"id": 0, "scale": 1.0}, {"id": 1, "scale": 1.2}]
    w = pick_hyprland_window(clients, monitors, "world of warcraft")
    assert w is not None
    assert (w.title, w.pid, w.left, w.top, w.width, w.height) == (
        "World of Warcraft", 42, 2560, 0, 3840, 2160,
    )  # fmt: skip
    assert (w.handle, w.compositor, w.scale) == (0x2A, "hyprland", 1.2)
    assert pick_hyprland_window(clients, monitors, "Hearthstone") is None

    for symbols in (NAMED_SYMBOLS, HEX_SYMBOLS):
        codes = keycodes_from_keymap(KEYMAP % symbols)
        assert codes == {"1": 2, "f": 33, "enter": 28, "f1": 59}, symbols
