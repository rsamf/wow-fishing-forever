"""Environment probes behind `wowfish doctor`. Probes never raise; failures become lines."""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from wowfish import __version__, paths
from wowfish.capture import linux_hyprland, macos

Level = Literal["ok", "warn", "fail"]


@dataclass(frozen=True)
class ProbeResult:
    name: str
    level: Level
    detail: str


Probe = Callable[[], ProbeResult]


def run_probes(probes: Sequence[Probe]) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    for probe in probes:
        try:
            results.append(probe())
        except Exception as e:  # doctor must never crash
            results.append(ProbeResult(probe.__name__, "fail", f"{type(e).__name__}: {e}"))
    return results


def render(results: Sequence[ProbeResult]) -> str:
    return "\n".join(f"{r.level:<5} {r.name}: {r.detail}" for r in results)


def exit_code(results: Sequence[ProbeResult]) -> int:
    return 1 if any(r.level == "fail" for r in results) else 0


# ---- probes -------------------------------------------------------------------


def probe_system() -> ProbeResult:
    detail = (
        f"{platform.system()} {platform.machine()}, "
        f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}, "
        f"wowfish {__version__}, log {paths.log_file()}"
    )
    return ProbeResult("system", "ok", detail)


def _x_display_opens() -> tuple[bool, str]:
    try:
        from Xlib import display

        display.Display().close()
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _screen_capture_available() -> tuple[bool, str]:
    from wowfish.capture._quartz import screen_capture_available

    return screen_capture_available()


def probe_display() -> ProbeResult:
    if macos():  # a permission, not a broken install: warn, so doctor still exits 0
        ok, detail = _screen_capture_available()
        return ProbeResult("display", "ok" if ok else "warn", f"native (Quartz); {detail}")
    if sys.platform != "linux":
        return ProbeResult("display", "ok", "native")
    wayland = "Hyprland (native Wayland windows supported)" if linux_hyprland() else ""
    name = os.environ.get("DISPLAY")
    if not name:
        if wayland:
            return ProbeResult("display", "ok", f"{wayland}, no Xwayland")
        return ProbeResult(
            "display",
            "warn",
            "DISPLAY is not set; the bot needs X11, Xwayland, or Hyprland "
            "(log into a session with one)",
        )
    ok, why = _x_display_opens()
    if not ok:
        if wayland:
            return ProbeResult("display", "ok", f"{wayland}; cannot open {name}: {why}")
        return ProbeResult("display", "fail", f"cannot open {name}: {why}")
    server = "Xwayland" if os.environ.get("WAYLAND_DISPLAY") else "X11"
    return ProbeResult("display", "ok", f"{name} ({server})" + (f", {wayland}" if wayland else ""))


def probe_opencv() -> ProbeResult:
    import cv2

    return ProbeResult("opencv", "ok", cv2.__version__)


def probe_audio() -> ProbeResult:
    from wowfish.audio import AudioUnavailable, devices

    try:
        found = devices.list_loopback_devices()
    except AudioUnavailable as e:
        return ProbeResult("audio", "warn", f"no audio backend ({e})")
    if not found:
        return ProbeResult(
            "audio",
            "warn",
            "no loopback devices (macOS: install a virtual device such as BlackHole)",
        )
    if macos() and not any(d.is_default for d in found):
        return ProbeResult(
            "audio",
            "warn",
            f"{len(found)} input device(s) but no BlackHole; install it and route the game's "
            "sound through it (see the README)",
        )
    default = next((d.name for d in found if d.is_default), found[0].name)
    return ProbeResult("audio", "ok", f"{len(found)} loopback device(s), default: {default}")


def probe_game_audio() -> ProbeResult:
    from wowfish.audio import AudioUnavailable, devices
    from wowfish.config import ConfigError, load_config

    if macos():  # no per-application capture: the bot listens to the configured device
        try:
            device = load_config()[0].audio.device
            return ProbeResult("game audio", "ok", f"device {devices.mac_device_name(device)!r}")
        except ConfigError as e:
            return ProbeResult("game audio", "fail", str(e))
        except AudioUnavailable as e:
            return ProbeResult("game audio", "warn", str(e))
    if sys.platform == "linux":
        from wowfish.audio._pipewire import pipewire_available

        ok, detail = pipewire_available()
        return ProbeResult("game audio", "ok" if ok else "warn", detail)
    if sys.platform == "win32":
        build = sys.getwindowsversion().build
        if build >= 19041:
            return ProbeResult(
                "game audio", "ok", f"WASAPI process loopback (Windows build {build})"
            )
        return ProbeResult(
            "game audio",
            "warn",
            "per-app audio needs Windows 10 version 2004 (build 19041) or newer",
        )
    return ProbeResult("game audio", "warn", "no game audio capture on this platform")


def _xtest_available() -> tuple[bool, str]:
    from wowfish.input._x11 import xtest_available

    return xtest_available()


def _virtual_keyboard_available() -> tuple[bool, str]:
    from wowfish.input._wayland import virtual_keyboard_available

    return virtual_keyboard_available()


def _accessibility_available() -> tuple[bool, str]:
    from wowfish.input._quartz import accessibility_available

    return accessibility_available()


def probe_input() -> ProbeResult:
    if sys.platform == "win32":
        return ProbeResult("input", "ok", "SendInput")
    if macos():
        ok, detail = _accessibility_available()
        return ProbeResult("input", "ok" if ok else "warn", detail)
    x11 = sys.platform == "linux" and bool(os.environ.get("DISPLAY"))
    checks: list[tuple[bool, str]] = []
    if linux_hyprland():  # a native Wayland game takes the virtual keyboard, an Xwayland one XTest
        checks.append(_virtual_keyboard_available())
    if x11:
        checks.append(_xtest_available())
    if not checks:
        return ProbeResult("input", "warn", "key presses not supported yet on this platform")
    ok = all(c[0] for c in checks)
    return ProbeResult("input", "ok" if ok else "fail", "; ".join(c[1] for c in checks))


def probe_config() -> ProbeResult:
    from wowfish.config import ConfigError, load_config

    path = paths.config_file()
    try:
        _, warnings = load_config(path)
    except ConfigError as e:
        return ProbeResult("config", "fail", str(e))
    if warnings:
        return ProbeResult("config", "warn", f"{path}: " + "; ".join(warnings))
    return ProbeResult("config", "ok", str(path) if path.exists() else f"{path} (defaults)")


def default_probes() -> list[Probe]:
    return [
        probe_system,
        probe_display,
        probe_opencv,
        *([probe_audio] if macos() else []),
        probe_game_audio,
        probe_input,
        probe_config,
    ]
