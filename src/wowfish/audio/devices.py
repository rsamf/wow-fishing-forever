"""What the bot can listen to: the game's own stream ("game") or a system loopback device.

`soundcard` is imported lazily: CI has no audio. Per-application capture is in `_pipewire`
(Linux) and `_wasapi` (Windows). macOS has neither per-application capture nor loopback, so there
the "loopback devices" are input devices, and a virtual one such as BlackHole carries the game.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from wowfish.audio import AudioUnavailable
from wowfish.audio._soundcard import SoundcardSource


@dataclass(frozen=True)
class LoopbackDevice:
    id: str
    name: str
    is_default: bool


GAME = "game"
GAME_CHOICE = LoopbackDevice(GAME, "World of Warcraft only (recommended)", True)
MAC_VIRTUAL = ("blackhole", "loopback", "soundflower")  # macOS devices that carry system audio
MAC_NO_GAME_AUDIO = (
    "macOS cannot capture one app's audio: route the game's sound through BlackHole and pick it "
    'on the Setup tab (or set audio.capture = "device" and audio.device = "BlackHole")'
)


def _soundcard() -> Any:
    try:
        import soundcard
    except Exception as e:  # ImportError, or the Pulse/CoreAudio client failing at import
        raise AudioUnavailable(f"soundcard backend unavailable: {type(e).__name__}: {e}") from e
    return soundcard


def _mac_virtual(mic: Any) -> bool:
    return any(v in str(mic.name).lower() for v in MAC_VIRTUAL)


def _loopback_mics() -> tuple[list[Any], Any]:
    sc = _soundcard()
    try:
        if sys.platform == "darwin":  # every input device, BlackHole and the like first
            mics = sorted(sc.all_microphones(), key=lambda m: not _mac_virtual(m))
        else:
            mics = sc.all_microphones(include_loopback=True)
            mics = [m for m in mics if getattr(m, "isloopback", False)]
    except Exception as e:
        raise AudioUnavailable(f"cannot list audio devices: {type(e).__name__}: {e}") from e
    try:
        speaker = sc.default_speaker()
    except Exception:
        speaker = None
    return mics, speaker


def _is_default(mic: Any, speaker: Any) -> bool:
    if sys.platform == "darwin":
        return _mac_virtual(mic)
    if speaker is None:
        return False
    return str(speaker.id) in str(mic.id) or str(speaker.name) in str(mic.name)


def _device_id(mic: Any) -> str:
    # CoreAudio ids are small integers that change across reboots; the name is what stays put
    return str(mic.name) if sys.platform == "darwin" else str(mic.id)


def list_loopback_devices() -> list[LoopbackDevice]:
    mics, speaker = _loopback_mics()
    return [LoopbackDevice(_device_id(m), str(m.name), _is_default(m, speaker)) for m in mics]


def open_loopback(
    selector: str = "", *, sample_rate: int = 22050, blocksize: int = 1024
) -> SoundcardSource:
    """Open by exact id or name, else by id substring or case-insensitive name substring.

    An empty selector opens the monitor of the default output (on macOS, BlackHole or the like).
    """
    mics, speaker = _loopback_mics()
    if not mics:
        raise AudioUnavailable(
            "no loopback devices found (macOS needs a virtual device such as BlackHole)"
        )
    if selector:
        wanted = selector.lower()
        mic = next((m for m in mics if selector in (str(m.id), str(m.name))), None) or next(
            (m for m in mics if selector in str(m.id) or wanted in str(m.name).lower()), None
        )
        if mic is None:
            raise AudioUnavailable(
                f"no loopback device matches {selector!r}; run `wowfish listen --list`"
            )
    else:
        mic = next((m for m in mics if _is_default(m, speaker)), mics[0])
    return SoundcardSource(mic, sample_rate, blocksize)


def capture_choices(title: str = "World of Warcraft") -> list[LoopbackDevice]:
    """The game's own audio first (the default), then every loopback monitor (never default)."""
    try:
        monitors = list_loopback_devices()
    except AudioUnavailable:
        monitors = []
    return [GAME_CHOICE] + [LoopbackDevice(d.id, d.name, False) for d in monitors]


def _open_pipewire(pid: int | None, name: str, **kw: Any) -> Any:
    from wowfish.audio._pipewire import open_app_stream

    return open_app_stream(pid=pid, name=name, **kw)


def _open_wasapi(pid: int, **kw: Any) -> Any:
    from wowfish.audio._wasapi import ProcessLoopbackSource

    return ProcessLoopbackSource(pid, **kw)


def open_capture(
    selector: str = GAME,
    *,
    pid: int | None = None,
    title: str = "World of Warcraft",
    sample_rate: int = 22050,
    blocksize: int = 1024,
) -> Any:
    """Open what `selector` names: "game" for the game's own stream, else a loopback device.

    For "game", `pid` is the game window's process (preferred) and `title` the fallback name.
    """
    if selector != GAME:
        return open_loopback(selector, sample_rate=sample_rate, blocksize=blocksize)
    if sys.platform == "linux":
        return _open_pipewire(pid, title, sample_rate=sample_rate, blocksize=blocksize)
    if sys.platform == "win32":
        if pid is None:
            raise AudioUnavailable(
                f"no window titled {title!r}; the game must be running to capture its audio"
            )
        return _open_wasapi(pid, sample_rate=sample_rate, blocksize=blocksize)
    raise AudioUnavailable(MAC_NO_GAME_AUDIO)


def describe_game_audio(pid: int | None, title: str = "World of Warcraft") -> str:
    """One line on how the game's audio would be captured here; AudioUnavailable when it cannot."""
    if sys.platform == "linux":
        from wowfish.audio._pipewire import describe_stream

        return describe_stream(pid=pid, name=title)
    if sys.platform == "win32":
        if pid is None:
            raise AudioUnavailable(f"no window titled {title!r}; is the game running?")
        return f"WASAPI process loopback on pid {pid}"
    raise AudioUnavailable(
        'per-application capture is not available on macOS; set audio.capture = "device"'
    )
