"""What the bot listens to: the app that owns the window the user picked.

Linux (`_pipewire`) and Windows (`_wasapi`) capture one application's audio by its process.
macOS has neither per-application capture nor loopback, so there the bot listens to an input
device, and a virtual one such as BlackHole carries the game. `soundcard` is imported lazily: CI
has no audio.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from wowfish.audio import AudioUnavailable
from wowfish.audio._soundcard import SoundcardSource
from wowfish.capture import WindowRef


@dataclass(frozen=True)
class LoopbackDevice:
    id: str
    name: str
    is_default: bool


MAC_VIRTUAL = ("blackhole", "loopback", "soundflower")  # macOS devices that carry system audio
MAC_NO_GAME_AUDIO = (
    "no BlackHole or other virtual audio device found; install BlackHole and route the game's "
    "sound through it (see the README)"
)


def audio_follows_window() -> bool:
    """Whether the game's audio is captured from the picked window's app (not on macOS)."""
    return sys.platform != "darwin"


def _soundcard() -> Any:
    try:
        import soundcard
    except Exception as e:  # ImportError, or the Pulse/CoreAudio client failing at import
        raise AudioUnavailable(f"soundcard backend unavailable: {type(e).__name__}: {e}") from e
    return soundcard


def _mac_virtual(mic: Any) -> bool:
    return any(v in str(mic.name).lower() for v in MAC_VIRTUAL)


def _mac_inputs() -> list[Any]:
    """Every macOS input device, BlackHole and the like first."""
    sc = _soundcard()
    try:
        return sorted(sc.all_microphones(), key=lambda m: not _mac_virtual(m))
    except Exception as e:
        raise AudioUnavailable(f"cannot list audio devices: {type(e).__name__}: {e}") from e


def list_loopback_devices() -> list[LoopbackDevice]:
    """The macOS input devices; a virtual one is a default. CoreAudio ids are small integers that
    change across reboots, so the name is the id."""
    return [LoopbackDevice(str(m.name), str(m.name), _mac_virtual(m)) for m in _mac_inputs()]


def _mac_input(selector: str) -> Any:
    mics = _mac_inputs()
    if not selector:
        mic = next((m for m in mics if _mac_virtual(m)), None)
        if mic is None:  # never the microphone: it would hear the room, not the game
            raise AudioUnavailable(MAC_NO_GAME_AUDIO)
        return mic
    wanted = selector.lower()
    mic = next((m for m in mics if selector in (str(m.id), str(m.name))), None) or next(
        (m for m in mics if selector in str(m.id) or wanted in str(m.name).lower()), None
    )
    if mic is None:
        raise AudioUnavailable(f"no input device matches {selector!r}; pick one on the Setup tab")
    return mic


def open_loopback(
    selector: str = "", *, sample_rate: int = 22050, blocksize: int = 1024
) -> SoundcardSource:
    """Open a macOS input device by exact id or name, else by id substring or case-insensitive
    name substring. An empty selector opens the first virtual device (BlackHole or the like)."""
    return SoundcardSource(_mac_input(selector), sample_rate, blocksize)


def mac_device_name(selector: str = "") -> str:
    """The name of the device `open_loopback(selector)` would open; raises the same errors."""
    return str(_mac_input(selector).name)


def open_game_audio(
    window: WindowRef | None,
    device: str = "",
    *,
    sample_rate: int = 22050,
    blocksize: int = 1024,
) -> Any:
    """The audio of the app that owns `window`; on macOS, the input `device` instead."""
    if not audio_follows_window():
        return open_loopback(device, sample_rate=sample_rate, blocksize=blocksize)
    if window is None:
        raise AudioUnavailable("pick the game window first")
    if sys.platform == "linux":
        from wowfish.audio._pipewire import open_app_stream

        return open_app_stream(window, sample_rate=sample_rate, blocksize=blocksize)
    if window.pid is None:
        raise AudioUnavailable(
            f"cannot tell which process owns {window.title!r}; pick another window"
        )
    from wowfish.audio._wasapi import ProcessLoopbackSource

    return ProcessLoopbackSource(window.pid, sample_rate=sample_rate, blocksize=blocksize)
