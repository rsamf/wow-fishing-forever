"""Linux: capture one application's audio stream through PipeWire.

`pw-dump` lists every node; an application's playback stream is a node with
media.class "Stream/Output/Audio" whose properties name the application and its process. A
capture stream whose target is that node is linked to the stream's output ports and hears that
application alone, whatever else plays on the speakers. `pw-record --target <serial>` does
exactly that, so the source is a pw-record subprocess writing raw float32 mono to a pipe.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from wowfish.audio import AudioUnavailable
from wowfish.capture import WindowRef

PW_DUMP = ["pw-dump"]
PW_RECORD = "pw-record"
STREAM_CLASS = "Stream/Output/Audio"


@dataclass(frozen=True)
class AppStream:
    serial: int  # object.serial: what `pw-record --target` accepts
    name: str  # application.name, or node.name
    pid: int | None


def _run_pw_dump() -> str:
    try:
        return subprocess.run(PW_DUMP, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        raise AudioUnavailable(f"cannot run pw-dump (is PipeWire running?): {e}") from e


def pipewire_available() -> tuple[bool, str]:
    """Whether one application's audio can be captured here, and what is missing if not."""
    for tool in (PW_RECORD, PW_DUMP[0]):
        if shutil.which(tool) is None:
            return False, f"{tool} not found; install PipeWire's command-line tools"
    try:
        _run_pw_dump()
    except AudioUnavailable as e:
        return False, str(e)
    return True, f"PipeWire per-app capture ({PW_RECORD})"


def list_app_streams(*, dump: Callable[[], str] | None = None) -> list[AppStream]:
    """Every application playback stream PipeWire knows about, in dump order."""
    text = (dump or _run_pw_dump)()
    try:
        objects = json.loads(text)
    except ValueError as e:
        raise AudioUnavailable(f"unexpected pw-dump output: {e}") from e

    def props_of(obj: object) -> dict:
        return ((obj.get("info") or {}).get("props") or {}) if isinstance(obj, dict) else {}

    # A native PipeWire client's stream carries no pid; its client object does (the pid the
    # daemon read off the socket). Streams through pipewire-pulse (Wine) name their own.
    client_pids: dict[object, int] = {}
    for obj in objects:
        if isinstance(obj, dict) and str(obj.get("type", "")).endswith(":Client"):
            pid = props_of(obj).get("application.process.id", props_of(obj).get("pipewire.sec.pid"))
            if pid is not None:
                client_pids[obj.get("id")] = int(pid)
    out: list[AppStream] = []
    for obj in objects:
        props = props_of(obj)
        if props.get("media.class") != STREAM_CLASS or "object.serial" not in props:
            continue
        pid = props.get("application.process.id", client_pids.get(props.get("client.id")))
        out.append(
            AppStream(
                int(props["object.serial"]),
                str(props.get("application.name") or props.get("node.name") or "?"),
                int(pid) if pid is not None else None,
            )
        )
    return out


def _parent_pid(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as f:
            return int(f.read().rsplit(")", 1)[1].split()[1])  # after "(comm)": state, ppid
    except (OSError, ValueError, IndexError):
        return None


def _descends_from(pid: int, ancestor: int, parent_of: Callable[[int], int | None]) -> bool:
    seen: set[int] = set()
    current: int | None = pid
    while current is not None and current > 1 and current not in seen and len(seen) < 64:
        seen.add(current)
        current = parent_of(current)
        if current == ancestor:
            return True
    return False


def pick_stream(
    streams: Sequence[AppStream], pid: int, parent_of: Callable[[int], int | None] = _parent_pid
) -> AppStream | None:
    """The stream of process `pid`, else of a process it started (as Windows captures a process
    tree). Never by name: the user picked the window, and only its app is heard."""
    for s in streams:
        if s.pid == pid:
            return s
    for s in streams:
        if s.pid is not None and _descends_from(s.pid, pid, parent_of):
            return s
    return None


class PipewireAppSource:
    """AudioSource over `pw-record --target SERIAL` writing raw float32 mono to stdout."""

    def __init__(
        self,
        serial: int,
        app_name: str,
        *,
        sample_rate: int = 22050,
        blocksize: int = 1024,
        command: Sequence[str] | None = None,
    ) -> None:
        self.serial = serial
        self.app_name = app_name
        self.name = f"{app_name} (game audio only)"
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.command = list(command) if command is not None else self._default_command()
        self._proc: subprocess.Popen[bytes] | None = None

    def _default_command(self) -> list[str]:
        return [
            PW_RECORD, "--target", str(self.serial), "--rate", str(self.sample_rate),
            "--channels", "1", "--format", "f32", "--raw", "-",
        ]  # fmt: skip

    def __enter__(self) -> PipewireAppSource:
        try:
            self._proc = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
        except OSError as e:
            raise AudioUnavailable(f"cannot start {self.command[0]}: {e}") from e
        return self

    def __exit__(self, *exc: object) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        if proc.stdout is not None:
            proc.stdout.close()

    def read(self, n_frames: int) -> np.ndarray:
        """Block until n_frames arrive. Raises RuntimeError when the stream ends (app closed)."""
        assert self._proc is not None and self._proc.stdout is not None, "use as a context manager"
        want = n_frames * 4
        fd = self._proc.stdout.fileno()
        parts: list[bytes] = []
        got = 0
        while got < want:
            chunk = os.read(fd, want - got)
            if not chunk:
                raise RuntimeError(f"the audio stream of {self.app_name!r} ended")
            parts.append(chunk)
            got += len(chunk)
        return np.frombuffer(b"".join(parts), dtype="<f4").astype(np.float32, copy=False)


def open_app_stream(
    window: WindowRef,
    *,
    sample_rate: int = 22050,
    blocksize: int = 1024,
    dump: Callable[[], str] | None = None,
) -> PipewireAppSource:
    """Capture the audio of the app that owns `window`."""
    if shutil.which(PW_RECORD) is None:
        raise AudioUnavailable(
            "pw-record not found; install PipeWire's command-line tools "
            "(package pipewire-bin, pipewire-utils, or pipewire-tools)"
        )
    if window.pid is None:
        raise AudioUnavailable(
            f"cannot tell which process owns {window.title!r}; pick another window"
        )
    stream = pick_stream(list_app_streams(dump=dump), window.pid)
    if stream is None:
        raise AudioUnavailable(
            f"no sound from {window.title!r} (pid {window.pid}); is its sound on? "
            "Make it play a sound and try again"
        )
    return PipewireAppSource(
        stream.serial, stream.name, sample_rate=sample_rate, blocksize=blocksize
    )
