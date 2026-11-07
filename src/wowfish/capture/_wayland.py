"""Linux Wayland: find a native Wayland game window through Hyprland's IPC and grab it with grim.

A game running as a Wayland client (e.g. Proton with PROTON_ENABLE_WAYLAND=1) has no X11
window, so XGetImage cannot see it. Hyprland lists every window with its logical position and
output; grim reads the composited screen through wlr-screencopy at that output's scale, so the
pixels are the ones the game drew. The window has to be on screen, which the engine already
requires: it only grabs while the game has keyboard focus.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
from collections.abc import Sequence
from typing import Any

import numpy as np

from wowfish.capture import Box, WindowRef
from wowfish.errors import NotSupported

GRIM = "grim"
_PPM_HEADER = re.compile(rb"P6\s+(\d+)\s+(\d+)\s+255\s")


def _hyprland_socket() -> str:
    sig = os.environ.get("HYPRLAND_INSTANCE_SIGNATURE", "")
    runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    for base in (os.path.join(runtime, "hypr"), "/tmp/hypr"):
        path = os.path.join(base, sig, ".socket.sock")
        if os.path.exists(path):
            return path
    raise NotSupported(f"Hyprland IPC (no .socket.sock for instance {sig!r})")


def hyprland_request(command: str) -> Any:
    """Send one request to Hyprland's IPC socket and return the JSON reply (`hyprctl -j`)."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(2.0)
        s.connect(_hyprland_socket())
        s.sendall(f"j/{command}".encode())
        parts: list[bytes] = []
        while chunk := s.recv(65536):
            parts.append(chunk)
    return json.loads(b"".join(parts) or b"{}")


def pick_hyprland_window(
    clients: Sequence[dict[str, Any]], monitors: Sequence[dict[str, Any]], title_substring: str
) -> WindowRef | None:
    """The largest mapped window whose title contains the text, sized in its output's pixels.

    Xwayland windows are candidates too. Hyprland reports them in the same logical pixels as
    native ones, whatever xwayland:force_zero_scaling makes their X11 size, so all compare
    alike. An Xwayland winner comes back without a handle or compositor: X11 grabs it.
    """
    scales = {m.get("id"): float(m.get("scale") or 1.0) for m in monitors}
    wanted = title_substring.lower()
    found: list[WindowRef] = []
    for c in clients:
        title = str(c.get("title") or "")
        if wanted not in title.lower() or not c.get("mapped", True) or c.get("hidden"):
            continue
        (x, y), (w, h) = c["at"], c["size"]
        scale = scales.get(c.get("monitor"), 1.0)
        pid = c.get("pid")
        native = not c.get("xwayland")
        found.append(
            WindowRef(
                title,
                int(pid) if pid and pid > 0 else None,
                int(x),
                int(y),
                int(w * scale),  # grim sizes its image the same way (C truncation)
                int(h * scale),
                int(str(c["address"]), 16) if native else None,
                "hyprland" if native else None,
                scale,
            )
        )
    return max(found, key=lambda r: r.width * r.height, default=None)


def find_window_hyprland(title_substring: str) -> WindowRef | None:
    return pick_hyprland_window(
        hyprland_request("clients"), hyprland_request("monitors"), title_substring
    )


def active_window_hyprland() -> int | None:
    """The address of the window with keyboard focus, or None when nothing has it."""
    address = hyprland_request("activewindow").get("address")
    return int(str(address), 16) if address else None


def _ppm_to_bgr(data: bytes) -> np.ndarray:
    m = _PPM_HEADER.match(data)
    if m is None:
        raise RuntimeError("grim returned something other than a binary PPM")
    w, h = int(m.group(1)), int(m.group(2))
    rgb = np.frombuffer(data, dtype=np.uint8, count=w * h * 3, offset=m.end()).reshape(h, w, 3)
    return np.ascontiguousarray(rgb[:, :, ::-1])


class GrimCapturer:
    """The window's screen area through `grim`; a box is cut from the whole window.

    A whole 4K window takes about 35 ms. Grabbing only the box would need logical coordinates,
    which land between pixels on a fractionally scaled output and come back resampled.
    """

    def __init__(self) -> None:
        self._window: WindowRef | None = None

    def open(self, window: WindowRef) -> None:
        if shutil.which(GRIM) is None:
            raise NotSupported("Wayland capture without grim (install the grim package)")
        self._window = window

    def grab(self, box: Box | None = None) -> np.ndarray:
        if self._window is None:
            raise RuntimeError("capturer is not open")
        w = self._window
        geometry = f"{w.left},{w.top} {round(w.width / w.scale)}x{round(w.height / w.scale)}"
        proc = subprocess.run(
            [GRIM, "-t", "ppm", "-s", str(w.scale), "-g", geometry, "-"],
            capture_output=True,
            stdin=subprocess.DEVNULL,
        )
        if proc.returncode != 0:
            why = proc.stderr.decode("utf-8", "replace").strip()
            raise RuntimeError(f"grim failed: {why or f'exit {proc.returncode}'}")
        frame = _ppm_to_bgr(proc.stdout)
        if box is None:
            return frame
        x, y, bw, bh = box
        return np.ascontiguousarray(frame[y : y + bh, x : x + bw])

    def close(self) -> None:
        self._window = None
