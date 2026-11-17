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

from wowfish.capture import Box, WindowRef, best_title_match
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


def _client_ref(c: dict[str, Any], scales: dict[Any, float]) -> WindowRef:
    """A client as a WindowRef sized in its output's pixels.

    Hyprland reports Xwayland windows in the same logical pixels as native ones, whatever
    xwayland:force_zero_scaling makes their X11 size, so all compare alike. An Xwayland window
    comes back without a handle or compositor: X11 grabs it.
    """
    (x, y), (w, h) = c["at"], c["size"]
    scale = scales.get(c.get("monitor"), 1.0)
    pid = c.get("pid")
    native = not c.get("xwayland")
    return WindowRef(
        str(c.get("title") or ""),
        int(pid) if pid and pid > 0 else None,
        int(x),
        int(y),
        int(w * scale),  # grim sizes its image the same way (C truncation)
        int(h * scale),
        int(str(c["address"]), 16) if native else None,
        "hyprland" if native else None,
        scale,
    )


def _scales(monitors: Sequence[dict[str, Any]]) -> dict[Any, float]:
    return {m.get("id"): float(m.get("scale") or 1.0) for m in monitors}


def hyprland_windows(
    clients: Sequence[dict[str, Any]], monitors: Sequence[dict[str, Any]]
) -> list[WindowRef]:
    """Every mapped, shown, titled window, native Wayland and Xwayland alike."""
    scales = _scales(monitors)
    return [
        _client_ref(c, scales)
        for c in clients
        if c.get("title") and c.get("mapped", True) and not c.get("hidden")
    ]


def pick_hyprland_window(
    clients: Sequence[dict[str, Any]], monitors: Sequence[dict[str, Any]], title_substring: str
) -> WindowRef | None:
    """The largest mapped window whose title contains the text, sized in its output's pixels."""
    return best_title_match(hyprland_windows(clients, monitors), title_substring)


def find_window_hyprland(title_substring: str) -> WindowRef | None:
    return pick_hyprland_window(
        hyprland_request("clients"), hyprland_request("monitors"), title_substring
    )


def refresh_hyprland(
    clients: Sequence[dict[str, Any]], monitors: Sequence[dict[str, Any]], window: WindowRef
) -> WindowRef | None:
    """The window's client as it is now, by address; None once Hyprland no longer lists it."""
    for c in clients:
        if c.get("xwayland") or int(str(c.get("address") or "0"), 16) != window.handle:
            continue
        if not c.get("mapped", True) or c.get("hidden"):
            return window  # still there, just not shown
        return _client_ref(c, _scales(monitors))
    return None


def list_windows_hyprland() -> list[WindowRef]:
    """The native Wayland windows; Xwayland ones are listed through X11, which can grab them."""
    found = hyprland_windows(hyprland_request("clients"), hyprland_request("monitors"))
    return [r for r in found if r.compositor]


def on_shown_workspace(client: dict[str, Any], monitors: Sequence[dict[str, Any]]) -> bool:
    """Whether the client's workspace is the one its monitor is showing (grim sees the screen)."""
    workspace = (client.get("workspace") or {}).get("id")
    for m in monitors:
        if m.get("id") == client.get("monitor"):
            return workspace in (
                (m.get("activeWorkspace") or {}).get("id"),
                (m.get("specialWorkspace") or {}).get("id"),
            )
    return False


def _client_of(window: WindowRef, clients: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    return next(
        (c for c in clients if int(str(c.get("address") or "0"), 16) == window.handle), None
    )


def snapshot_hyprland(window: WindowRef) -> np.ndarray | None:
    """The window's part of the screen, or None when it is on a workspace that is not shown."""
    client = _client_of(window, hyprland_request("clients"))
    if client is None or not on_shown_workspace(client, hyprland_request("monitors")):
        return None
    cap = GrimCapturer()
    cap.open(window)
    try:
        return cap.grab()
    finally:
        cap.close()


def _data_dirs() -> list[str]:
    home = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return [home, *(os.environ.get("XDG_DATA_DIRS") or "/usr/local/share:/usr/share").split(":")]


def _desktop_icon_name(app_class: str) -> str | None:
    """The Icon= of the .desktop file named after the window class (or claiming it)."""
    wanted = app_class.lower()
    for base in _data_dirs():
        folder = os.path.join(base, "applications")
        try:
            names = sorted(n for n in os.listdir(folder) if n.endswith(".desktop"))
        except OSError:
            continue
        for name in names:
            try:
                with open(os.path.join(folder, name), encoding="utf-8", errors="replace") as f:
                    lines = f.read().splitlines()
            except OSError:
                continue
            entry = dict(line.split("=", 1) for line in lines if "=" in line)
            claims = entry.get("StartupWMClass", "").strip().lower()
            if wanted in (name[: -len(".desktop")].lower(), claims) and entry.get("Icon"):
                return entry["Icon"].strip()
    return None


def icon_hyprland(window: WindowRef, size: int) -> np.ndarray | None:
    """The app's PNG icon from its .desktop file and the hicolor theme; None for anything else."""
    import cv2

    client = _client_of(window, hyprland_request("clients"))
    icon = _desktop_icon_name(str((client or {}).get("class") or ""))
    if not icon:
        return None
    paths = [icon] if os.path.isabs(icon) else []
    for base in _data_dirs():
        paths += [
            os.path.join(base, "icons", "hicolor", f"{px}x{px}", "apps", f"{icon}.png")
            for px in (48, 64, 32, 128, 256)
        ]
    paths.append(f"/usr/share/pixmaps/{icon}.png")
    for path in paths:
        if path.endswith(".png") and os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None or img.ndim != 3:
                continue
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return None


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
