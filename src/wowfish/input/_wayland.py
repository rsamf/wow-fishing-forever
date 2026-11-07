"""Key presses into a native Wayland game through the compositor's virtual-keyboard protocol.

XTest only reaches X11 clients, so a game drawing through Wayland never sees those presses.
zwp_virtual_keyboard_v1 (Hyprland, Sway, and other wlroots compositors) adds a keyboard to the
seat; its keys go to whichever window has focus, like a real keyboard's. The keyboard needs an
XKB keymap: we upload the seat's own, so a key resolves to the same keycode as on the user's
layout (as XTest's keysym lookup does) and the game sees no layout change.

The protocol is spoken directly over the compositor's socket: four requests and five events do
not justify a libwayland binding.
"""

from __future__ import annotations

import array
import mmap
import os
import re
import socket
import struct
import time
from collections.abc import Callable

from wowfish.capture import WindowRef
from wowfish.errors import InputUnavailable
from wowfish.input.keys import KEYSYMS, X11_NAMES, is_supported_key, normalize_key

_DISPLAY_ID = 1
_KEYMAP_FORMAT_XKB_V1 = 1
_MAX_FDS = 28  # libwayland's limit per message
FOCUS_CACHE_S = 0.05  # the engine asks every loop pass; spare the compositor's IPC

# ---- keymap -----------------------------------------------------------------------------

_KEYCODE = re.compile(r"^\s*<([^>]+)>\s*=\s*(\d+)\s*;", re.M)
_ALIAS = re.compile(r"^\s*alias\s+<([^>]+)>\s*=\s*<([^>]+)>\s*;", re.M)
_KEY = re.compile(r"\bkey\s+<([^>]+)>\s*\{(.*?)\}\s*;", re.S)
# `[ f, F ]` or `symbols[Group1]= [ f, F ]`; skips `actions[...]= [...]` and `type[Group1]`
_LEVELS = re.compile(r"(?:(\w+)\[[^\]]*\]\s*=\s*|(?<!\w))\[([^\]]*)\]")


def keycodes_from_keymap(text: str) -> dict[str, int]:
    """Evdev keycode for every supported key name the keymap types in its first group.

    A key reachable on several keys or levels takes the lowest level, then the lowest code.
    Keysyms may be written as names (`f`, `Return`) or as hex values (`0x66`).
    """
    codes = {name: int(code) for name, code in _KEYCODE.findall(text)}
    for alias, real in _ALIAS.findall(text):
        if real in codes:
            codes.setdefault(alias, codes[real])
    by_name = {X11_NAMES[k]: k for k in KEYSYMS}
    by_value = {v: k for k, v in KEYSYMS.items()}
    best: dict[str, tuple[int, int]] = {}  # key -> (level, xkb keycode)
    for key_name, body in _KEY.findall(text):
        code = codes.get(key_name)
        if code is None:
            continue
        levels = next(
            (m.group(2) for m in _LEVELS.finditer(body) if m.group(1) in (None, "symbols")), ""
        )
        for level, sym in enumerate(s.strip() for s in levels.split(",")):
            key = by_value.get(int(sym, 16)) if sym.lower().startswith("0x") else by_name.get(sym)
            if key is not None and (level, code) < best.get(key, (99, 1 << 30)):
                best[key] = (level, code)
    return {key: code - 8 for key, (_, code) in best.items()}  # xkb keycodes are evdev + 8


# ---- wire protocol ----------------------------------------------------------------------


def _string(text: str) -> bytes:
    raw = text.encode() + b"\0"
    return struct.pack("=I", len(raw)) + raw + b"\0" * (-len(raw) % 4)


def _read_string(payload: bytes, offset: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("=I", payload, offset)
    text = payload[offset + 4 : offset + 4 + n - 1].decode("utf-8", "replace")
    return text, offset + 4 + n + (-n % 4)


def _socket_path() -> str:
    name = os.environ.get("WAYLAND_DISPLAY") or "wayland-0"
    if os.path.isabs(name):
        return name
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if not runtime:
        raise InputUnavailable("Wayland input (XDG_RUNTIME_DIR is not set)")
    return os.path.join(runtime, name)


class _Connection:
    """One Wayland client connection: send requests, dispatch events, round-trip."""

    def __init__(self) -> None:
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self._sock.connect(_socket_path())
        except OSError as e:
            self._sock.close()
            raise InputUnavailable(f"Wayland input (cannot connect to the compositor: {e})") from e
        self._next_id = _DISPLAY_ID + 1
        self._buf = b""
        self._fds: list[int] = []
        # (object id, event opcode) -> handler(payload)
        self.handlers: dict[tuple[int, int], Callable[[bytes], None]] = {
            (_DISPLAY_ID, 0): self._on_error,
        }

    def new_id(self) -> int:
        self._next_id += 1
        return self._next_id - 1

    def send(self, obj: int, opcode: int, payload: bytes = b"", fds: tuple[int, ...] = ()) -> None:
        msg = struct.pack("=II", obj, ((8 + len(payload)) << 16) | opcode) + payload
        ancillary = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", fds))] if fds else []
        try:
            self._sock.sendmsg([msg], ancillary)
        except OSError as e:
            raise InputUnavailable(f"Wayland input (the compositor went away: {e})") from e

    def take_fd(self) -> int:
        if not self._fds:
            raise InputUnavailable("Wayland input (an event's file descriptor is missing)")
        return self._fds.pop(0)

    def roundtrip(self) -> None:
        """Wait until the compositor has handled every request sent so far."""
        callback = self.new_id()
        done: list[bool] = []
        self.handlers[(callback, 0)] = lambda _: done.append(True)
        self.send(_DISPLAY_ID, 0, struct.pack("=I", callback))  # wl_display.sync
        while not done:
            self._receive()
        del self.handlers[(callback, 0)]
        for fd in self._fds:  # sent with events nobody handled
            os.close(fd)
        self._fds.clear()

    def _receive(self) -> None:
        try:
            data, ancillary, _, _ = self._sock.recvmsg(65536, socket.CMSG_SPACE(_MAX_FDS * 4))
        except OSError as e:
            raise InputUnavailable(f"Wayland input (the compositor went away: {e})") from e
        if not data:
            raise InputUnavailable("Wayland input (the compositor closed the connection)")
        for level, kind, raw in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                fds = array.array("i")
                fds.frombytes(raw[: len(raw) - len(raw) % fds.itemsize])
                self._fds.extend(fds)
        self._buf += data
        while len(self._buf) >= 8:
            obj, size_op = struct.unpack_from("=II", self._buf)
            size = size_op >> 16
            if len(self._buf) < size:
                break
            payload, self._buf = self._buf[8:size], self._buf[size:]
            handler = self.handlers.get((obj, size_op & 0xFFFF))
            if handler is not None:
                handler(payload)

    def _on_error(self, payload: bytes) -> None:
        obj, code = struct.unpack_from("=II", payload)
        message, _ = _read_string(payload, 8)
        raise InputUnavailable(f"Wayland input (protocol error {code} on object {obj}: {message})")

    def close(self) -> None:
        for fd in self._fds:
            os.close(fd)
        self._fds.clear()
        self._sock.close()


class WaylandInputDevice:
    def __init__(self) -> None:
        self._conn = _Connection()
        try:
            self._keyboard = self._setup()
        except BaseException:
            self._conn.close()
            raise
        self._focus: tuple[float, int | None] = (-1.0, None)

    def _setup(self) -> int:
        conn = self._conn
        globals_: dict[str, tuple[int, int]] = {}

        def on_global(p: bytes) -> None:
            (name,) = struct.unpack_from("=I", p)
            interface, offset = _read_string(p, 4)
            globals_[interface] = (name, struct.unpack_from("=I", p, offset)[0])

        registry = conn.new_id()
        conn.handlers[(registry, 0)] = on_global
        conn.send(_DISPLAY_ID, 1, struct.pack("=I", registry))  # wl_display.get_registry
        conn.roundtrip()
        manager_global = globals_.get("zwp_virtual_keyboard_manager_v1")
        if "wl_seat" not in globals_ or manager_global is None:
            raise InputUnavailable(
                "Wayland input on this compositor (no zwp_virtual_keyboard_manager_v1)"
            )

        def bind(interface: str, version: int) -> int:
            new = conn.new_id()
            name = struct.pack("=I", globals_[interface][0])
            # wl_registry.bind
            conn.send(registry, 0, name + _string(interface) + struct.pack("=II", version, new))
            return new

        seat_version = min(globals_["wl_seat"][1], 5)
        seat = bind("wl_seat", seat_version)
        manager = bind("zwp_virtual_keyboard_manager_v1", 1)

        keymaps: list[bytes] = []

        def on_keymap(p: bytes) -> None:
            fmt, size = struct.unpack_from("=II", p)
            fd = conn.take_fd()
            try:
                if fmt == _KEYMAP_FORMAT_XKB_V1 and size:
                    with mmap.mmap(fd, size, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ) as m:
                        keymaps.append(m[:size])
            finally:
                os.close(fd)

        seat_keyboard = conn.new_id()
        conn.handlers[(seat_keyboard, 0)] = on_keymap
        conn.send(seat, 1, struct.pack("=I", seat_keyboard))  # wl_seat.get_keyboard
        conn.roundtrip()
        if seat_version >= 3:
            conn.send(seat_keyboard, 0)  # wl_keyboard.release: the keymap is all we wanted
        if not keymaps:
            raise InputUnavailable("Wayland input (the seat has no keyboard keymap)")
        keymap = keymaps[-1].rstrip(b"\0") + b"\0"
        self._codes = keycodes_from_keymap(keymap.decode("utf-8", "replace"))

        keyboard = conn.new_id()
        conn.send(manager, 0, struct.pack("=II", seat, keyboard))  # create_virtual_keyboard
        fd = os.memfd_create("wowfish-keymap", os.MFD_CLOEXEC)
        try:
            os.write(fd, keymap)
            payload = struct.pack("=II", _KEYMAP_FORMAT_XKB_V1, len(keymap))
            conn.send(keyboard, 0, payload, fds=(fd,))  # zwp_virtual_keyboard_v1.keymap
        finally:
            os.close(fd)
        conn.roundtrip()  # a refused keyboard comes back as a protocol error here
        return keyboard

    def close(self) -> None:
        try:
            self._conn.send(self._keyboard, 3)  # zwp_virtual_keyboard_v1.destroy
            self._conn.roundtrip()
        except InputUnavailable:
            pass
        self._conn.close()

    def _key(self, code: int, pressed: bool) -> None:
        ms = int(time.monotonic() * 1000) & 0xFFFFFFFF
        self._conn.send(self._keyboard, 1, struct.pack("=III", ms, code, int(pressed)))

    def press_key(self, key: str, hold_s: float = 0.06) -> None:
        name = normalize_key(key)
        if not is_supported_key(name):
            raise InputUnavailable(f"key {key!r} (not a supported key name)")
        code = self._codes.get(name)
        if code is None:
            raise InputUnavailable(f"key {key!r} (no keycode on this keyboard layout)")
        self._key(code, True)
        self._conn.roundtrip()
        time.sleep(hold_s)
        self._key(code, False)
        self._conn.roundtrip()

    def is_game_focused(self, window: WindowRef) -> bool | None:
        if window.handle is None or window.compositor != "hyprland":
            return None
        from wowfish.capture._wayland import active_window_hyprland

        now = time.monotonic()
        checked, active = self._focus
        if now - checked >= FOCUS_CACHE_S:
            try:
                active = active_window_hyprland()
            except Exception:
                return None
            self._focus = (now, active)
        return active == window.handle


def virtual_keyboard_available() -> tuple[bool, str]:
    """Whether the compositor gives us a virtual keyboard with the seat's keymap."""
    try:
        WaylandInputDevice().close()
    except InputUnavailable as e:
        return False, str(e)
    return True, "Wayland virtual keyboard (zwp_virtual_keyboard_v1)"
