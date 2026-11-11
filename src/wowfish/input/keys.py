"""Key names accepted in the config and their X11 keysyms, Windows virtual-key codes, and macOS
virtual keycodes."""

from __future__ import annotations

import string

_LETTERS = tuple(string.ascii_lowercase)
_DIGITS = tuple(string.digits)
_FKEYS = tuple(f"f{i}" for i in range(1, 13))
_NAMED = ("space", "enter", "tab", "escape")
_PUNCT = ("-", "=", "[", "]", ";", "'", ",", ".", "/", "`", "\\")

SUPPORTED_KEYS: tuple[str, ...] = _LETTERS + _DIGITS + _FKEYS + _NAMED + _PUNCT

_X11_SPECIAL = {
    "space": "space",
    "enter": "Return",
    "tab": "Tab",
    "escape": "Escape",
    "-": "minus",
    "=": "equal",
    "[": "bracketleft",
    "]": "bracketright",
    ";": "semicolon",
    "'": "apostrophe",
    ",": "comma",
    ".": "period",
    "/": "slash",
    "`": "grave",
    "\\": "backslash",
}
X11_NAMES: dict[str, str] = {
    **{k: k for k in _LETTERS},
    **{k: k for k in _DIGITS},
    **{k: k.upper() for k in _FKEYS},
    **_X11_SPECIAL,
}

# Keysym values: printable ASCII keysyms equal their character codes.
KEYSYMS: dict[str, int] = {
    **{k: ord(k) for k in _LETTERS + _DIGITS + _PUNCT},
    **{f"f{i}": 0xFFBE + i - 1 for i in range(1, 13)},
    "space": 0x20,
    "enter": 0xFF0D,
    "tab": 0xFF09,
    "escape": 0xFF1B,
}

_VK_SPECIAL = {
    "space": 0x20,
    "enter": 0x0D,
    "tab": 0x09,
    "escape": 0x1B,
    "-": 0xBD,
    "=": 0xBB,
    "[": 0xDB,
    "]": 0xDD,
    ";": 0xBA,
    "'": 0xDE,
    ",": 0xBC,
    ".": 0xBE,
    "/": 0xBF,
    "`": 0xC0,
    "\\": 0xDC,
}
VK_CODES: dict[str, int] = {
    **{k: ord(k.upper()) for k in _LETTERS},
    **{k: ord(k) for k in _DIGITS},
    **{f"f{i}": 0x70 + i - 1 for i in range(1, 13)},
    **_VK_SPECIAL,
}

# macOS virtual keycodes (kVK_ANSI_* and friends in Carbon's Events.h). Like Windows scan codes
# they name a key position on a US ANSI keyboard, not the character it types.
MAC_KEYCODES: dict[str, int] = {
    **dict(zip("asdfhgzxcv", range(0x00, 0x0A), strict=True)),
    **dict(zip("bqwerty", range(0x0B, 0x12), strict=True)),
    "1": 0x12, "2": 0x13, "3": 0x14, "4": 0x15, "6": 0x16, "5": 0x17, "=": 0x18, "9": 0x19,
    "7": 0x1A, "-": 0x1B, "8": 0x1C, "0": 0x1D, "]": 0x1E, "o": 0x1F, "u": 0x20, "[": 0x21,
    "i": 0x22, "p": 0x23, "enter": 0x24, "l": 0x25, "j": 0x26, "'": 0x27, "k": 0x28, ";": 0x29,
    "\\": 0x2A, ",": 0x2B, "/": 0x2C, "n": 0x2D, "m": 0x2E, ".": 0x2F, "tab": 0x30,
    "space": 0x31, "`": 0x32, "escape": 0x35,
    "f1": 0x7A, "f2": 0x78, "f3": 0x63, "f4": 0x76, "f5": 0x60, "f6": 0x61, "f7": 0x62,
    "f8": 0x64, "f9": 0x65, "f10": 0x6D, "f11": 0x67, "f12": 0x6F,
}  # fmt: skip


def normalize_key(name: str) -> str:
    return name.strip().lower()


def is_supported_key(name: str) -> bool:
    return normalize_key(name) in SUPPORTED_KEYS
