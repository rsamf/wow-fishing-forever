"""Key names accepted in the config and their X11 keysym names and Windows virtual-key codes."""

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


def normalize_key(name: str) -> str:
    return name.strip().lower()


def is_supported_key(name: str) -> bool:
    return normalize_key(name) in SUPPORTED_KEYS
