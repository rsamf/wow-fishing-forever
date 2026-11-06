"""Files shipped inside the package: interact-icon sprites and the game's splash clips."""

from __future__ import annotations

import sys
from pathlib import Path

ICON_SIZES: tuple[int, ...] = (29, 42, 53, 79, 103)
# The splash sounds the game plays when a fish bites; any of them counts as the bite.
CLIP_NAMES: tuple[str, ...] = ("fishingbobber_1", "fishingbobber_2", "fishingbobber_3", "fishbite")


def asset_dir() -> Path:
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        return Path(bundle) / "wowfish" / "assets"
    return Path(__file__).resolve().parent


def icon_path(native_size: int) -> Path:
    return asset_dir() / "interact_icon" / f"interact_gold_{native_size}.png"


def clip_path(name: str) -> Path:
    return asset_dir() / "clips" / f"{name}.wav"


def default_config_path() -> Path:
    return asset_dir() / "default_config"
