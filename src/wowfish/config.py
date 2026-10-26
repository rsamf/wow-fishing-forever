"""User configuration: one TOML file, defaults merged in, unknown keys warned about."""

from __future__ import annotations

import re
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import tomli_w

from wowfish import paths
from wowfish.assets import CLIP_FAMILIES, default_config_path

_LINE_RE = re.compile(r"\(at line (\d+), column \d+\)")


class ConfigError(Exception):
    def __init__(self, path: Path, message: str, line: int | None = None) -> None:
        self.path = path
        self.message = message
        self.line = line
        super().__init__(str(self))

    def __str__(self) -> str:
        where = f"{self.path} line {self.line}" if self.line is not None else str(self.path)
        return f"Config error in {where}: {self.message}"


@dataclass(frozen=True)
class GameConfig:
    cast_key: str = "1"
    interact_key: str = "f"
    session_minutes: int = 0


@dataclass(frozen=True)
class VisionConfig:
    icon_size_px: int = 0
    icon_native_size: int = 0
    icon_threshold: float = 0.75
    match_width: int = 1920


@dataclass(frozen=True)
class AudioConfig:
    capture: str = "game"  # "game": the game's own audio stream; "device": the loopback `device`
    device: str = ""
    bite_clip: str = "fishingbobber"
    bite_threshold: float = 0.66
    refractory_s: float = 3.0

    @property
    def selector(self) -> str:
        """What to open: "game", or the loopback device selector ("" = default output monitor)."""
        return "game" if self.capture == "game" else self.device


@dataclass(frozen=True)
class RunConfig:
    icon_timeout_s: float = 4.0
    cast_timeout_s: float = 30.0
    loot_timeout_s: float = 3.0
    pause_min_s: float = 0.5
    pause_max_s: float = 2.0
    key_hold_ms: int = 60


@dataclass(frozen=True)
class Config:
    game: GameConfig
    vision: VisionConfig
    audio: AudioConfig
    run: RunConfig = field(default_factory=RunConfig)


_SECTIONS: dict[str, type] = {
    "game": GameConfig,
    "vision": VisionConfig,
    "audio": AudioConfig,
    "run": RunConfig,
}


def ensure_config_file(path: Path | None = None) -> bool:
    path = path or paths.config_file()
    if path.exists():
        return False
    path.write_text(default_config_path().read_text(encoding="utf-8"), encoding="utf-8")
    return True


def load_config(path: Path | None = None) -> tuple[Config, list[str]]:
    path = path or paths.config_file()
    if not path.exists():
        return Config(game=GameConfig(), vision=VisionConfig(), audio=AudioConfig()), []
    try:
        raw = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as e:
        m = _LINE_RE.search(str(e))
        raise ConfigError(path, str(e), int(m.group(1)) if m else None) from e

    warnings: list[str] = []
    built: dict[str, Any] = {}
    for section, cls in _SECTIONS.items():
        table = raw.pop(section, {})
        if not isinstance(table, dict):
            raise ConfigError(path, f"[{section}] must be a table")
        built[section] = _build_section(path, section, cls, table, warnings)
    for leftover in raw:
        warnings.append(f"unknown section [{leftover}] ignored")
    cfg = Config(**built)
    _validate(path, cfg)
    return cfg, warnings


def save_config(cfg: Config, path: Path | None = None) -> None:
    path = path or paths.config_file()
    path.write_text(to_toml(cfg))


def _build_section(
    path: Path, section: str, cls: type, table: dict[str, Any], warnings: list[str]
) -> Any:
    known = {f.name: f for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in table.items():
        if key not in known:
            warnings.append(f"unknown key {section}.{key} ignored")
            continue
        expected = _expected_type(known[key].type)
        if expected is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
            raise ConfigError(
                path, f"{section}.{key} must be {expected.__name__}, got {type(value).__name__}"
            )
        kwargs[key] = value
    return cls(**kwargs)


def _expected_type(annotation: Any) -> type:
    # Annotations are strings because of `from __future__ import annotations`.
    text = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
    base = text.replace(" | None", "").strip()
    return {"str": str, "int": int, "bool": bool, "float": float}[base]


def _validate(path: Path, cfg: Config) -> None:
    def check(ok: bool, key: str, rule: str) -> None:
        if not ok:
            raise ConfigError(path, f"{key} {rule}")

    check(cfg.game.session_minutes >= 0, "game.session_minutes", "must be 0 (forever) or positive")
    v, a = cfg.vision, cfg.audio
    check(0.0 < v.icon_threshold <= 1.0, "vision.icon_threshold", "must be in (0, 1]")
    check(v.match_width >= 320, "vision.match_width", "must be at least 320")
    check(v.icon_size_px >= 0, "vision.icon_size_px", "must be 0 or positive")
    check(v.icon_native_size >= 0, "vision.icon_native_size", "must be 0 or positive")
    check(a.capture in ("game", "device"), "audio.capture", 'must be "game" or "device"')
    check(0.0 < a.bite_threshold <= 1.0, "audio.bite_threshold", "must be in (0, 1]")
    check(a.refractory_s >= 0.0, "audio.refractory_s", "must be 0 or positive")
    families = ", ".join(CLIP_FAMILIES)
    check(a.bite_clip in CLIP_FAMILIES, "audio.bite_clip", f"must be one of: {families}")
    r = cfg.run
    check(r.icon_timeout_s > 0, "run.icon_timeout_s", "must be positive")
    check(r.cast_timeout_s > 0, "run.cast_timeout_s", "must be positive")
    check(r.loot_timeout_s > 0, "run.loot_timeout_s", "must be positive")
    check(r.pause_min_s >= 0, "run.pause_min_s", "must be 0 or positive")
    check(r.pause_min_s <= r.pause_max_s, "run.pause_min_s", "must not exceed run.pause_max_s")
    check(10 <= r.key_hold_ms <= 500, "run.key_hold_ms", "must be between 10 and 500")


def to_toml(cfg: Config) -> str:
    return tomli_w.dumps(asdict(cfg))
