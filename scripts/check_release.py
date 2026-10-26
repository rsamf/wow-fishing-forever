"""Preflight for release.yml. Fails fast before anything is built.

Usage: uv run python scripts/check_release.py
Prints `prerelease=true|false` on success.
"""

from __future__ import annotations

import re
import sys
from importlib.metadata import version as installed_version
from pathlib import Path

_PRERELEASE_RE = re.compile(r"^\d+(\.\d+)*(a|b|rc)\d+")


def is_prerelease(version: str) -> bool:
    return bool(_PRERELEASE_RE.match(version))


def check_assets_present(root: Path) -> None:
    assets = root / "src" / "wowfish" / "assets"
    expected = [assets / "interact_icon" / f"interact_gold_{n}.png" for n in (29, 42, 53, 79, 103)]
    clips = ("fishingbobber_1", "fishingbobber_2", "fishingbobber_3", "fishbite")
    expected += [assets / "clips" / f"{n}.wav" for n in clips]
    expected += [assets / "default_config"]
    missing = [str(p.relative_to(root)) for p in expected if not p.is_file()]
    if missing:
        print(f"error: missing package assets: {', '.join(missing)}", file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    version = installed_version("wowfish")
    check_assets_present(root)
    print(f"prerelease={'true' if is_prerelease(version) else 'false'}")


if __name__ == "__main__":
    main()
