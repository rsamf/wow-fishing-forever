"""Helpers shared by the CLI commands."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import typer

from wowfish.config import Config, ConfigError, load_config

if TYPE_CHECKING:
    from wowfish.capture import WindowRef
    from wowfish.capture.choices import WindowChoice

TITLE_MAX = 70  # longer window titles are cut in the terminal list


def fail(message: str) -> None:
    """Print a one-line error and exit 1."""
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=1)


def load_config_or_fail() -> Config:
    try:
        cfg, warnings = load_config()
    except ConfigError as e:
        fail(str(e))
    for w in warnings:
        typer.echo(f"warning: {w}", err=True)
    return cfg


def preflight_or_fail() -> None:
    """Stop before any window is listed when the platform or a macOS permission rules it out."""
    from wowfish.engine.engine import preflight_problems

    problems = preflight_problems(request=True)
    if problems:
        fail("; ".join(problems))


def pick_window_or_fail(
    choices: list[WindowChoice] | None = None,
    *,
    title: str | None = None,
    ask: Callable[[str], str] = input,
) -> WindowRef:
    """List the open windows by number and return the one the user types (Enter = the first).

    The program never picks for the user: no windows, or the end of input, is an error.
    """
    from wowfish.errors import NotSupported

    if choices is None:
        from wowfish.capture import GAME_TITLE
        from wowfish.capture.choices import window_choices

        try:
            choices = window_choices((title or "").strip() or GAME_TITLE)
        except (NotSupported, OSError) as e:
            fail(str(e))
    if not choices:
        fail("no windows to pick from; is the game running?")
    for n, c in enumerate(choices, 1):
        w = c.window
        title = w.title if len(w.title) <= TITLE_MAX else w.title[: TITLE_MAX - 1] + "…"
        typer.echo(f"{n}) {title} — {c.app or '?'} — {w.width}x{w.height}", err=True)
    while True:
        try:
            reply = ask("Window [1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            fail("no window picked")
        if not reply:
            return choices[0].window
        if reply.isascii() and reply.isdigit() and 1 <= int(reply) <= len(choices):
            return choices[int(reply) - 1].window
        typer.echo(f"type a number from 1 to {len(choices)}", err=True)
