"""Typer commands, one module each. `register_all` wires them onto the app."""

from __future__ import annotations

import typer


def register_all(app: typer.Typer) -> None:
    from wowfish.commands import gui, run

    for module in (run, gui):
        module.register(app)
