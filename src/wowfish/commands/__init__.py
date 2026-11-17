"""Typer commands, one module each. `register_all` wires them onto the app."""

from __future__ import annotations

import typer


def register_all(app: typer.Typer) -> None:
    from wowfish.commands import gui, listen, run, verify_bite, watch

    for module in (watch, listen, verify_bite, run, gui):
        module.register(app)
