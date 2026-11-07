"""`wowfish run`: fish until Ctrl-C or the session limit, printing every event."""

from __future__ import annotations

import queue
import time
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from wowfish.audio import AudioUnavailable
from wowfish.commands._common import fail, load_config_or_fail
from wowfish.engine import events as ev
from wowfish.engine.engine import build_engine
from wowfish.engine.thread import EngineThread
from wowfish.errors import NotSupported
from wowfish.gui.presenter import describe


def _line(event: ev.Event) -> str:
    text = describe(event)
    logger.debug(text)  # the log file keeps the session; the terminal shows it below
    return f"{time.strftime('%H:%M:%S')}  {text}"


def run(
    minutes: float | None = typer.Option(
        None, "--minutes", min=0.0, help="Stop after this long (default: game.session_minutes)."
    ),
    title: str = typer.Option("World of Warcraft", "--title", help="Window title to drive."),
    record_audio: Annotated[
        Path | None,
        typer.Option(
            "--record-audio",
            help="Save everything the bot hears to this WAV when it stops (for `verify-bite`).",
        ),
    ] = None,
) -> None:
    """Cast, find the bobber, hear the splash, press Interact, loot, repeat. Ctrl-C stops."""
    cfg = load_config_or_fail()
    try:
        engine = build_engine(
            cfg,
            session_s=None if minutes is None else minutes * 60,
            title=title,
            record_audio=record_audio,
        )
    except (RuntimeError, NotSupported, AudioUnavailable) as e:
        fail(str(e))
    thread = EngineThread(engine)
    thread.start()
    w = engine.window
    typer.echo(f"fishing in {w.title!r} ({w.width}x{w.height}); Ctrl-C stops", err=True)
    audio_name = getattr(engine.audio, "name", type(engine.audio).__name__)
    logger.debug(f"run: window {w.width}x{w.height}, audio {audio_name}")
    try:
        while True:
            try:
                event = engine.events.get(timeout=0.2)
            except queue.Empty:
                if not thread.is_alive():
                    break
                continue
            if not isinstance(event, ev.Scores):  # the GUI's graph feed, 10 a second
                typer.echo(_line(event))
            if isinstance(event, ev.Stopped):
                break
    except KeyboardInterrupt:
        typer.echo("stopping", err=True)
        thread.stop()
        while not engine.events.empty():
            event = engine.events.get_nowait()
            if not isinstance(event, ev.Scores):
                typer.echo(_line(event))
    # The engine thread is a daemon: let it close the audio (and write the WAV) before exiting.
    thread.stop(timeout=10.0)
    if record_audio is not None:
        typer.echo(f"wrote {record_audio}", err=True)
    typer.echo(engine.stats.as_line())


def register(app: typer.Typer) -> None:
    app.command("run")(run)
