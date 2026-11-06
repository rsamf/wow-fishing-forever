"""`wowfish listen`: list loopback devices, meter game audio, and report bites live."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from wowfish.capture import find_window
from wowfish.commands._common import fail, load_config_or_fail

BLOCK = 2205  # 100 ms at 22050 Hz


def listen(
    list_devices: bool = typer.Option(False, "--list", help="List loopback devices and exit."),
    device: str | None = typer.Option(
        None,
        "--device",
        help='Loopback device id or name substring ("" = default output monitor); '
        "default: the config's capture setting, normally the game's own audio.",
    ),
    title: str = typer.Option("World of Warcraft", "--title", help="Game window title."),
    source_file: Annotated[
        Path | None,
        typer.Option(
            "--from", exists=True, dir_okay=False, help="Replay a WAV instead of capturing."
        ),
    ] = None,
    record: Annotated[
        Path | None, typer.Option("--record", help="Also write the raw audio to this WAV.")
    ] = None,
    seconds: float = typer.Option(
        0.0, "--seconds", min=0.0, help="Stop after this long (0 = until Ctrl-C or end of file)."
    ),
    threshold: float | None = typer.Option(
        None, "--threshold", min=0.0, max=1.0, help="Override audio.bite_threshold."
    ),
) -> None:
    """Show the audio level and the four clip scores; print BITE lines when the detector fires."""
    from wowfish.audio import AudioChunk, AudioUnavailable, devices
    from wowfish.audio._file import WavFileSource
    from wowfish.audio.detector import detector_for
    from wowfish.audio.wav import write_wav

    if list_devices:
        try:
            found = devices.list_loopback_devices()
        except AudioUnavailable as e:
            typer.echo(f"no audio backend: {e}")
            return
        if not found:
            typer.echo(
                "no loopback devices found (macOS: install a virtual device such as BlackHole)"
            )
        for d in found:
            typer.echo(f"{'*' if d.is_default else ' '} {d.name}  [{d.id}]")
        return

    cfg = load_config_or_fail()
    detector = detector_for(cfg.audio, threshold=threshold)
    if source_file is not None:
        source = WavFileSource(source_file)
    else:
        selector = device if device is not None else cfg.audio.selector
        window = find_window(title) if selector == devices.GAME else None
        try:
            source = devices.open_capture(selector, pid=window.pid if window else None, title=title)
        except AudioUnavailable as e:
            fail(str(e))
        name = getattr(source, "name", type(source).__name__)
        if selector == devices.GAME:
            typer.echo(f"listening to {title} only ({name})", err=True)
        else:
            typer.echo(f"listening on {name}", err=True)

    recorded: list[np.ndarray] = []
    frames_read = 0
    next_report = 0.0
    try:
        with source:
            while True:
                samples = source.read(BLOCK)
                frames_read += len(samples)
                t_end = frames_read / source.sample_rate
                if record is not None:
                    recorded.append(samples)
                for event in detector.feed(AudioChunk(samples, source.sample_rate, t_end)):
                    typer.echo(f"BITE {event.clip} score={event.score:.2f} at {event.t:.2f}s")
                if t_end >= next_report:
                    scores = " ".join(f"{s.name}={s.score:4.2f}" for s in detector.last_scores)
                    typer.echo(f"t={t_end:6.1f}s level={detector.level_dbfs:6.1f}dBFS {scores}")
                    next_report += 0.25
                if (seconds and t_end >= seconds) or getattr(source, "exhausted", False):
                    break
    except KeyboardInterrupt:
        typer.echo("stopped", err=True)
    finally:
        if record is not None and recorded:
            write_wav(record, np.concatenate(recorded), source.sample_rate)
            typer.echo(f"wrote {record}")


def register(app: typer.Typer) -> None:
    app.command("listen")(listen)
