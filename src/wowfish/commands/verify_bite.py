"""`wowfish verify-bite RECORDING.wav`: do the bites stand out, and what threshold to use."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

from wowfish import paths
from wowfish.commands._common import load_config_or_fail
from wowfish.config import save_config

MIN_MARGIN = 0.15


def verify_bite(
    recording: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="WAV from `wowfish listen --record`."),
    ],
    save: bool = typer.Option(
        False, "--save", help="Write the suggested bite_threshold into the config."
    ),
) -> None:
    """Score a recording against every clip; print each clip's peak, floor, and the suggestion."""
    from wowfish.audio.clips import build_fingerprints
    from wowfish.audio.features import MelParams
    from wowfish.audio.verify import bite_report, clip_peaks, score_recording
    from wowfish.audio.wav import read_wav

    cfg = load_config_or_fail()
    p = MelParams()
    fps = build_fingerprints(p)
    y, rate = read_wav(recording)
    scores = score_recording(y, rate, fps, p)
    typer.echo(f"{recording}: {len(y) / rate:.1f}s at {rate} Hz")
    peaks = clip_peaks(scores, fps, p)
    for cs in peaks:
        typer.echo(f"{cs.name:16s} peak {cs.score:5.2f} at {cs.t:6.2f}s")
    best = max(peaks, key=lambda c: c.score)
    report = bite_report(scores, fps, p)
    threshold = report.threshold
    if report.stands_out:
        lo = min(b.score for b in report.bites)
        hi = max(b.score for b in report.bites)
        when = ", ".join(f"{b.t:.1f}s ({b.name})" for b in sorted(report.bites, key=lambda b: b.t))
        typer.echo(f"{len(report.bites)} bite(s) scoring {lo:.2f}-{hi:.2f} at {when}")
        typer.echo(f"everything else scores at most {report.floor:.2f}")
    typer.echo(f"suggested bite_threshold={threshold:.2f}")
    if not report.stands_out:
        typer.echo(
            f"warning: no bite stands out (best peak {best.score:.2f}, everything else up to "
            f"{report.floor:.2f}); record again with Ambience and Music at 0 and a clear bite",
            err=True,
        )
        if save:
            typer.echo("not saved", err=True)
            raise typer.Exit(code=1)
    if save:
        audio = replace(cfg.audio, bite_threshold=threshold)
        save_config(replace(cfg, audio=audio))
        typer.echo(f"saved to {paths.config_file()}")


def register(app: typer.Typer) -> None:
    app.command("verify-bite")(verify_bite)
