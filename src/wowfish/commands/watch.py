"""`wowfish watch`: run the acquire/track icon search over files or the live game window."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from wowfish.commands._common import fail, load_config_or_fail

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def _file_frames(source: Path, fps: float) -> Iterator[tuple[float, np.ndarray, float]]:
    import cv2

    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    else:
        files = [source]
    if not files:
        fail(f"no images in {source}")
    for i, p in enumerate(files):
        frame = cv2.imread(str(p))
        if frame is None:
            fail(f"cannot read image {p}")
        yield i / fps, frame, 0.0


def _live_frames(
    title: str, fps: float, seconds: float
) -> Iterator[tuple[float, np.ndarray, float]]:
    from wowfish import capture

    window = capture.find_window(title)
    if window is not None:
        cap = capture.get_capturer(window)
    elif capture.linux_x11() or capture.linux_hyprland():
        fail(f"no window titled {title!r}; is the game running?")
    else:
        from wowfish.capture import _mss

        try:
            window = _mss.primary_monitor_window()
            cap = _mss.MssCapturer()
        except Exception as e:
            fail(f"no window titled {title!r} and no screen capture available ({e})")
        typer.echo(f"window {title!r} not found; capturing the primary monitor", err=True)
    try:
        cap.open(window)
    except Exception as e:
        fail(f"cannot open capture for {window.title!r}: {e}")
    typer.echo(f"capturing {window.title} {window.width}x{window.height} at {fps:g} fps", err=True)
    start = time.perf_counter()
    try:
        while True:
            t = time.perf_counter() - start
            if seconds and t >= seconds:
                return
            grabbed = time.perf_counter()
            frame = cap.grab()
            yield t, frame, (time.perf_counter() - grabbed) * 1000
            sleep = (1.0 / fps) - (time.perf_counter() - start - t)
            if sleep > 0:
                time.sleep(sleep)
    finally:
        cap.close()


def watch(
    source: Annotated[
        Path | None, typer.Option("--from", exists=True, help="Image file or directory of images.")
    ] = None,
    live: bool = typer.Option(False, "--live", help="Capture the game window."),
    fps: float = typer.Option(
        10.0, "--fps", min=0.1, help="Frames per second (live), or the time base for files."
    ),
    seconds: float = typer.Option(
        0.0, "--seconds", min=0.0, help="Stop after this long (live only; 0 = until Ctrl-C)."
    ),
    save_frames: Annotated[
        Path | None, typer.Option("--save-frames", help="Write annotated thumbnails here.")
    ] = None,
    title: str = typer.Option("World of Warcraft", "--title", help="Window title to capture."),
) -> None:
    """Print phase, score, and position for every frame; shows what the bot would see."""
    import cv2

    from wowfish.vision.annotate import annotate_match
    from wowfish.vision.calibrate import calibrate_icon_scale
    from wowfish.vision.matcher import MIN_TEMPLATE_PX, IconMatcher
    from wowfish.vision.search import IconSearch
    from wowfish.vision.template import load_icon_templates, template_for_size

    if (source is None) == (not live):
        fail("pass exactly one of --from PATH or --live")
    cfg = load_config_or_fail()
    templates = load_icon_templates()
    threshold = cfg.vision.icon_threshold
    frames = _live_frames(title, fps, seconds) if live else _file_frames(source, fps)
    if save_frames:
        save_frames.mkdir(parents=True, exist_ok=True)
    search: IconSearch | None = None
    try:
        for i, (t, frame, grab_ms) in enumerate(frames):
            started = time.perf_counter()
            if search is None:
                size = cfg.vision.icon_size_px
                if size == 0:
                    cal = calibrate_icon_scale(frame, templates, match_width=cfg.vision.match_width)
                    if not cal.valid:
                        typer.echo(
                            f"t={t:7.2f}s not calibrated and no gold icon in this frame; waiting"
                        )
                        continue
                    size = cal.icon_size_px
                    typer.echo(
                        f"not calibrated; using icon_size_px={size} from this frame "
                        f"(score {cal.score:.2f}); run `wowfish calibrate --save` to persist"
                    )
                matcher = IconMatcher(template_for_size(templates, size), cfg.vision.match_width)
                px = matcher.template_px(frame.shape[1])
                if px < MIN_TEMPLATE_PX:
                    fail(
                        f"icon_size_px={size} is {px} px after downscaling this "
                        f"{frame.shape[1]}-wide frame to match_width={cfg.vision.match_width} "
                        f"(minimum {MIN_TEMPLATE_PX}); run `wowfish calibrate --save` on a "
                        "screenshot with the gear icon visible"
                    )
                search = IconSearch(matcher, threshold)
            phase, m = search.step(frame)
            ms = grab_ms + (time.perf_counter() - started) * 1000  # grab + match
            found = "FOUND" if m.accepted(threshold) else "-"
            typer.echo(
                f"t={t:7.2f}s phase={phase:<7} score={m.score:5.2f} gold={m.gold:4.2f} "
                f"xy=({m.x},{m.y}) {found:<5} {ms:5.1f}ms"
            )
            if save_frames:
                thumb = annotate_match(frame, m, threshold=threshold)
                cv2.imwrite(str(save_frames / f"frame_{i:05d}.jpg"), thumb)
    except KeyboardInterrupt:
        typer.echo("stopped", err=True)


def register(app: typer.Typer) -> None:
    app.command("watch")(watch)
