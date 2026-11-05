"""Command-line entry point. Subcommands are registered by other modules."""

from __future__ import annotations

from pathlib import Path

import typer

from wowfish import __version__
from wowfish.commands._common import fail

_STDERR_FORMAT = "<level>{level: <7}</level> {message}"
_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | {name}:{line} | {message}"

app = typer.Typer(
    name="wowfish",
    help="A fishing bot for World of Warcraft: Forever.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"wowfish {__version__}")
        raise typer.Exit()


def _setup_logging(verbose: bool = False) -> Path:
    import sys

    from loguru import logger

    from wowfish import paths

    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "INFO", format=_STDERR_FORMAT)
    log_path = paths.log_file()
    logger.add(log_path, level="DEBUG", format=_FILE_FORMAT, rotation="5 MB", retention=3)
    return log_path


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print the version and exit.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    _setup_logging(verbose=verbose)


def main() -> None:
    app()


@app.command()
def config(
    path_only: bool = typer.Option(False, "--path", help="Print only the config file path."),
) -> None:
    """Show the config file location and its resolved contents."""
    from wowfish import paths
    from wowfish.config import ConfigError, ensure_config_file, load_config, to_toml

    cfg_path = paths.config_file()
    if path_only:
        typer.echo(str(cfg_path))
        return
    if ensure_config_file(cfg_path):
        typer.echo(f"created {cfg_path} with defaults")
    try:
        cfg, warnings = load_config(cfg_path)
    except ConfigError as e:
        fail(str(e))
    for w in warnings:
        typer.echo(f"warning: {w}", err=True)
    typer.echo(f"# {cfg_path}")
    typer.echo(to_toml(cfg), nl=False)


@app.command()
def doctor() -> None:
    """Check this machine: display, OpenCV, audio, config. Exits 1 only on hard failures."""
    from wowfish import doctor as doc

    results = doc.run_probes(doc.default_probes())
    typer.echo(doc.render(results))
    raise typer.Exit(code=doc.exit_code(results))


from wowfish.commands import register_all  # noqa: E402

register_all(app)
