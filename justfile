set shell := ["bash", "-cu"]

default:
    @just --list

# Install everything including dev tools
sync:
    uv sync --group dev

# Lint and format check
lint:
    uv run ruff check .
    uv run ruff format --check .

# Auto-fix lint and format
fmt:
    uv run ruff check --fix .
    uv run ruff format .

# Unit tests (no network)
test:
    uv run pytest

# Build the single-file executable into dist/
build:
    uv run pyinstaller --clean --noconfirm packaging/wowfish.spec

# Run the built executable the way CI does
smoke:
    ./dist/wowfish --version
    ./dist/wowfish doctor
