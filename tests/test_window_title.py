from __future__ import annotations

import sys
from itertools import count
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from wowfish import __version__, paths
from wowfish.config import ConfigError, load_config
from wowfish.gui import app as gui_app
from wowfish.gui.presenter import Presenter


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("WOWFISH_HOME", str(tmp_path))


def test_window_title_preferences_are_saved():
    path = paths.config_file()
    path.write_text('[game]\ncast_key = "7"\n')
    presenter = Presenter()
    assert presenter.cfg.gui.randomize_title is False
    assert presenter.cfg.gui.title_interval_s == 300

    presenter.save_window_title(True, "45")

    cfg, warnings = load_config()
    assert not warnings
    assert cfg.gui.randomize_title is True
    assert cfg.gui.title_interval_s == 45
    assert cfg.game.cast_key == "7"


@pytest.mark.parametrize("interval", ["", "0", "-1", "1.5", "2147484"])
def test_invalid_window_title_interval_cannot_be_saved(interval):
    path = paths.config_file()
    original = "[gui]\nrandomize_title = true\ntitle_interval_s = 45\n"
    path.write_text(original)

    with pytest.raises(ValueError, match="whole seconds"):
        Presenter().save_window_title(False, interval)

    assert path.read_text() == original


def test_invalid_window_title_interval_in_config_is_rejected():
    paths.config_file().write_text("[gui]\nrandomize_title = true\ntitle_interval_s = 0\n")

    with pytest.raises(ConfigError, match=r"gui\.title_interval_s"):
        load_config()


class _Window:
    """Stand in for the display and advance its scheduled callbacks without waiting."""

    def __init__(self, *, title, **_kwargs):
        self.caption = title
        self.pending = {}
        self._ids = count()

    def title(self, caption):
        self.caption = caption

    def after(self, ms, callback):
        timer = str(next(self._ids))
        self.pending[timer] = (ms, callback)
        return timer

    def after_cancel(self, timer):
        del self.pending[timer]

    def protocol(self, *_args):
        pass

    def destroy(self):
        pass


def test_window_title_rotates_and_follows_saved_preferences(monkeypatch):
    ttkbootstrap = SimpleNamespace(Window=_Window, Notebook=Mock())
    monkeypatch.setitem(sys.modules, "ttkbootstrap", ttkbootstrap)
    setup = Mock()
    monkeypatch.setattr(gui_app, "SetupPage", setup)
    monkeypatch.setattr(gui_app, "FishPage", Mock())
    monkeypatch.setattr(Presenter, "prerequisites", lambda self: [])
    paths.config_file().write_text("[gui]\nrandomize_title = true\ntitle_interval_s = 2\n")

    gui = gui_app.App()
    window = gui.root
    default = f"wowfish {__version__}"
    first = window.caption
    assert first and first != default
    (timer,) = [key for key, (ms, _) in window.pending.items() if ms == 2000]
    _, rotate = window.pending.pop(timer)
    rotate()
    assert window.caption not in (default, first, "")
    assert sorted(ms for ms, _ in window.pending.values()) == [50, 2000]

    apply_settings = setup.call_args.kwargs["on_title_change"]
    gui.p.save_window_title(True, "4")
    apply_settings()
    assert sorted(ms for ms, _ in window.pending.values()) == [50, 4000]

    gui.p.save_window_title(False, "4")
    apply_settings()
    assert window.caption == default
    assert [ms for ms, _ in window.pending.values()] == [50]

    gui.p.save_window_title(True, "4")
    apply_settings()
    assert window.caption != default
    gui.close()
    assert not window.pending
