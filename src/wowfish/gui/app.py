"""The ttkbootstrap window: a Setup tab and a Fish tab over one Presenter."""

from __future__ import annotations

import ctypes
import gc
import secrets
import string
import sys

from wowfish.gui.fish_page import FishPage
from wowfish.gui.presenter import Presenter
from wowfish.gui.setup_page import SetupPage

POLL_MS = 50
_ALNUM = string.ascii_letters + string.digits


def _rand_alnum(length: int = 12) -> str:
    first = secrets.choice(string.ascii_letters)
    rest = "".join(secrets.choice(_ALNUM) for _ in range(length - 1))
    return first + rest


def _cloak_linux_process(name: str) -> None:
    if sys.platform != "linux":
        return
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(15, name[:15].encode("ascii"), 0, 0, 0)
    except Exception:
        pass


class App:
    def __init__(self, presenter: Presenter | None = None) -> None:
        import ttkbootstrap as tb

        self._session_tag = _rand_alnum(12)
        _cloak_linux_process(self._session_tag)

        self.p = presenter or Presenter()
        self.root = tb.Window(
            themename="bootstrap-dark",
            title=self._session_tag,
            className=self._session_tag,
            minsize=(900, 640),
        )
        nb = tb.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=8, pady=8)
        self.setup = SetupPage(nb, self.p)
        self.fish = FishPage(nb, self.p)
        nb.add(self.setup.frame, text="Setup")
        nb.add(self.fish.frame, text="Fish")
        if not self.p.prerequisites():
            nb.select(1)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._after = self.root.after(POLL_MS, self._tick)

    def tick(self) -> None:
        view = self.p.poll()
        self.fish.render(view)
        self.setup.render(view)

    def _tick(self) -> None:
        try:
            self.tick()
        finally:
            self._after = self.root.after(POLL_MS, self._tick)

    def run(self) -> None:
        try:
            self.root.mainloop()
        finally:
            self.close()

    def close(self) -> None:
        """Stop everything and free the Tk interpreter on this thread.

        Tcl panics (an Illegal-instruction trap) if its interpreter is deleted from another
        thread, which is what happens when a later garbage collection on a worker thread frees
        a window that was merely destroyed. So drop every reference and collect here.
        """
        if self.root is None:
            return
        if self._after is not None:
            self.root.after_cancel(self._after)
            self._after = None
        self.p.shutdown()
        self.root.destroy()
        self.setup = self.fish = None  # type: ignore[assignment]
        self.root = None  # type: ignore[assignment]
        gc.collect()


def run() -> None:
    App().run()
