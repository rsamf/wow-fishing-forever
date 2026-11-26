"""The ttkbootstrap window: a Setup tab and a Fish tab over one Presenter."""

from __future__ import annotations

import gc

from wowfish import __version__
from wowfish.gui.fish_page import FishPage
from wowfish.gui.presenter import Presenter
from wowfish.gui.setup_page import SetupPage

POLL_MS = 50
DEFAULT_SIZE = (1200, 900)  # the window's size at launch, at 96 dpi


class App:
    def __init__(self, presenter: Presenter | None = None) -> None:
        import ttkbootstrap as tb

        self.p = presenter or Presenter()
        self.root = tb.Window(
            themename="bootstrap-dark", title=f"wowfish {__version__}", minsize=(900, 760)
        )
        nb = tb.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=8, pady=8)
        self.setup = SetupPage(nb, self.p)
        self.fish = FishPage(nb, self.p)
        nb.add(self.setup.frame, text="Setup")
        nb.add(self.fish.frame, text="Fish")
        if not self.p.prerequisites():
            nb.select(1)
        # Open larger than the least the widgets ask for, so the game view has room. Sizes are
        # for a 96 dpi screen and grow with the display's scaling.
        self.root.update_idletasks()
        scale = self.root.winfo_fpixels("1i") / 96
        width = max(self.root.winfo_reqwidth(), round(DEFAULT_SIZE[0] * scale))
        height = max(self.root.winfo_reqheight(), round(DEFAULT_SIZE[1] * scale))
        self.root.geometry(f"{width}x{height}")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._after = self.root.after(POLL_MS, self._tick)

    def tick(self) -> None:
        view = self.p.poll()
        self.fish.render(view)

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
