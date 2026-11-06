"""Fish tab: Start/Stop, what the bot did and will do next, the frame it matched, stats, and
Scores | Logs sub-tabs (the clip scores of the current cast, or the event log)."""

from __future__ import annotations

from typing import Any

from wowfish.gui import image
from wowfish.gui.presenter import Presenter, View
from wowfish.gui.score_graph import ScoreGraph


class FishPage:
    def __init__(self, parent: Any, presenter: Presenter) -> None:
        from tkinter.scrolledtext import ScrolledText

        import ttkbootstrap as tb

        self.p = presenter
        self.frame = tb.Frame(parent, padding=10)
        f = self.frame
        top = tb.Frame(f)
        top.pack(fill="x")
        self.start = tb.Button(
            top, text="Start", command=self._start, bootstyle="success", width=10
        )
        self.start.pack(side="left")
        self.stop = tb.Button(
            top, text="Stop", command=self.p.stop, bootstyle="danger", width=10, state="disabled"
        )
        self.stop.pack(side="left", padx=6)
        self.state = tb.Label(top, text="IDLE", bootstyle="@secondary", padding=(8, 2))
        self.state.pack(side="left", padx=12)
        self.blockers = tb.Label(top, text="", bootstyle="warning")
        self.blockers.pack(side="left", padx=6)

        self.did = tb.Label(
            f, text="", font=("TkDefaultFont", 13, "bold"), wraplength=860, justify="left"
        )
        self.did.pack(anchor="w", pady=(10, 0))
        self.next = tb.Label(f, text="", font=("TkDefaultFont", 11), wraplength=860, justify="left")
        self.next.pack(anchor="w", pady=(2, 0))
        self.stats = tb.Label(f, text=presenter.stats)
        self.stats.pack(anchor="w", pady=(2, 8))

        mid = tb.Frame(f)
        mid.pack(fill="both", expand=True)
        self.image = tb.Label(mid, text="no icon seen yet", anchor="center", width=60)
        self.image.pack(side="left", anchor="n")
        self._photo = None
        right = tb.Frame(mid)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))
        tabs = tb.Notebook(right)
        tabs.pack(fill="both", expand=True)
        self.graph = ScoreGraph(tabs)
        self.log = ScrolledText(tabs, height=18, wrap="word")
        self.log.configure(state="disabled")
        tabs.add(self.graph.canvas, text="Scores")  # first, so it is the tab shown at start
        tabs.add(self.log, text="Logs")

    def _start(self) -> None:
        reasons = self.p.prerequisites()
        if reasons:
            self.blockers.configure(text="; ".join(reasons))
            return
        self.blockers.configure(text="")
        self.p.start()

    def render(self, view: View) -> None:
        self.state.configure(text=view.state)
        self.did.configure(text=f"Did: {view.did}")
        self.next.configure(text=f"Next: {view.next}")
        self.stats.configure(text=view.stats)
        self.start.configure(state="disabled" if view.running else "normal")
        self.stop.configure(state="normal" if view.running else "disabled")
        if view.thumbnail_png is not None and view.thumbnail_png is not getattr(
            self, "_last_png", None
        ):
            self._last_png = view.thumbnail_png
            self._photo = image.photo_from_png(view.thumbnail_png)
            self.image.configure(image=self._photo, text="")
        self.graph.show(view.trace, self.p.cfg.run.cast_timeout_s)
        if view.new_lines:
            self.log.configure(state="normal")
            for line in view.new_lines:
                self.log.insert("end", line + "\n")
            self.log.see("end")
            self.log.configure(state="disabled")
