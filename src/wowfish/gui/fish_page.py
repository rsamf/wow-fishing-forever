"""Fish tab. Left: Start/Stop, the live picture of the game with the search zone on it, and the
Scores graph (the clip scores of the current cast). Right: the state, what the bot did and will
do next, the stats, and the event log down the rest of the height."""

from __future__ import annotations

from typing import Any

from wowfish.gui.live_picture import LivePicture
from wowfish.gui.presenter import Presenter, View
from wowfish.gui.score_graph import ScoreGraph
from wowfish.gui.window_picker import pick_window

ZONE_NOTE = (
    "Do not move or zoom the camera. Shrinking the icon detection zone finds the gear faster"
)

LEFT_SHARE, RIGHT_SHARE = 3, 2  # how extra window width is split: picture side, log side
PICTURE_SHARE, SCORES_SHARE = 2, 1  # how extra height on the left is split
# The status lines ask for a fixed width, so a long line wraps instead of widening the window.
STATUS_CHARS = 30
LOG_CHARS = 44  # the log's asked-for width; the layout widens it with the window


class FishPage:
    def __init__(self, parent: Any, presenter: Presenter) -> None:
        from tkinter.scrolledtext import ScrolledText

        import ttkbootstrap as tb

        self.p = presenter
        self.frame = tb.Frame(parent, padding=10)
        # Two columns that both widen with the window. Left: Start/Stop, the game's picture, its
        # note, the Scores graph; extra height is split PICTURE_SHARE : SCORES_SHARE. Right: what
        # the bot is doing, then the log down the rest of the height.
        self.frame.columnconfigure(0, weight=LEFT_SHARE)
        self.frame.columnconfigure(1, weight=RIGHT_SHARE)
        self.frame.rowconfigure(0, weight=1)
        left = tb.Frame(self.frame)
        left.grid(row=0, column=0, sticky="nsew")
        right = tb.Frame(self.frame)
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=PICTURE_SHARE)
        left.rowconfigure(3, weight=SCORES_SHARE)
        top = tb.Frame(left)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.start = tb.Button(
            top, text="Start", command=self._start, bootstyle="success", width=10
        )
        self.start.pack(side="left")
        self.stop = tb.Button(
            top, text="Stop", command=self.p.stop, bootstyle="danger", width=10, state="disabled"
        )
        self.stop.pack(side="left", padx=6)
        self.blockers = tb.Label(top, text="", bootstyle="warning")
        self.blockers.pack(side="left", padx=6)
        self.picture = LivePicture(left, self.p.set_zone, self.p.zone)
        self.picture.canvas.grid(row=1, column=0, sticky="nsew")
        tb.Label(left, text=ZONE_NOTE, bootstyle="warning", anchor="center", justify="center").grid(
            row=2, column=0, sticky="ew", pady=(4, 8)
        )
        self.graph = ScoreGraph(left)
        self.graph.canvas.grid(row=3, column=0, sticky="nsew")

        right.columnconfigure(0, weight=1)
        right.rowconfigure(4, weight=1)
        self.state = tb.Label(right, text="IDLE", bootstyle="@secondary", padding=(8, 2))
        self.state.grid(row=0, column=0, sticky="w")
        self.did = tb.Label(
            right, text="", font=("TkDefaultFont", 13, "bold"), justify="left", width=STATUS_CHARS
        )
        self.did.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.next = tb.Label(
            right, text="", font=("TkDefaultFont", 11), justify="left", width=STATUS_CHARS
        )
        self.next.grid(row=2, column=0, sticky="ew", pady=(2, 0))
        self.stats = tb.Label(right, text=presenter.stats, width=STATUS_CHARS)
        self.stats.grid(row=3, column=0, sticky="ew", pady=(2, 8))
        self.log = ScrolledText(right, width=LOG_CHARS, height=6, wrap="word")
        self.log.configure(state="disabled")
        self.log.grid(row=4, column=0, sticky="nsew")
        # the status lines wrap at the column's width, whatever the window's size
        right.bind("<Configure>", lambda e: self._wrap_status(e.width))

    def _wrap_status(self, width: int) -> None:
        for label in (self.did, self.next, self.stats):
            label.configure(wraplength=max(width - 4, 100))

    def _start(self) -> None:
        reasons = self.p.prerequisites(request=True)
        if reasons:
            self.blockers.configure(text="; ".join(reasons))
            return
        self.blockers.configure(text="")
        pick_window(self.frame, self._start_with)

    def _start_with(self, window: Any) -> None:
        if window is None:
            self.blockers.configure(text="No window picked")
            return
        self.p.start(window)

    def render(self, view: View) -> None:
        self.state.configure(text=view.state)
        self.did.configure(text=f"Did: {view.did}")
        self.next.configure(text=f"Next: {view.next}")
        self.stats.configure(text=view.stats)
        self.start.configure(state="disabled" if view.running else "normal")
        self.stop.configure(state="normal" if view.running else "disabled")
        self.picture.show(view.preview, view.running)
        self.graph.show(view.trace, self.p.cfg.run.cast_timeout_s)
        if view.new_lines:
            self.log.configure(state="normal")
            for line in view.new_lines:
                self.log.insert("end", line + "\n")
            self.log.see("end")
            self.log.configure(state="disabled")
