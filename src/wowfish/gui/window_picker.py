"""The window picker: every open window as a card (snapshot, icon, title), the title guess first.

The user says which window is the game, as in a screen share; the bot never picks one itself.
Snapshots and icons are made on a worker thread as PNG bytes and become Tk images on the Tk
thread only.
"""

from __future__ import annotations

import contextlib
import gc
import queue
import threading
from collections.abc import Callable
from typing import Any

from wowfish.capture import WindowRef
from wowfish.capture.choices import WindowChoice, window_choices, window_icon, window_snapshot
from wowfish.errors import NotSupported
from wowfish.gui import image

SNAPSHOT_W = 240  # at 96 dpi; a HiDPI screen gets whole multiples
SNAPSHOT_H = 135  # the placeholder's height: 16:9
ICON_PX = 24
COLUMNS = 3
POLL_MS = 50


def _elide(text: str, fits: Callable[[str], bool]) -> str:
    """The text, cut and ended with dots until `fits` says it is narrow enough. Three plain
    dots: Tk's default font draws "…" as stray glyphs on some systems."""
    if fits(text):
        return text
    while len(text) > 1 and not fits(text + "..."):
        text = text[:-1]
    return text.rstrip() + "..."


class _Picker:
    def __init__(
        self,
        parent: Any,
        on_done: Callable[[WindowRef | None], None],
        choices: Callable[[], list[WindowChoice]],
        found: list[WindowChoice],
    ) -> None:
        import tkinter as tk

        import ttkbootstrap as tb

        self._tk, self._tb = tk, tb
        self._on_done = on_done
        self._choices = choices
        self._found: list[WindowChoice] = []
        self._selected = 0
        self._generation = 0
        self._results: queue.Queue[tuple[int, int, str, bytes]] = queue.Queue()
        self._photos: dict[tuple[int, str], Any] = {}
        self._cards: list[Any] = []
        self._snapshots: list[Any] = []
        self._icons: list[Any] = []
        self._after: str | None = None
        self._hidden = False
        from tkinter import font as tkfont

        scale = max(1, round(parent.winfo_fpixels("1i") / 96))
        self._snap_w, self._snap_h, self._icon_px = (
            SNAPSHOT_W * scale,
            SNAPSHOT_H * scale,
            ICON_PX * scale,
        )
        measure = tkfont.nametofont("TkDefaultFont").measure
        self._fits = lambda text: measure(text) <= self._snap_w - self._icon_px - 8 * scale
        colors = tb.Style().colors
        self._plain, self._accent = colors.bg, colors.success

        top = self.top = tb.Toplevel(title="Pick the game window")
        top.transient(parent.winfo_toplevel())
        top.protocol("WM_DELETE_WINDOW", lambda: self._finish(None))
        top.bind("<Escape>", lambda _e: self._finish(None))
        top.bind("<Return>", lambda _e: self._use())
        tb.Label(top, text="Pick your World of Warcraft client").pack(
            anchor="w", padx=12, pady=(12, 6)
        )

        buttons = tb.Frame(top)
        buttons.pack(side="bottom", fill="x", padx=12, pady=12)
        self._use_button = tb.Button(
            buttons, text="Use this window", command=self._use, bootstyle="success"
        )
        self._use_button.pack(side="right")
        tb.Button(
            buttons, text="Cancel", command=lambda: self._finish(None), bootstyle="secondary"
        ).pack(side="right", padx=6)
        tb.Button(buttons, text="Refresh", command=self._refresh, bootstyle="secondary").pack(
            side="left"
        )

        body = tb.Frame(top)
        body.pack(fill="both", expand=True, padx=12)
        width = COLUMNS * (self._snap_w + 28)
        height = 2 * (self._snap_h + 60 * scale)
        self._canvas = tk.Canvas(
            body,
            width=width,
            height=height,
            highlightthickness=0,
            bg=self._plain,
        )
        bar = tb.Scrollbar(body, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)
        self._grid = tb.Frame(self._canvas)
        self._canvas.create_window((0, 0), window=self._grid, anchor="nw")
        self._grid.bind(
            "<Configure>",
            lambda _e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        for sequence, step in (("<MouseWheel>", 0), ("<Button-4>", -1), ("<Button-5>", 1)):
            top.bind(sequence, lambda e, step=step: self._scroll(e, step))
        self._blank = tk.PhotoImage(width=self._snap_w, height=self._snap_h)
        self._no_icon = tk.PhotoImage(width=self._icon_px, height=self._icon_px)

        self._show(found)
        self._after = top.after(POLL_MS, self._poll)

    # -- the cards --------------------------------------------------------------------------

    def _show(self, found: list[WindowChoice]) -> None:
        """Lay out a card per window and start making its snapshot and icon."""
        tb = self._tb
        self._generation += 1
        self._found = found
        self._selected = 0
        self._photos.clear()
        for child in self._grid.winfo_children():
            child.destroy()
        self._cards, self._snapshots, self._icons = [], [], []
        if not found:
            tb.Label(self._grid, text="No windows found. Start the game, then press Refresh.").grid(
                padx=8, pady=24
            )
        for i, choice in enumerate(found):
            w = choice.window
            card = self._tk.Frame(
                self._grid, bg=self._plain, highlightthickness=3, highlightbackground=self._plain
            )
            card.grid(row=i // COLUMNS, column=i % COLUMNS, padx=4, pady=4, sticky="n")
            snapshot = tb.Label(
                card, image=self._blank, text="no preview", compound="center", anchor="center"
            )
            snapshot.pack(padx=4, pady=(4, 2))
            title = tb.Label(
                card, image=self._no_icon, text=" " + _elide(w.title, self._fits), compound="left"
            )
            title.pack(anchor="w", padx=4)
            about = " · ".join(x for x in (choice.app, f"{w.width}x{w.height}") if x)
            detail = tb.Label(card, text=about, bootstyle="secondary")
            detail.pack(anchor="w", padx=4, pady=(0, 2 if choice.guess else 4))
            parts = [card, snapshot, title, detail]
            for part in parts:
                part.bind("<Button-1>", lambda _e, i=i: self._select(i))
                part.bind("<Double-Button-1>", lambda _e, i=i: self._use(i))
            self._cards.append(card)
            self._snapshots.append(snapshot)
            self._icons.append(title)
        self._use_button.configure(state="normal" if found else "disabled")
        self._select(0)
        # grim copies the screen, so on Hyprland the picker stays out of the way until the
        # snapshots are taken; everywhere else a window is grabbed by itself
        self._hidden = any(c.window.compositor for c in found)
        if self._hidden:
            self.top.grab_release()
            self.top.withdraw()
        else:
            self._reveal()
        threading.Thread(
            target=self._snapshot_worker,
            args=(self._generation, list(found)),
            name="wowfish-snapshots",
            daemon=True,
        ).start()

    def _reveal(self) -> None:
        self._hidden = False
        self.top.deiconify()
        self.top.lift()
        self.top.focus_set()
        try:
            self.top.grab_set()
        except Exception:  # not viewable yet: the picker still works, just not modal
            self.top.after(100, self._grab_later)

    def _grab_later(self) -> None:
        with contextlib.suppress(Exception):
            self.top.grab_set()

    def _select(self, index: int) -> None:
        if not self._cards:
            return
        self._selected = max(0, min(index, len(self._cards) - 1))
        for i, card in enumerate(self._cards):
            card.configure(highlightbackground=self._accent if i == self._selected else self._plain)

    def _scroll(self, event: Any, step: int) -> None:
        delta = step or (-1 if getattr(event, "delta", 0) > 0 else 1)
        self._canvas.yview_scroll(delta, "units")

    # -- snapshots and icons ----------------------------------------------------------------

    def _snapshot_worker(self, generation: int, found: list[WindowChoice]) -> None:
        for kind, make in (("snapshot", window_snapshot), ("icon", window_icon)):
            size = self._snap_w if kind == "snapshot" else self._icon_px
            for i, choice in enumerate(found):
                frame = make(choice.window, size)
                if frame is not None:
                    with contextlib.suppress(Exception):
                        self._results.put((generation, i, kind, image.bgr_to_png_bytes(frame)))
            if kind == "snapshot":
                self._results.put((generation, -1, "shown", b""))

    def _poll(self) -> None:
        while True:
            try:
                generation, i, kind, png = self._results.get_nowait()
            except queue.Empty:
                break
            if generation != self._generation:
                continue  # from before a Refresh
            if kind == "shown":
                if self._hidden:
                    self._reveal()
                continue
            photo = self._photos[(i, kind)] = image.photo_from_png(png)
            if kind == "snapshot":
                self._snapshots[i].configure(image=photo, text="")
            else:
                self._icons[i].configure(image=photo)
        self._after = self.top.after(POLL_MS, self._poll)

    # -- buttons ----------------------------------------------------------------------------

    def _refresh(self) -> None:
        try:
            found = self._choices()
        except (NotSupported, OSError):
            return  # keep what is shown
        self._show(found)

    def _use(self, index: int | None = None) -> None:
        if index is not None:
            self._select(index)
        if self._found:
            self._finish(self._found[self._selected].window)

    def _finish(self, picked: WindowRef | None) -> None:
        """Close, free every Tk image on this (the Tk) thread, then hand the answer over."""
        if self._after is not None:
            self.top.after_cancel(self._after)
            self._after = None
        self._generation += 1  # whatever the worker still sends is dropped
        self.top.grab_release()
        self.top.destroy()
        self._photos.clear()
        self._cards, self._snapshots, self._icons = [], [], []
        self._blank = self._no_icon = None
        on_done, self._on_done = self._on_done, lambda _picked: None
        gc.collect()
        on_done(picked)


def pick_window(
    parent: Any,
    on_done: Callable[[WindowRef | None], None],
    *,
    choices: Callable[[], list[WindowChoice]] = window_choices,
) -> None:
    """Open the modal picker over `parent`'s window and return at once.

    `on_done` runs once on the Tk thread after the picker closes: with the picked window, or
    None when the user cancelled or the windows could not be listed.
    """
    try:
        found = choices()
    except (NotSupported, OSError) as e:
        from ttkbootstrap.dialogs import Messagebox

        Messagebox.show_error(str(e), title="Cannot list the windows", parent=parent)
        on_done(None)
        return
    _Picker(parent, on_done, choices, found)
