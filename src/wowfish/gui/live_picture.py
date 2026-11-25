"""The Fish tab's live picture of the game with the search zone drawn on it, on a Tk canvas.
The picture is scaled to the canvas, which grows and shrinks with the window.
While the bot runs, drag a corner of the zone to resize it, or drag inside it to move it."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from wowfish.capture import Box
from wowfish.engine.events import Preview
from wowfish.gui import image, zone
from wowfish.gui.score_graph import MUTED, SURFACE
from wowfish.vision.region import Region

WIDTH, HEIGHT = 640, 360  # the size asked for; the layout gives more as the window grows
ZONE_COLOR = "#ffc107"  # bootstrap warning yellow: shows on water, grass and the dark UI
ZONE_STOPPED = "#9aa0a6"  # grey: the bot is stopped and the zone cannot be dragged
ZONE_NAME = "Icon detection zone"
GEAR_COLOR = "#00c800"  # the gear the bot is following
MATCH_COLOR = "#ff8c00"  # the best match so far, while it is still looking
SCAN_TRACK = "#6c757d"  # the scan bar's unfilled part
SCAN_BAR = 6  # px height of the scan bar along the zone's bottom edge
HANDLE = 12  # px side of a corner handle
GRAB = 14  # px around a corner that still takes hold of it
CURSORS = {"nw": "top_left_corner", "ne": "top_right_corner", "sw": "bottom_left_corner",
           "se": "bottom_right_corner", "move": "fleur"}  # fmt: skip


class LivePicture:
    def __init__(self, parent: Any, on_zone: Callable[[Region], None], start: Region) -> None:
        import tkinter as tk

        self.canvas = tk.Canvas(
            parent, background=SURFACE, highlightthickness=0, width=WIDTH, height=HEIGHT
        )
        self._on_zone = on_zone
        self.zone = start
        self._preview: Preview | None = None
        self._running = False  # the zone can be dragged only while the bot runs
        self._photo: Any = None
        self._fit: Box = (0, 0, WIDTH, HEIGHT)  # where the picture is drawn: x, y, w, h
        self._shown: tuple[int, int, int] | None = None  # id of the frame drawn, and its size
        self._held: tuple[str, float, float, Region] | None = None  # grip, press fx, fy, zone then
        self._named = False  # the pointer is over the zone, so its name shows inside it
        self._resize: str | None = None  # the pending redraw after a resize, if any
        self._image = self.canvas.create_image(0, 0, anchor="nw")
        self._note = self.canvas.create_text(
            WIDTH // 2, HEIGHT // 2, text="No picture yet. Press Start.", fill=MUTED
        )
        self.canvas.bind("<Configure>", self._resized)
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", lambda _e: setattr(self, "_held", None))
        self.canvas.bind("<Motion>", self._hover)
        self.canvas.bind("<Leave>", lambda _e: self._name(False))

    def show(self, preview: Preview | None, running: bool) -> None:
        """Put up the newest picture and what the bot marks on it; a no-op when nothing changed."""
        if preview is self._preview and running == self._running:
            return
        self._preview, self._running = preview, running
        self._render()

    def _resized(self, _e: Any) -> None:
        """A drag-resize sends a burst of these: redraw once, when the burst is over."""
        if self._resize is None:
            self._resize = self.canvas.after_idle(self._render)

    def _render(self) -> None:
        """Draw the picture as large as the canvas allows, and the marks over it."""
        self._resize = None
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if self._preview is None:
            self.canvas.coords(self._note, cw // 2, ch // 2)
            return
        frame = self._preview.frame
        fh, fw = frame.shape[:2]
        x, y, w, h = image.fit_box(fw, fh, max(cw, 1), max(ch, 1))
        if self._shown != (id(frame), w, h):  # a scan sends the same frame with new marks
            self._shown = (id(frame), w, h)
            self._photo = image.photo_scaled(frame, w, h)
            self.canvas.itemconfigure(self._image, image=self._photo)
            self.canvas.delete(self._note)
        # zoom rounds down, so the image says how large it really is
        self._fit = (x, y, self._photo.width(), self._photo.height())
        self.canvas.coords(self._image, x, y)
        self._draw_marks()

    # -- the marks: zone, scan bar, gear or best match ---------------------------------------

    def _draw_marks(self) -> None:
        self.canvas.delete("mark")
        if self._preview is None:
            return
        px, py, w, h = self._fit
        zx0, zy0, zx1, zy1 = self.zone.to_box(w, h)
        # kept a pixel inside the picture so a whole-window zone still shows its outline
        x0, y0 = px + max(1, zx0), py + max(1, zy0)
        x1, y1 = px + min(w - 2, zx1), py + min(h - 2, zy1)
        color = ZONE_COLOR if self._running else ZONE_STOPPED
        c = self.canvas
        c.create_rectangle(x0, y0, x1, y1, outline=color, width=2, tags="mark")
        for cx, sx in ((x0, 1), (x1, -1)):
            for cy, sy in ((y0, 1), (y1, -1)):
                c.create_rectangle(
                    cx, cy, cx + sx * HANDLE, cy + sy * HANDLE, fill=color, outline="", tags="mark"
                )
        scan = self._preview.scan
        if scan is not None:
            y = y1 - SCAN_BAR - 2
            xa, xb = x0 + HANDLE + 4, x1 - HANDLE - 4  # between the corner handles
            c.create_rectangle(xa, y, xb, y + SCAN_BAR, fill=SCAN_TRACK, outline="", tags="mark")
            c.create_rectangle(
                xa, y, xa + round((xb - xa) * scan), y + SCAN_BAR,
                fill=color, outline="", tags="mark",
            )  # fmt: skip
            c.create_text(
                xa + 2, y - 4, anchor="sw", fill=color, tags="mark",
                text=f"Scanning the zone for the gear: {scan:.0%}",
            )  # fmt: skip
        if self._named:
            c.create_text(
                (x0 + x1) // 2, y0 + 8, anchor="n", text=ZONE_NAME, fill=color, tags="mark",
                font=("TkDefaultFont", 11, "bold"),
            )  # fmt: skip
        if self._preview.box is not None:
            k = w / self._preview.frame.shape[1]  # frame pixels -> drawn pixels
            bx, by, bw, bh = (v * k for v in self._preview.box)
            score = self._preview.score
            color = GEAR_COLOR if score is None else MATCH_COLOR
            c.create_rectangle(
                px + bx - 2, py + by - 2, px + bx + bw + 2, py + by + bh + 2,
                outline=color, width=2, tags="mark",
            )  # fmt: skip
            if score is not None:
                c.create_text(
                    px + bx - 2, py + by - 6, anchor="sw", text=f"{score:.2f}", fill=color,
                    tags="mark",
                )  # fmt: skip

    def _name(self, show: bool) -> None:
        if show != self._named:
            self._named = show
            self._draw_marks()

    # -- dragging the zone -------------------------------------------------------------------

    def _at(self, e: Any) -> tuple[float, float]:
        """The pointer's place in the picture, as fractions of it."""
        x, y, w, h = self._fit
        return (e.x - x) / w, (e.y - y) / h

    def _grip(self, e: Any) -> str | None:
        if self._preview is None:
            return None
        _, _, w, h = self._fit
        return zone.grip(self.zone, *self._at(e), GRAB / w, GRAB / h)

    def _hover(self, e: Any) -> None:
        grip = self._grip(e)
        self._name(grip is not None)
        self.canvas.configure(cursor=CURSORS.get(grip or "", "") if self._running else "")

    def _press(self, e: Any) -> None:
        grip = self._grip(e) if self._running else None
        self._held = (grip, *self._at(e), self.zone) if grip else None

    def _drag(self, e: Any) -> None:
        if self._held is None or not self._running:
            return
        grip, fx0, fy0, start = self._held
        fx, fy = self._at(e)
        if grip == "move":
            self.zone = zone.pan(start, fx - fx0, fy - fy0)
        else:
            self.zone = zone.resize(self.zone, grip, fx, fy)
        self._draw_marks()
        self._on_zone(self.zone)
