"""The Fish tab's Scores graph: each splash clip's score over the current cast, on a Tk canvas."""

from __future__ import annotations

from typing import Any

from wowfish.assets import CLIP_NAMES
from wowfish.gui.presenter import ScoreTrace

# One colour per clip, fixed in CLIP_NAMES order. Validated as a set on the bootstrap-dark
# surface (all checks pass; worst adjacent colour-blind separation dE 8.4, inside the band that
# needs names on screen too), so the legend and the hover readout always name the clips.
CLIP_COLORS = dict(zip(CLIP_NAMES, ("#3987e5", "#d95926", "#199e70", "#c98500"), strict=True))
SURFACE = "#212529"  # bootstrap-dark background
GRID = "#45484b"
MUTED = "#bdc4ca"
TEXT = "#f8f9fa"
FONT = ("TkDefaultFont", 10)
GAP = 6  # px between a label and what it labels
SWATCH = 18  # legend / readout colour sample length in px
SILENT = -1.0  # the detector's score for a slice too quiet to match; drawn at 0, like the log


class ScoreGraph:
    def __init__(self, parent: Any) -> None:
        import tkinter as tk
        import tkinter.font as tkfont

        self.canvas = tk.Canvas(parent, background=SURFACE, highlightthickness=0, height=280)
        self._font = tkfont.Font(font=FONT)  # sizes come from the font: HiDPI scales it
        self._lh = self._font.metrics("linespace")
        self._trace: ScoreTrace | None = None
        self._span = 30.0
        self._key: tuple[int, int, float] | None = None
        self._box: tuple[float, float, float, float] | None = None  # plot x0, x1, y0, y1
        self.canvas.bind("<Configure>", lambda _e: self._redraw())
        self.canvas.bind("<Motion>", self._hover)
        self.canvas.bind("<Leave>", lambda _e: self.canvas.delete("hover"))

    def show(self, trace: ScoreTrace, span_s: float) -> None:
        """Draw `trace` over 0..span_s seconds since the bobber was found; no-op if unchanged."""
        key = (id(trace), trace.version, span_s)
        if key != self._key:
            self._trace, self._span, self._key = trace, max(span_s, 1.0), key
            self._redraw()

    # -- geometry ----------------------------------------------------------------------------

    def _x(self, t: float) -> float:
        assert self._box is not None
        x0, x1, _, _ = self._box
        return x0 + (x1 - x0) * min(max(t / self._span, 0.0), 1.0)

    def _y(self, score: float) -> float:
        assert self._box is not None
        _, _, y0, y1 = self._box
        return y1 - (y1 - y0) * min(max(score, 0.0), 1.0)

    def _ticks(self) -> list[float]:
        step = 5.0 if self._span > 10 else 1.0
        return [i * step for i in range(int(self._span / step + 1e-9) + 1)]

    # -- drawing -----------------------------------------------------------------------------

    def _redraw(self) -> None:
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        legend_h = self._legend(w)
        left = self._font.measure("0.0") + 2 * GAP
        right = self._font.measure(f"{self._ticks()[-1]:g} s") / 2 + GAP
        top, bottom = legend_h + self._lh, h - self._lh - 2 * GAP
        if w - right - left < 120 or bottom - top < 60:
            self._box = None
            return
        self._box = (left, w - right, top, bottom)
        self._axes()
        tr = self._trace
        if tr is None or not tr.points:
            x0, x1, y0, y1 = self._box
            c.create_text(
                (x0 + x1) / 2, (y0 + y1) / 2, fill=MUTED, font=FONT, width=x1 - x0,
                text="The clip scores show here once the bobber is found",
            )  # fmt: skip
        if tr is not None:
            self._outcome(tr)  # under the lines, so the bite's own line stays visible
            for name in CLIP_NAMES:
                self._series(tr, name)
            self._threshold(tr)

    def _legend(self, width: int) -> int:
        """Swatch + name per clip, wrapped onto more rows when they don't fit; returns height."""
        c = self.canvas
        x, y = float(GAP), GAP + self._lh / 2
        for name in CLIP_NAMES:
            item_w = SWATCH + GAP + self._font.measure(name)
            if x > GAP and x + item_w > width - GAP:
                x, y = float(GAP), y + self._lh
            c.create_line(x, y, x + SWATCH, y, fill=CLIP_COLORS[name], width=3)
            c.create_text(x + SWATCH + GAP, y, text=name, anchor="w", fill=TEXT, font=FONT)
            x += item_w + 3 * GAP
        return int(y + self._lh / 2)

    def _axes(self) -> None:
        c = self.canvas
        assert self._box is not None
        x0, x1, _, y1 = self._box
        for i in range(6):
            s = i / 5
            c.create_line(x0, self._y(s), x1, self._y(s), fill=GRID)
            c.create_text(x0 - GAP, self._y(s), text=f"{s:.1f}", anchor="e", fill=MUTED, font=FONT)
        for t in self._ticks():
            c.create_line(self._x(t), y1, self._x(t), y1 + 4, fill=GRID)
            c.create_text(self._x(t), y1 + GAP, text=f"{t:g} s", anchor="n", fill=MUTED, font=FONT)

    def _series(self, tr: ScoreTrace, name: str) -> None:
        """One clip's line; silent slices sit at 0 (they can't reach the threshold either)."""
        coords: list[float] = []
        for t, peaks in tr.points:
            coords += [self._x(t - tr.t0), self._y(max(peaks.get(name, SILENT), 0.0))]
        color = CLIP_COLORS[name]
        if len(coords) >= 4:
            self.canvas.create_line(*coords, fill=color, width=2)
        elif coords:  # the first slice of a cast
            x, y = coords
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color, outline=color)

    def _threshold(self, tr: ScoreTrace) -> None:
        if tr.threshold is None:
            return
        assert self._box is not None
        x0, x1, _, _ = self._box
        y = self._y(tr.threshold)
        self.canvas.create_line(x0, y, x1, y, fill=MUTED, dash=(6, 4))
        self.canvas.create_text(
            x1, y - 3, text=f"threshold {tr.threshold:.2f}", anchor="se", fill=MUTED, font=FONT
        )

    def _outcome(self, tr: ScoreTrace) -> None:
        assert self._box is not None
        x0, x1, y0, y1 = self._box
        if tr.bite is not None:
            t, clip, score = tr.bite
            x = self._x(t - tr.t0)
            self.canvas.create_line(x, y0, x, y1, fill=MUTED, dash=(2, 3))
            right = x > (x0 + x1) / 2  # keep the label inside the plot
            self.canvas.create_text(
                x - GAP if right else x + GAP, y0, anchor="ne" if right else "nw",
                text=f"bite {score:.2f} {clip}", fill=TEXT, font=FONT,
            )  # fmt: skip
        elif tr.timed_out:
            self.canvas.create_text(
                x1, y0, anchor="ne", fill=MUTED, font=FONT, text=f"no splash in {self._span:g} s"
            )

    # -- hover -------------------------------------------------------------------------------

    def _hover(self, e: Any) -> None:
        """Crosshair on the nearest slice, and a readout naming every clip's score there."""
        c = self.canvas
        c.delete("hover")
        tr = self._trace
        if tr is None or not tr.points or self._box is None:
            return
        x0, x1, y0, y1 = self._box
        if not (x0 <= e.x <= x1 and y0 <= e.y <= y1):
            return
        t = (e.x - x0) / (x1 - x0) * self._span
        pt, peaks = min(tr.points, key=lambda p: abs(p[0] - tr.t0 - t))
        x = self._x(pt - tr.t0)
        c.create_line(x, y0, x, y1, fill=MUTED, dash=(2, 2), tags="hover")
        rows = sorted(peaks.items(), key=lambda kv: -kv[1])
        lines = [(None, f"{pt - tr.t0:.1f} s")] + [
            (name, f"{name}  {max(s, 0.0):.2f}{' (silent)' if s <= SILENT else ''}")
            for name, s in rows
        ]
        box_w = SWATCH + 3 * GAP + max(self._font.measure(text) for _, text in lines)
        box_h = 2 * GAP + len(lines) * self._lh
        bx = x + 2 * GAP if x + 2 * GAP + box_w <= x1 else x - 2 * GAP - box_w
        bx = max(float(x0), bx)
        c.create_rectangle(bx, y0, bx + box_w, y0 + box_h, fill=SURFACE, outline=GRID, tags="hover")
        for i, (name, text) in enumerate(lines):
            ty = y0 + GAP + (i + 0.5) * self._lh
            if name is not None:
                sx, color = bx + GAP, CLIP_COLORS[name]
                c.create_line(sx, ty, sx + SWATCH, ty, fill=color, width=3, tags="hover")
            c.create_text(
                bx + 2 * GAP + SWATCH, ty, text=text, anchor="w", fill=TEXT, font=FONT, tags="hover"
            )
