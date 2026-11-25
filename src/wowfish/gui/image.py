"""BGR frames -> PNG or PPM bytes -> Tk images, without Pillow."""

from __future__ import annotations

import base64

import cv2
import numpy as np

from wowfish.capture import Box


def fit_box(fw: int, fh: int, cw: int, ch: int) -> Box:
    """Where a fw x fh picture goes in a cw x ch space: as large as fits, same shape, centred.
    Returns x, y, w, h."""
    scale = min(cw / fw, ch / fh)
    w, h = max(1, round(fw * scale)), max(1, round(fh * scale))
    return (cw - w) // 2, (ch - h) // 2, w, h


def bgr_to_png_bytes(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        raise ValueError("cannot encode frame as PNG")
    return buf.tobytes()


def bgr_to_ppm_bytes(frame: np.ndarray) -> bytes:
    """Binary PPM, which Tk reads without decompressing: for pictures redrawn many times a
    second, where PNG costs ten times as long."""
    h, w = frame.shape[:2]
    return b"P6 %d %d 255\n" % (w, h) + cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes()


def png_base64(png: bytes) -> str:
    return base64.b64encode(png).decode("ascii")


def photo_scaled(frame: np.ndarray, w: int, h: int):  # -> tkinter.PhotoImage
    """`frame` as a Tk image of about w x h. Shrinking is smooth; growing past the frame's own
    size repeats whole pixels (Tk's zoom), which keeps a maximized window's redraw quick. The
    image may be a few pixels smaller than asked: ask it for its size."""
    import tkinter

    k = -(-w // frame.shape[1])  # 1 unless the picture is drawn larger than it came
    size = max(1, w // k), max(1, h // k)
    if size != frame.shape[1::-1]:
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    small = tkinter.PhotoImage(data=bgr_to_ppm_bytes(frame))
    return small if k == 1 else small.zoom(k)


def photo_from_png(png: bytes):  # -> tkinter.PhotoImage; imported lazily so tests stay Tk-free
    import tkinter

    return tkinter.PhotoImage(data=png_base64(png))
