"""Setup tab: prerequisites, keys, audio device with a live meter, calibration, bite check."""

from __future__ import annotations

from typing import Any

from wowfish.gui import image
from wowfish.gui.presenter import (
    CALIBRATE_WATCH_S,
    CalibrateDone,
    Presenter,
    VerifyDone,
    View,
    WorkerFailed,
)

CHECKLIST = (
    "In the game, before starting:\n"
    "  1. Stand at the water with Fishing on an action bar key (the cast key below).\n"
    "  2. Bind Interact With Target (Options > Keybindings > Targeting) to the interact key.\n"
    "  3. Enable Auto Loot (Options > Controls).\n"
    "  4. Type /console SoftTargetIconGameObject 1 once (shows the gold gear over the bobber).\n"
    "  5. Set Ambience and Music to 0 so the splash is easy to hear.\n"
    "Then press Start on the Fish tab. The first cast of every run measures the gear icon, so\n"
    "the bite check and the calibration below are optional: use them when fishing misbehaves."
)
VERIFY_HINT = "Click, then fish by hand in the game until a fish bites (it records for 60 s)."
CALIBRATE_HINT = (
    f"Click, then cast in the game: it watches for the gold gear for up to {CALIBRATE_WATCH_S:g} s."
)


class SetupPage:
    def __init__(self, parent: Any, presenter: Presenter) -> None:
        import ttkbootstrap as tb

        self.p = presenter
        self.frame = tb.Frame(parent, padding=10)
        f = self.frame
        tb.Label(f, text=CHECKLIST, justify="left").pack(anchor="w", pady=(0, 10))

        keys_row = tb.Labelframe(f, text="Keys", padding=8)
        keys_row.pack(fill="x", pady=4)
        tb.Label(keys_row, text="Cast key").grid(row=0, column=0, sticky="w")
        self.cast = tb.Entry(keys_row, width=8)
        self.cast.insert(0, presenter.cfg.game.cast_key)
        self.cast.grid(row=0, column=1, padx=6)
        tb.Label(keys_row, text="Interact key").grid(row=0, column=2, sticky="w")
        self.interact = tb.Entry(keys_row, width=8)
        self.interact.insert(0, presenter.cfg.game.interact_key)
        self.interact.grid(row=0, column=3, padx=6)
        tb.Button(keys_row, text="Save keys", command=self._save_keys, bootstyle="secondary").grid(
            row=0, column=4, padx=6
        )
        self.keys_note = tb.Label(keys_row, text="")
        self.keys_note.grid(row=0, column=5, sticky="w")

        audio = tb.Labelframe(f, text="Audio (what the bot hears) and bite check", padding=8)
        audio.pack(fill="x", pady=4)
        self.devices = presenter.audio_devices()
        names = [d.name for d in self.devices]
        self.device = tb.Combobox(audio, values=names, state="readonly", width=48)
        current = presenter.cfg.audio.device
        pick = next(
            (i for i, d in enumerate(self.devices) if current and current in (d.id, d.name)), None
        )
        if pick is None:
            default = presenter.default_device()
            pick = self.devices.index(default) if default in self.devices else None
        if pick is not None:
            self.device.current(pick)
        self.device.grid(row=0, column=0, columnspan=3, sticky="w")
        tb.Button(
            audio, text="Use this and listen", command=self._save_device, bootstyle="secondary"
        ).grid(row=0, column=3, padx=6)
        tb.Label(audio, text="Level").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.meter = tb.Progressbar(audio, maximum=100, value=0, bootstyle="info", length=300)
        self.meter.grid(row=1, column=1, columnspan=2, sticky="w", pady=(8, 0))
        self.level_text = tb.Label(audio, text="")
        self._meter_error = ""
        self.level_text.grid(row=1, column=3, sticky="w", padx=6, pady=(8, 0))
        self.verify_button = tb.Button(
            audio, text="Record 60 s and check the bite", command=self._verify, bootstyle="info"
        )
        self.verify_button.grid(row=2, column=0, sticky="w", pady=(8, 0))
        self._meter_was_on = False
        self.verify_note = tb.Label(audio, text=VERIFY_HINT, wraplength=560, justify="left")
        self.verify_note.grid(row=2, column=1, columnspan=3, sticky="w", padx=6, pady=(8, 0))
        bite_row = tb.Frame(audio)
        bite_row.grid(row=3, column=0, columnspan=4, sticky="w", pady=(8, 0))
        tb.Label(bite_row, text="Bite threshold").pack(side="left")
        self.bite = tb.Entry(bite_row, width=6)
        self.bite.insert(0, f"{presenter.cfg.audio.bite_threshold:.2f}")
        self.bite.pack(side="left", padx=6)
        self.bite.bind("<Return>", lambda _e: self._save_bite())
        tb.Button(bite_row, text="Save", command=self._save_bite, bootstyle="success").pack(
            side="left"
        )
        self.bite_note = tb.Label(bite_row, text="A splash must score at least this to count.")
        self.bite_note.pack(side="left", padx=6)

        cal = tb.Labelframe(f, text="Icon calibration (what the bot looks for)", padding=8)
        cal.pack(fill="both", expand=True, pady=4)
        self.cal_button = tb.Button(
            cal, text="Calibrate the gear icon", command=self._calibrate, bootstyle="info"
        )
        self.cal_button.grid(row=0, column=0, sticky="w")
        size = presenter.cfg.vision.icon_size_px
        current = f"Saved: {size} px. " if size > 0 else "Not calibrated yet. "
        self.cal_note = tb.Label(cal, text=current + CALIBRATE_HINT, wraplength=560, justify="left")
        self.cal_note.grid(row=0, column=1, sticky="w", padx=6)
        self.save_cal = tb.Button(
            cal,
            text="Save calibration",
            command=self._save_cal,
            bootstyle="success",
            state="disabled",
        )
        self.save_cal.grid(row=0, column=2, padx=6)
        self.cal_image = tb.Label(cal)
        self.cal_image.grid(row=1, column=0, columnspan=3, pady=(8, 0), sticky="w")
        self._photo = None

    # -- actions ----------------------------------------------------------------------------

    def _save_keys(self) -> None:
        try:
            self.p.save_keys(self.cast.get(), self.interact.get())
            self.keys_note.configure(text="saved")
        except ValueError as e:
            self.keys_note.configure(text=str(e))

    def _selected_device(self) -> str | None:
        i = self.device.current()
        return self.devices[i].id if 0 <= i < len(self.devices) else None

    def _save_device(self) -> None:
        sel = self._selected_device()
        if sel is not None:
            self.p.save_audio_device(sel)
            self._meter_error = ""
            self.p.start_meter(sel)

    def _verify(self) -> None:
        self._meter_was_on = self.p.meter_running
        self.p.stop_meter()  # the recording shows its own level on the meter
        self.verify_button.configure(state="disabled")
        self.verify_note.configure(text="Starting the recording...")
        self.p.record_and_verify(60.0, self._selected_device())

    def _calibrate(self) -> None:
        self.cal_button.configure(state="disabled")
        self.save_cal.configure(state="disabled")
        self.cal_note.configure(text="Starting...")
        self.p.grab_and_calibrate()

    def _save_bite(self) -> None:
        try:
            saved = self.p.save_bite_threshold(self.bite.get())
        except ValueError as e:
            self.bite_note.configure(text=str(e))
            return
        self.bite_note.configure(text=f"saved {saved:.2f}; used from the next Start")

    def _save_cal(self) -> None:
        self.p.save_calibration()
        self.cal_note.configure(text=self.cal_note.cget("text") + "  (saved)")

    # -- rendering --------------------------------------------------------------------------

    def render(self, view: View) -> None:
        pct = max(0.0, min(100.0, (view.level_dbfs + 60.0) / 60.0 * 100.0))
        self.meter.configure(value=pct)
        if not view.listening:
            self.level_text.configure(
                text=self._meter_error or "not listening (press Use this and listen)"
            )
        else:
            level = "silent" if view.level_dbfs <= -60 else f"{view.level_dbfs:5.1f} dBFS"
            best = max(view.last_scores.values(), default=None)
            score = f"   bite score {best:.2f}" if best is not None and best > -1.0 else ""
            self.level_text.configure(text=level + score)
        if view.verify_status:
            self.verify_note.configure(text=view.verify_status)
        if view.calibrate_status:
            self.cal_note.configure(text=view.calibrate_status)
        for item, line in zip(view.log_events, view.new_lines, strict=True):
            match item:
                case CalibrateDone(calibration=cal):
                    self._photo = image.photo_from_png(item.thumbnail_png)
                    self.cal_image.configure(image=self._photo)
                    if cal.valid:
                        self.cal_note.configure(
                            text=f"Found it: {cal.icon_size_px} px, score {cal.score:.2f}. "
                            "Save calibration to keep it."
                        )
                        self.save_cal.configure(state="normal")
                    else:
                        self.cal_note.configure(
                            text=f"No gold gear icon seen in {CALIBRATE_WATCH_S:g} s. "
                            "Is SoftTargetIconGameObject on "
                            "and the bobber close enough to interact with? Try again."
                        )
                    self.cal_button.configure(state="normal")
                case VerifyDone(report=report):
                    self.verify_note.configure(text=line)
                    if report.stands_out:  # suggest, but only Save changes the config
                        self.bite.delete(0, "end")
                        self.bite.insert(0, f"{report.threshold:.2f}")
                        self.bite_note.configure(text="suggested by the bite check; Save to use it")
                    self._verify_finished()
                case WorkerFailed(what="verify the bite"):
                    self.verify_note.configure(text=line)
                    self._verify_finished()
                case WorkerFailed(what="calibrate"):
                    self.cal_note.configure(text=line)
                    self.cal_button.configure(state="normal")
                case WorkerFailed():
                    self._meter_error = line

    def _verify_finished(self) -> None:
        self.verify_button.configure(state="normal")
        if self._meter_was_on:
            self._meter_was_on = False
            self.p.start_meter(self._selected_device())
