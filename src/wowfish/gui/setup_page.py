"""Setup tab: prerequisites, keys, audio with a live meter."""

from __future__ import annotations

from typing import Any

from wowfish.gui.presenter import (
    Presenter,
    View,
    WorkerFailed,
)
from wowfish.gui.window_picker import pick_window

CHECKLIST = (
    "In the game, before starting:\n"
    "  1. Stand at the water with Fishing on an action bar key (the cast key below).\n"
    "  2. Bind Interact With Target (Options > Keybindings > Targeting) to the interact key.\n"
    "  3. Enable Auto Loot (Options > Controls).\n"
    "  4. Type /console SoftTargetIconGameObject 1 once (shows the gold gear over the bobber).\n"
    "  5. Set Ambience and Music to 0 so the splash is easy to hear.\n"
    "  6. Optional: put a macro with /quit on an action bar key and enter it as the quit key;\n"
    "     when the time limit below runs out, the bot presses it to close the game.\n"
    "Then press Start on the Fish tab. Nothing below needs setting up first."
)
LURE_HINT = (
    "Optional. A macro like /use Shiny Bauble + /use 16 that puts a lure on the pole; it is "
    "pressed before the first cast and again whenever the lure runs out. Empty = no lure."
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
        tb.Label(keys_row, text="Quit key").grid(row=0, column=4, sticky="w")
        self.quit = tb.Entry(keys_row, width=8)
        self.quit.insert(0, presenter.cfg.game.quit_key)
        self.quit.grid(row=0, column=5, padx=6)
        tb.Button(keys_row, text="Save keys", command=self._save_keys, bootstyle="secondary").grid(
            row=0, column=6, padx=6
        )
        self.keys_note = tb.Label(keys_row, text="")
        self.keys_note.grid(row=0, column=7, sticky="w")
        tb.Label(keys_row, text="Lure key").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.lure = tb.Entry(keys_row, width=8)
        self.lure.insert(0, presenter.cfg.game.lure_key)
        self.lure.grid(row=1, column=1, padx=6, pady=(6, 0))
        tb.Label(keys_row, text=LURE_HINT, wraplength=420, justify="left").grid(
            row=1, column=2, columnspan=6, sticky="w", pady=(6, 0)
        )

        limit = tb.Labelframe(f, text="Time limit", padding=8)
        limit.pack(fill="x", pady=4)
        tb.Label(limit, text="Stop after").pack(side="left")
        self.minutes = tb.Spinbox(limit, from_=0, to=1440, increment=15, width=5)
        self.minutes.set(presenter.cfg.game.session_minutes)
        self.minutes.pack(side="left", padx=6)
        self.minutes.bind("<Return>", lambda _e: self._save_limit())
        tb.Label(limit, text="min (0 = never)").pack(side="left")
        tb.Button(limit, text="Save", command=self._save_limit, bootstyle="secondary").pack(
            side="left", padx=6
        )
        self.limit_note = tb.Label(limit, text="")
        self.limit_note.pack(side="left")

        audio = tb.Labelframe(f, text="Audio (what the bot hears)", padding=8)
        audio.pack(fill="x", pady=4)
        self.devices = presenter.audio_devices()
        self._heard: Any = None  # the window the meter is on (Linux, Windows)
        self._listen_label = "Listen" if presenter.audio_follows_window else "Use this and listen"
        if presenter.audio_follows_window:
            self.device = None
            tb.Button(audio, text="Listen", command=self._listen, bootstyle="secondary").grid(
                row=0, column=0, sticky="w"
            )
            tb.Label(audio, text="The bot hears only the app of the game window.").grid(
                row=0, column=1, columnspan=3, sticky="w", padx=6
            )
        else:  # macOS: an input device carries the game's sound
            names = [d.name for d in self.devices]
            self.device = tb.Combobox(audio, values=names, state="readonly", width=48)
            current = presenter.cfg.audio.device
            pick = next(
                (i for i, d in enumerate(self.devices) if current and current in (d.id, d.name)),
                None,
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

    # -- actions ----------------------------------------------------------------------------

    def _save_keys(self) -> None:
        try:
            self.p.save_keys(self.cast.get(), self.interact.get(), self.quit.get(), self.lure.get())
            self.keys_note.configure(text="saved")
        except ValueError as e:
            self.keys_note.configure(text=str(e))

    def _save_limit(self) -> None:
        try:
            self.p.save_session_minutes(int(self.minutes.get()))
        except ValueError:
            self.limit_note.configure(text="enter whole minutes, 0 for never")
            return
        quit_key = self.p.cfg.game.quit_key
        then = f"; then the bot presses {quit_key} to quit the game" if quit_key else ""
        self.limit_note.configure(text=f"saved; used from the next Start{then}")

    def _selected_device(self) -> str | None:
        if self.device is None:
            return None
        i = self.device.current()
        return self.devices[i].id if 0 <= i < len(self.devices) else None

    def _save_device(self) -> None:
        sel = self._selected_device()
        if sel is not None:
            self.p.save_audio_device(sel)
            self._meter_error = ""
            self.p.start_meter(device=sel)

    def _blocked(self) -> str:
        """Why no window can be picked right now (platform, macOS permission); "" when one can."""
        return "; ".join(self.p.preflight(request=True))

    def _listen(self) -> None:
        self._meter_error = self._blocked()
        if not self._meter_error:
            pick_window(self.frame, self._listen_to)

    def _listen_to(self, window: Any) -> None:
        if window is None:
            self._meter_error = "No window picked"
            return
        self._heard = window
        self._meter_error = ""
        self.p.start_meter(window=window)

    def _save_bite(self) -> None:
        try:
            saved = self.p.save_bite_threshold(self.bite.get())
        except ValueError as e:
            self.bite_note.configure(text=str(e))
            return
        self.bite_note.configure(text=f"saved {saved:.2f}; used from the next Start")

    # -- rendering --------------------------------------------------------------------------

    def render(self, view: View) -> None:
        pct = max(0.0, min(100.0, (view.level_dbfs + 60.0) / 60.0 * 100.0))
        self.meter.configure(value=pct)
        if not view.listening:
            self.level_text.configure(
                text=self._meter_error or f"not listening (press {self._listen_label})"
            )
        else:
            level = "silent" if view.level_dbfs <= -60 else f"{view.level_dbfs:5.1f} dBFS"
            best = max(view.last_scores.values(), default=None)
            score = f"   bite score {best:.2f}" if best is not None and best > -1.0 else ""
            heard = f"listening to {self._heard.title}: " if self._heard is not None else ""
            self.level_text.configure(text=heard + level + score)
        for item, line in zip(view.log_events, view.new_lines, strict=True):
            match item:
                case WorkerFailed():
                    self._meter_error = line
