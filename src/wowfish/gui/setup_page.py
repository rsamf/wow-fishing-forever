"""Setup tab: prerequisites, keys, time limit, audio settings."""

from __future__ import annotations

from typing import Any

from wowfish.capture import linux_hyprland
from wowfish.config import DEFAULT_WINDOW_TITLE
from wowfish.gui.presenter import Presenter

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

HELP_WRAP_PX = 420


def background_help(hyprland: bool) -> str:
    """What the "?" beside the background-fishing checkbox says; Hyprland gets its catch."""
    text = (
        "With this on you can use other windows while the bot fishes. When it hears the splash "
        "it brings the game to the front, presses Interact, waits for the loot, casts again, "
        "and then gives focus back to the window you were in. That takes a few seconds per "
        "fish; anything you type in that time goes to the game.\n\n"
        "Turn on Sound in Background in the game's sound options, or the bot cannot hear the "
        "splash. Keep the game window open (not minimized).\n\n"
        "With this off, the bot pauses whenever the game is not the focused window."
    )
    if hyprland:
        text += (
            "\n\nOn Hyprland, a game running as a native Wayland window must also stay "
            "uncovered on a visible workspace: the bot can only read the screen there. A game "
            "under Xwayland can be covered."
        )
    return text


class SetupPage:
    def __init__(self, parent: Any, presenter: Presenter) -> None:
        import ttkbootstrap as tb
        from ttkbootstrap.widgets import ToolTip

        self.p = presenter
        self.frame = tb.Frame(parent, padding=10)
        f = self.frame
        tb.Label(f, text=CHECKLIST, justify="left").pack(anchor="w", pady=(0, 10))

        window_row = tb.Labelframe(f, text="Game window", padding=8)
        window_row.pack(fill="x", pady=4)
        tb.Label(window_row, text="Window title").pack(side="left")
        self.window_title = tb.Entry(window_row, width=28)
        self.window_title.insert(0, presenter.cfg.game.window_title)
        self.window_title.pack(side="left", padx=6)
        self.window_title.bind("<Return>", lambda _e: self._save_title())
        tb.Button(window_row, text="Save", command=self._save_title, bootstyle="secondary").pack(
            side="left"
        )
        self.title_note = tb.Label(
            window_row,
            text=f'Preselects the game window (empty = "{DEFAULT_WINDOW_TITLE}").',
        )
        self.title_note.pack(side="left", padx=6)

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

        background = tb.Labelframe(f, text="Fishing in the background", padding=8)
        background.pack(fill="x", pady=4)
        self.background = tb.BooleanVar(value=presenter.cfg.run.background)
        tb.Checkbutton(
            background,
            text="Fish while I use another window (the bot takes focus for a few seconds per fish)",
            variable=self.background,
            command=self._save_background,
        ).pack(side="left")
        help_icon = tb.Label(background, text="?", bootstyle="inverse-info", padding=(6, 0))
        help_icon.pack(side="left", padx=8)
        wrap = round(HELP_WRAP_PX * max(1.0, parent.winfo_fpixels("1i") / 96))  # wider on HiDPI
        ToolTip(help_icon, text=background_help(linux_hyprland()), wraplength=wrap)
        self.background_note = tb.Label(background, text="")
        self.background_note.pack(side="left")

        audio = tb.Labelframe(f, text="Audio (what the bot hears)", padding=8)
        audio.pack(fill="x", pady=4)
        self.devices = presenter.audio_devices()
        if presenter.audio_follows_window:
            self.device = None
            tb.Label(audio, text="The bot hears only the app of the game window you pick.").grid(
                row=0, column=0, columnspan=4, sticky="w"
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
                audio, text="Use this device", command=self._save_device, bootstyle="secondary"
            ).grid(row=0, column=3, padx=6)
            self.device_note = tb.Label(audio, text="The input device the game's sound goes to.")
            self.device_note.grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))
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

    def _save_title(self) -> None:
        raw = self.window_title.get()
        saved = self.p.save_window_title(raw)
        self.window_title.delete(0, "end")
        self.window_title.insert(0, saved)
        if not raw.strip():
            self.title_note.configure(text=f"saved default ({saved})")
        else:
            self.title_note.configure(text=f"saved ({saved})")

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

    def _save_device(self) -> None:
        i = self.device.current()
        if 0 <= i < len(self.devices):
            name = self.devices[i].id
            self.p.save_audio_device(name)
            self.device_note.configure(text=f"saved {name}; used from the next Start")

    def _save_background(self) -> None:
        on = bool(self.background.get())
        self.p.save_background(on)
        self.background_note.configure(
            text=f"saved {'on' if on else 'off'}; used from the next Start"
        )

    def _save_bite(self) -> None:
        try:
            saved = self.p.save_bite_threshold(self.bite.get())
        except ValueError as e:
            self.bite_note.configure(text=str(e))
            return
        self.bite_note.configure(text=f"saved {saved:.2f}; used from the next Start")
