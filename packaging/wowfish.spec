# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the two single-file executables (console and windowed).

Build with: uv run pyinstaller --clean --noconfirm packaging/wowfish.spec
"""

import os
import secrets
import string

from PyInstaller.utils.hooks import collect_data_files, copy_metadata


def _rand_name(length: int = 10) -> str:
    alphabet = string.ascii_letters + string.digits
    return secrets.choice(string.ascii_letters) + "".join(
        secrets.choice(alphabet) for _ in range(length - 1)
    )


CLI_NAME = os.environ.get("BUILD_CLI_NAME") or _rand_name(10)
GUI_NAME = os.environ.get("BUILD_GUI_NAME") or _rand_name(10)

datas, binaries, hiddenimports = [], [], []
# importlib.metadata.version("wowfish") needs the dist-info inside the bundle.
datas += copy_metadata("wowfish")
datas += collect_data_files("soundcard")  # it reads bundled *.py.h headers at import
datas += collect_data_files("ttkbootstrap")  # themes and icons
datas += [("../src/wowfish/assets", "wowfish/assets")]
# Pillow's C extension _imagingtk imports this at load time (ttkbootstrap uses PIL.ImageTk);
# PyInstaller can't see imports made from C.
hiddenimports += ["PIL._tkinter_finder"]
EXCLUDES = ["matplotlib", "torch", "torchvision"]

a = Analysis(
    ["../src/wowfish/__main__.py"],
    pathex=["../src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=CLI_NAME,
    debug=False,
    strip=False,
    upx=False,
    console=True,
)

g = Analysis(
    ["gui_entry.py"],
    pathex=["../src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)
gui_pyz = PYZ(g.pure)
gui_exe = EXE(
    gui_pyz,
    g.scripts,
    g.binaries,
    g.datas,
    [],
    name=GUI_NAME,
    debug=False,
    strip=False,
    upx=False,
    console=False,
)
