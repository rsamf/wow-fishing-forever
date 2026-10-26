# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the two wowfish single-file executables: `wowfish` (console) and
`wowfish-gui` (windowed, opens the GUI).

Build with: uv run pyinstaller --clean --noconfirm packaging/wowfish.spec
"""

from PyInstaller.utils.hooks import collect_data_files, copy_metadata

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
    name="wowfish",
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
    name="wowfish-gui",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)
