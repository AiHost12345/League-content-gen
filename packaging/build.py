"""Build the one-file desktop app with PyInstaller.

    pip install . pyinstaller
    python packaging/build.py

Output: dist/LeagueBuildOptimizer(.exe)
"""

import sys
from pathlib import Path

import PyInstaller.__main__

here = Path(__file__).parent
args = [
    str(here / "launcher.py"),
    "--name", "LeagueBuildOptimizer",
    "--onefile",
    "--windowed",
    "--noconfirm",
    "--collect-data", "buildopt",  # Data Dragon snapshot + Riot root certificate
    "--hidden-import", "websocket",
    "--hidden-import", "buildopt.gui.app",
]
icon = here / ("icon.ico" if sys.platform == "win32" else "icon.icns")
if icon.exists():
    args += ["--icon", str(icon)]
PyInstaller.__main__.run(args)
