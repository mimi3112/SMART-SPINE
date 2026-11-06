"""
10_dashboard.py — compatibility shim.

This file is superseded by app.py at the project root.
Running this script will automatically delegate to the unified app.

Usage (preferred):
    streamlit run app.py        (from project root)
    python desktop_app.py       (desktop application)

This shim exists so that old references to '10_dashboard.py' still work seamlessly.
"""

import runpy
import sys
from pathlib import Path

# Walk upward to find app.py at the project root
_here = Path(__file__).resolve().parent
_candidate = _here
while True:
    _app = _candidate / "app.py"
    if _app.exists():
        break
    _parent = _candidate.parent
    if _parent == _candidate:
        raise FileNotFoundError(
            "Could not find app.py while walking upward from " + str(_here)
        )
    _candidate = _parent

# Add the project root to sys.path so dashboard/* imports resolve
if str(_candidate) not in sys.path:
    sys.path.insert(0, str(_candidate))

runpy.run_path(str(_app), run_name="__main__")
