"""
desktop_app.py — SMART-SPINE native desktop launcher.

Starts Streamlit silently in the background, waits until the server is
ready, then opens a native OS window (via pywebview) showing the app.
Closing the window cleanly shuts down the Streamlit server.

Usage:
    python desktop_app.py
    run_desktop.bat          (Windows shortcut)
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# ── Port configuration ─────────────────────────────────────────────────────────
PORT = 8501
URL = f"http://localhost:{PORT}"
ROOT = Path(__file__).resolve().parent
APP = ROOT / "app.py"

# ── Load environment variables (.env) ──────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
except Exception:
    pass

# ── Splash / loading screen ───────────────────────────────────────────────────
SPLASH_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {
    margin: 0;
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
    font-family: 'Segoe UI', sans-serif;
    color: #e2e8f0;
  }
  .logo  { font-size: 64px; margin-bottom: 16px; }
  .title { font-size: 32px; font-weight: 700; letter-spacing: 2px; margin-bottom: 8px; }
  .sub   { font-size: 14px; color: #94a3b8; margin-bottom: 40px; }
  .spinner {
    width: 48px; height: 48px;
    border: 4px solid #334155;
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 0.9s linear infinite;
  }
  .msg { margin-top: 20px; font-size: 13px; color: #64748b; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
  <div class="logo">🔬</div>
  <div class="title">SMART-SPINE</div>
  <div class="sub">Posture Analysis System</div>
  <div class="spinner"></div>
  <div class="msg">Starting up, please wait…</div>
</body>
</html>
"""


def _cleanup_port(port: int) -> None:
    """Kill any zombie process holding the port on Windows."""
    if sys.platform != "win32":
        return
    try:
        cmd = f'powershell -Command "Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess | ForEach-Object {{ Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }}"'
        subprocess.run(cmd, shell=True, capture_output=True, timeout=5)
        time.sleep(0.5)
    except Exception:
        pass


def _wait_for_server(timeout: int = 60) -> bool:
    """Poll localhost until Streamlit responds or timeout (seconds)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(URL, timeout=1)
            return True
        except Exception:
            time.sleep(0.4)
    return False


def _load_app(window) -> None:
    """Background thread: wait for server then navigate window."""
    ready = _wait_for_server(timeout=60)
    if ready:
        window.load_url(URL)
    else:
        window.load_html("""
        <html><body style='font-family:sans-serif;padding:40px;background:#0f172a;color:#f87171'>
        <h2>⚠ Streamlit failed to start</h2>
        <p>Please check that all packages are installed:<br>
        <code>pip install -r requirements.txt</code></p>
        </body></html>
        """)


def main() -> None:
    # ── 0. Clean up any stale process holding the port ─────────────────────────
    _cleanup_port(PORT)

    # ── 1. Start Streamlit server in background ────────────────────────────────
    streamlit_proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", str(APP),
            "--server.port", str(PORT),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--server.fileWatcherType", "none",
        ],
        cwd=str(ROOT),
        env=os.environ.copy(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )

    # ── 2. Launch pywebview window ─────────────────────────────────────────────
    try:
        import webview
    except ImportError:
        print(
            "\n[SMART-SPINE] pywebview is not installed.\n"
            "Run:  pip install pywebview\n"
            "Then restart the app.\n"
        )
        streamlit_proc.terminate()
        sys.exit(1)

    window = webview.create_window(
        title="SMART-SPINE — Posture Analysis",
        html=SPLASH_HTML,
        width=1440,
        height=900,
        resizable=True,
        min_size=(900, 600),
        background_color="#0f172a",
    )

    threading.Thread(target=_load_app, args=(window,), daemon=True).start()

    for gui_backend in ("edgechromium", "cef", None):
        try:
            if gui_backend:
                webview.start(gui=gui_backend, debug=False)
            else:
                webview.start(debug=False)
            break
        except Exception:
            continue

    # ── 3. Clean up Streamlit when window is closed ────────────────────────────
    try:
        streamlit_proc.terminate()
        streamlit_proc.wait(timeout=5)
    except Exception:
        try:
            streamlit_proc.kill()
        except Exception:
            pass


if __name__ == "__main__":
    main()
