"""
dashboard/tracker.py — Subprocess manager for landmark_detection.py.
Windows-safe. One process at a time. Logs to results/tracker.log.
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
import streamlit as st
from paths import LIVE_SCRIPT, PROJECT_ROOT, TRACKER_LOG


def _proc() -> subprocess.Popen | None:
    return st.session_state.get("tracker_proc")


def is_running() -> bool:
    p = _proc()
    return p is not None and p.poll() is None


def get_mode() -> str:
    return st.session_state.get("tracker_mode", "")


def start_continuous(weight: float) -> None:
    if is_running():
        return
    log = open(TRACKER_LOG, "a", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(LIVE_SCRIPT), "--weight", str(weight), "--interval", "0"],
        cwd=str(PROJECT_ROOT),
        stdout=log, stderr=log,
        creationflags=0,
    )
    st.session_state["tracker_proc"] = proc
    st.session_state["tracker_mode"] = "continuous"
    st.session_state["tracker_log_fh"] = log


def capture_once(weight: float) -> None:
    if is_running():
        return
    log = open(TRACKER_LOG, "a", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(LIVE_SCRIPT), "--weight", str(weight), "--once"],
        cwd=str(PROJECT_ROOT),
        stdout=log, stderr=log,
        creationflags=0,
    )
    st.session_state["tracker_proc"] = proc
    st.session_state["tracker_mode"] = "once"
    st.session_state["tracker_log_fh"] = log


def stop() -> None:
    p = _proc()
    if p is None:
        return
    try:
        p.terminate()
        p.wait(timeout=5)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass
    fh = st.session_state.pop("tracker_log_fh", None)
    if fh:
        try:
            fh.close()
        except Exception:
            pass
    st.session_state.pop("tracker_proc", None)
    st.session_state.pop("tracker_mode", None)


def last_log_lines(n: int = 10) -> str:
    try:
        lines = TRACKER_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""
