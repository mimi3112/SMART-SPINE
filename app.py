"""
app.py — SMART-SPINE unified entry point.

Run with:
    streamlit run app.py
    python app.py          (re-launches itself via streamlit)
    run.bat                (Windows convenience wrapper)
    run_desktop.bat        (Native desktop window wrapper)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# ── Ensure the project root is on sys.path so dashboard/* can import paths.py ──
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Load environment variables (.env) ──────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env", override=True)
except Exception:
    pass


# ── Bootstrap: if not inside a Streamlit runtime, re-launch via streamlit ──────
try:
    from streamlit.runtime import exists as _st_runtime_exists
    _in_streamlit = _st_runtime_exists()
except Exception:
    _in_streamlit = False

if not _in_streamlit:
    result = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())],
        check=False,
    )
    sys.exit(result.returncode)


# ── Normal Streamlit execution path ────────────────────────────────────────────
import streamlit as st

# MUST be the first Streamlit call
st.set_page_config(
    page_title="SMART-SPINE",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.common import (
    DASHBOARD_CSS,
    NON_CLINICAL_FOOTER,
    load_features_df,
    load_model_package,
)
from dashboard import tracker as _tracker
from dashboard.phase1_tab import render_phase1
from dashboard.live_tab import render_live
from dashboard.trends_tab import render_trends

# ── Inject CSS ─────────────────────────────────────────────────────────────────
st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand block
    st.markdown(
        """
        <div class="brand">
            <div class="brand-title">SMART-SPINE</div>
            <div class="brand-subtitle">
                Posture Analysis System<br>
                Phases 1 · 2 · Trends
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Global body-weight input
    if "body_weight" not in st.session_state:
        st.session_state["body_weight"] = 60.0

    body_weight = st.number_input(
        "Body weight (kg)",
        min_value=20.0,
        max_value=250.0,
        value=float(st.session_state["body_weight"]),
        step=0.5,
        help="Used in Phase 2 live analysis to estimate cervical load.",
        key="body_weight_input",
    )
    st.session_state["body_weight"] = body_weight

    # Tracker status chip
    running = _tracker.is_running()
    mode    = _tracker.get_mode()
    if running and mode == "continuous":
        chip_class, chip_text = "chip-running", "🟢 Tracking"
    elif running and mode == "once":
        chip_class, chip_text = "chip-once",    "📷 Capturing"
    else:
        chip_class, chip_text = "chip-idle",    "⚪ Idle"

    st.markdown(
        f'<div style="margin:6px 0 12px 0;">Tracker: '
        f'<span class="status-chip {chip_class}">{chip_text}</span></div>',
        unsafe_allow_html=True,
    )

    # Privacy note
    st.markdown(
        """
        <div class="note" style="margin-top:12px;">
            🔒 <b>Privacy:</b> No images are stored.<br>
            Only numeric posture values are saved to
            <code>results/live_analysis.csv</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Load shared data ───────────────────────────────────────────────────────────
model_package  = load_model_package()
raw_df         = load_features_df()

# Resolve feature columns once
if model_package is not None and isinstance(model_package, dict):
    feature_columns = model_package.get("feature_columns") or []
elif raw_df is not None:
    feature_columns = [
        c for c in raw_df.columns
        if c.endswith("_x") or c.endswith("_y") or c.endswith("_z")
    ]
else:
    feature_columns = []

if raw_df is not None and feature_columns:
    feature_columns = [c for c in feature_columns if c in raw_df.columns]

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Phase 1: Offline Analytics",
    "📡 Phase 2: Live Monitor",
    "📈 Posture Trends",
])

with tab1:
    if raw_df is None or model_package is None:
        st.error(
            "Could not load the dataset or model. "
            "Ensure `results/features.csv` and `models/posture_model.pkl` exist."
        )
    else:
        render_phase1(raw_df, model_package, feature_columns)

with tab2:
    render_live()

with tab3:
    render_trends()
