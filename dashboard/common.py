"""
dashboard/common.py — Shared constants, CSS, cached loaders, and live-log reader.
"""
from __future__ import annotations
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from paths import FEATURE_FILE, LIVE_LOG, MODEL_FILE

# ── Posture definitions ────────────────────────────────────────────────────────
POSTURE_NAMES: dict[str, str] = {
    "TUP": "Upright",
    "TLF": "Leaning Forward",
    "TLB": "Leaning Backward",
    "TLL": "Leaning Left",
    "TLR": "Leaning Right",
}
POSTURE_COLORS: dict[str, str] = {
    "Upright":          "#3498DB",
    "Leaning Backward": "#2ECC71",
    "Leaning Forward":  "#F39C12",
    "Leaning Left":     "#E74C3C",
    "Leaning Right":    "#9B59B6",
}
TEST_PARTICIPANTS: list[int] = [10, 11, 12, 13]
CVA_GOOD_MIN: int  = 75
CVA_WARN_MIN: int  = 60
KG_TO_N: float     = 9.80665

# ── CSS ────────────────────────────────────────────────────────────────────────
DASHBOARD_CSS = """
<style>
html, body, [class*="css"] { font-family: Arial, sans-serif; }
.stApp { background: #f7f8fa; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5e7eb; }
[data-testid="stSidebar"] > div:first-child { padding-top: 18px; }
.brand { padding: 0 10px 16px 10px; border-bottom: 1px solid #e5e7eb; margin-bottom: 16px; }
.brand-title { font-size: 24px; font-weight: 700; color: #18212f; margin-bottom: 4px; }
.brand-subtitle { font-size: 12px; color: #6b7280; line-height: 1.45; }
.side-box { border: 1px solid #e5e7eb; border-radius: 9px; padding: 12px; margin-top: 18px; background: #ffffff; }
.side-label { color: #6b7280; font-size: 11px; margin-top: 9px; }
.side-value { color: #263241; font-size: 14px; font-weight: 600; }
.main-title { font-size: 28px; font-weight: 700; color: #18212f; line-height: 1.25; margin: 0; padding: 0; }
.main-subtitle { font-size: 13px; color: #7b8491; margin-top: 4px; line-height: 1.4; }
.top-line { margin: 13px 0 16px 0; border-top: 1px solid #e2e5e9; }
.section-title { font-size: 15px; font-weight: 700; color: #263241; margin: 4px 0 8px 0; }
.metric-card { background: white; border: 1px solid #e1e5ea; border-radius: 9px; padding: 13px 15px; min-height: 88px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
.metric-label { color: #687282; font-size: 12px; margin-bottom: 7px; }
.metric-value { color: #1d4ed8; font-size: 25px; font-weight: 700; line-height: 1.1; }
.metric-small { color: #7a8491; font-size: 10px; margin-top: 5px; }
.panel { background: white; border: 1px solid #e1e5ea; border-radius: 9px; padding: 8px 10px 4px 10px; margin-top: 12px; }
.info-card { background: white; border: 1px solid #e1e5ea; border-radius: 9px; padding: 14px 16px; min-height: 180px; }
.info-title { font-size: 14px; font-weight: 700; color: #263241; margin-bottom: 10px; }
.info-row { display: flex; justify-content: space-between; border-bottom: 1px solid #f0f1f3; padding: 7px 0; font-size: 12px; }
.info-row:last-child { border-bottom: none; }
.info-key { color: #737d8a; }
.info-val { color: #273444; font-weight: 600; }
.note { background: #ffffff; border: 1px solid #e1e5ea; border-radius: 8px; padding: 10px 13px; color: #697382; font-size: 11px; line-height: 1.5; margin-top: 12px; }
.stSelectbox label { font-size: 12px !important; color: #657080 !important; }
.posture-badge { display: inline-block; padding: 6px 18px; border-radius: 999px; font-size: 15px; font-weight: 700; color: #ffffff; letter-spacing: 0.04em; margin-bottom: 4px; }
.status-chip { display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: 11px; font-weight: 600; margin-left: 6px; }
.chip-running { background: #d1fae5; color: #065f46; }
.chip-idle    { background: #f3f4f6; color: #4b5563; }
.chip-once    { background: #fef3c7; color: #92400e; }
footer { visibility: hidden; }
</style>
"""

NON_CLINICAL_FOOTER = """
<div class="note">
<b>What do these numbers mean?</b><br>
<b>CVA proxy</b> is an angle computed from camera landmarks of the ear and shoulder.
It is <em>not</em> a clinical craniovertebral angle and cannot diagnose forward head posture.<br>
<b>Estimated cervical load</b> is a model-based estimate from a published posture-load reference curve,
scaled by your entered body weight. It is <b>NOT</b> a measurement of the force in your spine.<br>
<em>For research and prototype use only. Not medical advice.
If you have neck pain or health concerns, consult a qualified clinician.</em>
</div>
"""

# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading posture model...")
def load_model_package() -> dict | None:
    if not MODEL_FILE.exists():
        st.error(f"Model file not found: `{MODEL_FILE}`")
        return None
    try:
        return joblib.load(MODEL_FILE)
    except Exception as exc:
        st.error(f"Could not load posture model: {exc}")
        return None


@st.cache_data(show_spinner="Loading dataset...")
def load_features_df() -> pd.DataFrame | None:
    if not FEATURE_FILE.exists():
        st.error(f"Feature file not found: `{FEATURE_FILE}`")
        return None
    try:
        return pd.read_csv(FEATURE_FILE)
    except Exception as exc:
        st.error(f"Could not load features.csv: {exc}")
        return None


# ── Live log loader (mtime-keyed cache) ───────────────────────────────────────
_LIVE_COLUMNS = [
    "timestamp", "body_weight_kg", "posture", "confidence",
    "cva_proxy_deg", "estimated_flexion_deg", "estimated_cervical_load_kg",
]
_EMPTY_LIVE = pd.DataFrame(columns=_LIVE_COLUMNS)


def _live_log_mtime() -> float:
    try:
        return os.path.getmtime(LIVE_LOG)
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def _load_live_log_cached(mtime: float) -> pd.DataFrame:
    if not LIVE_LOG.exists():
        return _EMPTY_LIVE.copy()
    try:
        df = pd.read_csv(LIVE_LOG, on_bad_lines="skip", names=_LIVE_COLUMNS, header=0)
    except Exception:
        return _EMPTY_LIVE.copy()
    if df.empty:
        return _EMPTY_LIVE.copy()
    for col in ["body_weight_kg", "confidence", "cva_proxy_deg",
                "estimated_flexion_deg", "estimated_cervical_load_kg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["estimated_cervical_load_N"] = df["estimated_cervical_load_kg"] * KG_TO_N
    df["posture_name"] = df["posture"].map(POSTURE_NAMES).fillna(df["posture"])
    return df


def load_live_log() -> pd.DataFrame:
    return _load_live_log_cached(_live_log_mtime())
