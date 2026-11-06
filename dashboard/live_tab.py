"""
dashboard/live_tab.py — Tab 2: Phase 2 Live Monitor.

Session isolation:
  - session_start_time is stored in st.session_state["session_start_time"]
    when the user clicks Start Tracking.
  - session_end_time is stored when Stop Tracking is clicked.
  - The session summary is generated from ONLY the rows between those two times.
  - A new Start clears the previous session context.
"""
from __future__ import annotations
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dashboard.common import (
    CVA_GOOD_MIN, CVA_WARN_MIN, KG_TO_N, NON_CLINICAL_FOOTER,
    POSTURE_COLORS, POSTURE_NAMES, load_live_log,
)
from dashboard import tracker as _tracker
from dashboard.summary import render_session_summary


def _get_session_data(full_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Returns (session_df, session_label).
    1. If a session is tracked in st.session_state (Start Tracking was clicked),
       filters strictly between session_start_time and session_end_time (or now).
    2. If no session was started in current runtime, but full_df has recorded data,
       returns the most recent recorded session (contiguous cluster of samples).
    3. If full_df is empty, returns (empty_df, "Live Session").
    """
    start = st.session_state.get("session_start_time")
    end   = st.session_state.get("session_end_time")

    if start is not None:
        if full_df.empty:
            return pd.DataFrame(), "Current Session"
        df = full_df[full_df["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["timestamp"] <= pd.Timestamp(end)]
        s = start.strftime("%H:%M:%S")
        e = end.strftime("%H:%M:%S") if end else "ongoing"
        return df.reset_index(drop=True), f"Current Session ({s} – {e})"

    if full_df.empty:
        return pd.DataFrame(), "Live Session"

    # Auto-detect latest session cluster (gap > 3 minutes)
    df_sorted = full_df.sort_values("timestamp").reset_index(drop=True)
    time_diffs = df_sorted["timestamp"].diff()
    session_starts = df_sorted.index[time_diffs > pd.Timedelta(minutes=3)].tolist()
    last_session_idx = session_starts[-1] if session_starts else 0
    latest_sess_df = df_sorted.iloc[last_session_idx:].reset_index(drop=True)

    t_start = latest_sess_df["timestamp"].min()
    t_end   = latest_sess_df["timestamp"].max()
    lbl = f"Last Recorded Session ({t_start.strftime('%H:%M:%S')} – {t_end.strftime('%H:%M:%S')})"
    return latest_sess_df, lbl


@st.fragment(run_every=2)
def _live_display(body_weight: float) -> None:
    """Auto-refreshing fragment — updates every 2 s while tracker is running."""
    full_df = load_live_log()
    sess_df, _ = _get_session_data(full_df)
    running = _tracker.is_running()

    if sess_df.empty:
        if running:
            st.info("Tracking in progress... webcam window is active, recording first observation.")
        else:
            st.info("Click **Start Tracking** to begin a session.")
        return

    # Ensure posture_name is present
    if "posture_name" not in sess_df.columns and "posture" in sess_df.columns:
        sess_df["posture_name"] = sess_df["posture"].map(POSTURE_NAMES).fillna(sess_df["posture"])

    # ── Latest reading ────────────────────────────────────────────────────────
    latest = sess_df.iloc[-1]
    posture_code = str(latest.get("posture", ""))
    posture_name = POSTURE_NAMES.get(posture_code, posture_code)
    confidence   = float(latest.get("confidence", 0))
    cva          = latest.get("cva_proxy_deg", None)
    flexion      = latest.get("estimated_flexion_deg", None)
    load_kg      = latest.get("estimated_cervical_load_kg", None)

    badge_color = POSTURE_COLORS.get(posture_name, "#6b7280")
    st.markdown(
        f'<div style="margin-bottom:12px;">'
        f'<span style="background:{badge_color};color:#fff;padding:6px 18px;'
        f'border-radius:999px;font-size:15px;font-weight:700;">'
        f'{posture_name}</span>'
        f'<span style="font-size:13px;color:#6b7280;margin-left:12px;">'
        f'Confidence: <b>{confidence:.1f}%</b></span>'
        f'<span style="font-size:12px;color:#9ca3af;margin-left:14px;">'
        f'Timestamp: {latest.get("timestamp", "")}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    # CVA gauge
    if cva is not None and pd.notna(cva):
        cva_val = float(cva)
        if cva_val >= CVA_GOOD_MIN:
            gauge_color = "#22c55e"
        elif cva_val >= CVA_WARN_MIN:
            gauge_color = "#f59e0b"
        else:
            gauge_color = "#ef4444"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cva_val,
            title={"text": "Head-Shoulder Angle (°)"},
            gauge={
                "axis": {"range": [0, 90]},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [0, CVA_WARN_MIN],  "color": "#fee2e2"},
                    {"range": [CVA_WARN_MIN, CVA_GOOD_MIN], "color": "#fef3c7"},
                    {"range": [CVA_GOOD_MIN, 90], "color": "#dcfce7"},
                ],
            },
            number={"suffix": "°"},
        ))
        fig_gauge.update_layout(height=210, margin=dict(l=20,r=20,t=40,b=10),
                                paper_bgcolor="white")
        col1.plotly_chart(fig_gauge, use_container_width=True)
    else:
        col1.info("Awaiting CVA reading...")

    with col2:
        if flexion is not None and pd.notna(flexion):
            st.metric("Head Lean (approx.)", f"{float(flexion):.1f}°",
                      help="0° = fully upright. Higher = more forward lean.")
        if load_kg is not None and pd.notna(load_kg):
            load_n = float(load_kg) * KG_TO_N
            st.metric("Estimated Neck Load", f"{float(load_kg):.1f} kg-eq",
                      help=f"= {load_n:.1f} N. Research estimate only — not actual spine force.")

    with col3:
        n_sess = len(sess_df)
        pct_up = (sess_df["posture"] == "TUP").mean() * 100 if n_sess > 0 else 0
        st.metric("Session Observations", n_sess)
        st.metric("Upright This Session", f"{pct_up:.0f}%")

    # ── Session Graphs (Timeline and Breakdown) ───────────────────────────────
    st.markdown("#### Session Posture Graphs")
    g_col1, g_col2 = st.columns([3, 2])

    with g_col1:
        # Timeline line chart
        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=sess_df["timestamp"],
            y=sess_df["cva_proxy_deg"],
            mode="lines+markers",
            name="Head-Shoulder Angle (°)",
            line=dict(color="#2563eb", width=2.5),
            marker=dict(size=7, color="#1d4ed8")
        ))
        if "estimated_cervical_load_kg" in sess_df.columns:
            fig_timeline.add_trace(go.Scatter(
                x=sess_df["timestamp"],
                y=sess_df["estimated_cervical_load_kg"],
                mode="lines+markers",
                name="Neck Load (kg-eq)",
                yaxis="y2",
                line=dict(color="#f97316", width=2, dash="dash"),
                marker=dict(size=6, color="#ea580c")
            ))

        fig_timeline.add_hline(y=CVA_GOOD_MIN, line_dash="dot", line_color="#16a34a",
                               annotation_text="Good (≥50°)", annotation_position="top left")
        fig_timeline.add_hline(y=CVA_WARN_MIN, line_dash="dot", line_color="#ef4444",
                               annotation_text="Caution (40°)", annotation_position="bottom left")

        fig_timeline.update_layout(
            title="Head-Shoulder Angle & Neck Load Over Time",
            xaxis_title="Timestamp",
            yaxis=dict(title="Angle (°)", range=[0, 90]),
            yaxis2=dict(title="Load (kg-eq)", overlaying="y", side="right", range=[0, 35]),
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="white",
            plot_bgcolor="#f8fafc",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    with g_col2:
        # Posture Breakdown Bar
        p_col = "posture_name" if "posture_name" in sess_df.columns else "posture"
        p_counts = sess_df[p_col].value_counts().reset_index()
        p_counts.columns = ["Posture", "Count"]
        p_counts["Percentage"] = (p_counts["Count"] / len(sess_df)) * 100
        
        fig_dist = px.bar(
            p_counts,
            x="Percentage",
            y="Posture",
            orientation="h",
            text=p_counts["Percentage"].apply(lambda v: f"{v:.1f}%"),
            title="Posture Breakdown This Session",
            color="Posture",
            color_discrete_map=POSTURE_COLORS
        )
        fig_dist.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
            paper_bgcolor="white",
            plot_bgcolor="#f8fafc",
            xaxis=dict(range=[0, max(100, float(p_counts["Percentage"].max() + 15))])
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # ── Last 10 rows table ────────────────────────────────────────────────────
    st.markdown("**Recent Session Observations:**")
    display_cols = ["timestamp", "posture_name", "confidence",
                    "cva_proxy_deg", "estimated_flexion_deg", "estimated_cervical_load_kg"]
    display_cols = [c for c in display_cols if c in sess_df.columns]
    st.dataframe(sess_df[display_cols].tail(10), use_container_width=True, hide_index=True)


def render_live() -> None:
    body_weight = float(st.session_state.get("body_weight", 60.0))

    st.markdown(
        '<div class="main-title">Phase 2: Live Monitor</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="main-subtitle">Body weight used: <b>{body_weight} kg</b> '
        f'(set in sidebar)</div>', unsafe_allow_html=True)
    st.markdown('<div class="top-line"></div>', unsafe_allow_html=True)

    running = _tracker.is_running()
    mode    = _tracker.get_mode()

    # ── Controls ──────────────────────────────────────────────────────────────
    btn1, btn2, btn3 = st.columns(3)

    with btn1:
        start_disabled = running
        if st.button("Start Tracking", disabled=start_disabled,
                     help="Already running" if running else None,
                     use_container_width=True, type="primary"):
            # Clear previous session
            st.session_state["session_start_time"] = datetime.now()
            st.session_state.pop("session_end_time", None)
            # Clear cached summaries for new session
            cache = st.session_state.get("_summary_cache", {})
            keys_to_drop = [k for k in cache if k.startswith("p2_")]
            for k in keys_to_drop:
                del cache[k]
            _tracker.start_continuous(body_weight)
            st.rerun()

    with btn2:
        if st.button("Stop Tracking", disabled=not running,
                     use_container_width=True):
            st.session_state["session_end_time"] = datetime.now()
            _tracker.stop()
            st.rerun()

    with btn3:
        capture_disabled = running
        if st.button("Capture Posture Now", disabled=capture_disabled,
                     help="Stop continuous tracking first" if running else None,
                     use_container_width=True):
            if "session_start_time" not in st.session_state:
                st.session_state["session_start_time"] = datetime.now()
                st.session_state.pop("session_end_time", None)
            _tracker.capture_once(body_weight)
            st.rerun()

    # Status line
    if running and mode == "continuous":
        st.info("🟢 **Tracking active:** The camera preview window ('Smart-Spine - Real-Time Analysis') is recording your posture. Real-time graphs and metrics update below. Click **Stop Tracking** or press 'Q' in the camera window to finalize your AI summary.")
    elif running and mode == "once":
        st.info("Capturing one sample via webcam...")

    # Tracker error log
    if not running and _tracker.last_log_lines(3):
        with st.expander("Tracker log"):
            st.code(_tracker.last_log_lines(15))

    st.markdown("---")

    # ── Session Summary (at the TOP, above live gauges, graphs & table) ─────────
    full_df = load_live_log()
    sess_df, sess_label = _get_session_data(full_df)

    if not running:
        render_session_summary(sess_df, session_label=sess_label)
        st.markdown("---")
    else:
        st.info(f"🟢 **Session in progress:** {len(sess_df)} observation(s) collected so far. The AI summary will be generated from this session when tracking stops.")
        st.markdown("---")

    # ── Live display (auto-refreshing fragment) ────────────────────────────────
    _live_display(body_weight)

    st.markdown(NON_CLINICAL_FOOTER, unsafe_allow_html=True)
