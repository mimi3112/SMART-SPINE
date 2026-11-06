"""
dashboard/trends_tab.py — Tab 3: Posture Trends (historical live data).

Displays historical live tracking data loaded via load_live_log():
  - Summary metric cards (Samples, Upright %, Avg CVA, Avg Load, Max streak)
  - Posture over time scatter
  - CVA proxy over time with threshold lines
  - Cervical load over time
  - Posture distribution pie / bar
  - CSV download of filtered data
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import (
    CVA_GOOD_MIN,
    CVA_WARN_MIN,
    KG_TO_N,
    NON_CLINICAL_FOOTER,
    POSTURE_COLORS,
    load_live_log,
)


def _longest_non_upright_streak(df: pd.DataFrame) -> int:
    """Count longest consecutive run of non-upright postures."""
    if df.empty:
        return 0
    streak = 0
    max_streak = 0
    for p in df.get("posture", []):
        if str(p) != "TUP":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def render_trends() -> None:
    df = load_live_log()

    if df.empty:
        st.info(
            "📊 No live data recorded yet. "
            "Start tracking in **Phase 2: Live Monitor** to collect posture history."
        )
        return

    # ── Controls ─────────────────────────────────────────────────────────────
    ctrl1, ctrl2 = st.columns([3, 2])

    with ctrl1:
        min_date = df["timestamp"].min().date()
        max_date = df["timestamp"].max().date()
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="trends_date_range",
        )

    with ctrl2:
        load_unit = st.radio(
            "Load unit",
            options=["kg-eq", "Newtons"],
            horizontal=True,
            key="trends_load_unit",
        )

    # Filter by date range
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_dt = pd.Timestamp(date_range[0])
        end_dt = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
        fdf = df[(df["timestamp"] >= start_dt) & (df["timestamp"] < end_dt)].copy()
    else:
        fdf = df.copy()

    if fdf.empty:
        st.warning("No data in the selected date range.")
        return

    # ── Summary metrics ───────────────────────────────────────────────────────
    n_samples = len(fdf)
    pct_upright = (fdf["posture"] == "TUP").mean() * 100
    avg_cva = fdf["cva_proxy_deg"].mean()
    avg_load = fdf["estimated_cervical_load_kg"].mean()
    streak_len = _longest_non_upright_streak(fdf)

    s1, s2, s3, s4, s5 = st.columns(5)
    cards = [
        (s1, "Samples", f"{n_samples:,}", "in date range"),
        (s2, "Upright %", f"{pct_upright:.1f}%", "of all samples"),
        (s3, "Avg CVA", f"{avg_cva:.1f}°" if pd.notna(avg_cva) else "—", "head-shoulder angle"),
        (
            s4,
            "Avg Load",
            f"{avg_load * (KG_TO_N if load_unit == 'Newtons' else 1.0):.1f} {'N' if load_unit == 'Newtons' else 'kg-eq'}",
            "research estimate",
        ),
        (s5, "Max non-upright streak", f"{streak_len}", "consecutive samples"),
    ]
    for col, label, val, sub in cards:
        col.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size:22px;">{val}</div>
                <div class="metric-small">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 1. Posture timeline ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">Posture Over Time</div>', unsafe_allow_html=True)
    fig_timeline = px.scatter(
        fdf,
        x="timestamp",
        y="posture_name",
        color="posture_name",
        color_discrete_map=POSTURE_COLORS,
        labels={"timestamp": "Time", "posture_name": "Posture"},
    )
    fig_timeline.update_traces(marker=dict(size=9, symbol="circle"))
    fig_timeline.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=True,
        legend=dict(font=dict(size=10)),
        xaxis=dict(showgrid=True, gridcolor="#f0f1f3"),
        yaxis=dict(showgrid=True, gridcolor="#f0f1f3"),
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    # ── 2. CVA proxy line chart ───────────────────────────────────────────────
    st.markdown('<div class="section-title">CVA Proxy Over Time</div>', unsafe_allow_html=True)
    fig_cva = go.Figure()
    fig_cva.add_trace(go.Scatter(
        x=fdf["timestamp"],
        y=fdf["cva_proxy_deg"],
        mode="lines+markers",
        name="CVA proxy",
        line=dict(color="#3b82f6", width=2),
        marker=dict(size=5),
    ))
    fig_cva.add_hline(
        y=CVA_GOOD_MIN,
        line=dict(color="#22c55e", dash="dash", width=1.5),
        annotation_text=f"Good threshold ({CVA_GOOD_MIN}°)",
        annotation_position="top right",
    )
    fig_cva.add_hline(
        y=CVA_WARN_MIN,
        line=dict(color="#f59e0b", dash="dash", width=1.5),
        annotation_text=f"Warning threshold ({CVA_WARN_MIN}°)",
        annotation_position="bottom right",
    )
    fig_cva.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis_title="CVA (degrees)",
        xaxis_title="Time",
        xaxis=dict(showgrid=True, gridcolor="#f0f1f3"),
        yaxis=dict(showgrid=True, gridcolor="#f0f1f3", range=[0, 95]),
    )
    st.plotly_chart(fig_cva, use_container_width=True)

    # ── 3. Cervical load chart ────────────────────────────────────────────────
    if load_unit == "Newtons":
        load_col = "estimated_cervical_load_N"
        load_label = "Cervical Load (N)"
    else:
        load_col = "estimated_cervical_load_kg"
        load_label = "Cervical Load (kg-eq)"

    st.markdown(f'<div class="section-title">{load_label} Over Time</div>', unsafe_allow_html=True)
    fig_load = go.Figure()
    fig_load.add_trace(go.Scatter(
        x=fdf["timestamp"],
        y=fdf[load_col],
        mode="lines+markers",
        name=load_label,
        line=dict(color="#8b5cf6", width=2),
        marker=dict(size=5),
        fill="tozeroy",
        fillcolor="rgba(139,92,246,0.08)",
    ))
    fig_load.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis_title=load_label,
        xaxis_title="Time",
        xaxis=dict(showgrid=True, gridcolor="#f0f1f3"),
        yaxis=dict(showgrid=True, gridcolor="#f0f1f3"),
    )
    st.plotly_chart(fig_load, use_container_width=True)

    # ── 4. Distribution donut & peak load bar ─────────────────────────────────
    d1, d2 = st.columns([1, 1])

    with d1:
        st.markdown('<div class="section-title">Posture Distribution</div>', unsafe_allow_html=True)
        dist_counts = (
            fdf["posture_name"]
            .value_counts()
            .rename_axis("Posture")
            .reset_index(name="Count")
        )
        fig_donut = px.pie(
            dist_counts,
            names="Posture",
            values="Count",
            hole=0.5,
            color="Posture",
            color_discrete_map=POSTURE_COLORS,
        )
        fig_donut.update_traces(textposition="inside", textinfo="percent+label")
        fig_donut.update_layout(
            height=300,
            margin=dict(l=5, r=5, t=5, b=5),
            showlegend=True,
            legend=dict(font=dict(size=10)),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with d2:
        st.markdown('<div class="section-title">Peak Load per Posture</div>', unsafe_allow_html=True)
        peak_per_posture = (
            fdf.groupby("posture_name")[load_col]
            .max()
            .reset_index()
            .rename(columns={"posture_name": "Posture", load_col: load_label})
        )
        fig_bar = px.bar(
            peak_per_posture,
            x="Posture",
            y=load_label,
            color="Posture",
            color_discrete_map=POSTURE_COLORS,
        )
        fig_bar.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=60),
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=False,
            yaxis=dict(showgrid=True, gridcolor="#f0f1f3"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Download button ───────────────────────────────────────────────────────
    csv_data = fdf.drop(
        columns=["posture_name", "estimated_cervical_load_N"], errors="ignore"
    ).to_csv(index=False)
    st.download_button(
        label="⬇ Download filtered data as CSV",
        data=csv_data,
        file_name="smart_spine_trends.csv",
        mime="text/csv",
    )

    st.markdown(NON_CLINICAL_FOOTER, unsafe_allow_html=True)
