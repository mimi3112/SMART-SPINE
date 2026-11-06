"""
dashboard/phase1_tab.py — Tab 1: Offline Analytics.
Based on 10_dashboard.py, refactored into render_phase1().
Adds participant-specific LLM summary at the bottom.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dashboard.common import (
    NON_CLINICAL_FOOTER, POSTURE_COLORS, POSTURE_NAMES, TEST_PARTICIPANTS,
)
from dashboard.summary import render_phase1_summary


def _prepare(df: pd.DataFrame, model_package: dict, feature_columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["Posture"] = df["upperbody_label"].astype(str)
    df["Posture Name"] = df["Posture"].apply(lambda x: POSTURE_NAMES.get(x, x))
    if "cva_proxy_deg" in df.columns:
        df["Alignment Angle"] = pd.to_numeric(df["cva_proxy_deg"], errors="coerce")
    elif "cva_left_deg" in df.columns and "cva_right_deg" in df.columns:
        df["Alignment Angle"] = (
            pd.to_numeric(df["cva_left_deg"], errors="coerce") +
            pd.to_numeric(df["cva_right_deg"], errors="coerce")
        ) / 2
    elif "cva_deg" in df.columns:
        df["Alignment Angle"] = pd.to_numeric(df["cva_deg"], errors="coerce")
    else:
        df["Alignment Angle"] = np.nan

    model = model_package.get("model") if isinstance(model_package, dict) else model_package
    X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    valid = X.notna().all(axis=1)
    pred_posture = np.empty(len(df), dtype=object); pred_posture[:] = None
    pred_name    = np.empty(len(df), dtype=object); pred_name[:] = None
    if valid.any():
        preds = [str(x) for x in model.predict(X.loc[valid])]
        for i, p in zip(np.flatnonzero(valid.to_numpy()), preds):
            pred_posture[i] = p
            pred_name[i]    = POSTURE_NAMES.get(p, p)
    df["Predicted Posture"] = pred_posture
    df["Predicted Name"]    = pred_name
    return df


def render_phase1(raw_df: pd.DataFrame, model_package: dict, feature_columns: list[str]) -> None:
    df = _prepare(raw_df, model_package, feature_columns)

    available = sorted(
        pd.to_numeric(df["subject"], errors="coerce").dropna().astype(int).unique().tolist()
    )
    test_avail = [p for p in TEST_PARTICIPANTS if p in available]
    options    = test_avail if test_avail else available

    # ── Header row ─────────────────────────────────────────────────────────────
    h1, h2 = st.columns([4, 1])
    with h1:
        st.markdown('<div class="main-title">Dashboard Overview</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="main-subtitle">Phase 1 | Dataset-based posture classification '
            'and CVA-based alignment analysis</div>', unsafe_allow_html=True)
    with h2:
        selected = st.selectbox("Participant", options, index=0, key="p1_participant")

    st.markdown('<div class="top-line"></div>', unsafe_allow_html=True)

    participant_df = df[
        pd.to_numeric(df["subject"], errors="coerce").astype("Int64") == selected
    ].copy()

    test_df = df[pd.to_numeric(df["subject"], errors="coerce").isin(TEST_PARTICIPANTS)].copy()
    test_valid = test_df[test_df["Predicted Posture"].notna()]
    test_accuracy = (
        (test_valid["Posture"].astype(str) == test_valid["Predicted Posture"].astype(str)).mean() * 100
        if len(test_valid) > 0 else np.nan
    )

    # ── LLM Summary for Selected Participant (at the TOP, above all charts) ──
    render_phase1_summary(
        participant_df = participant_df,
        participant_id = selected,
        test_accuracy  = test_accuracy if pd.notna(test_accuracy) else None,
    )

    # ── Metric cards ───────────────────────────────────────────────────────────
    dominant = (participant_df["Posture Name"].value_counts().idxmax()
                if not participant_df.empty else "—")
    avg_angle = participant_df["Alignment Angle"].mean()
    angle_txt = f"{avg_angle:.1f}°" if pd.notna(avg_angle) else "—"

    m1, m2, m3, m4 = st.columns(4)
    for col, label, val, sub in [
        (m1, "Selected Participant", str(selected), "Held-out test participant"),
        (m2, "Total Observations", f"{len(participant_df):,}", "Records for selected participant"),
        (m3, "Most Common Posture", dominant, "Based on dataset labels"),
        (m4, "Average Alignment Angle", angle_txt, "CVA-based measure"),
    ]:
        col.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value" style="font-size:20px;">{val}</div>'
            f'<div class="metric-small">{sub}</div></div>', unsafe_allow_html=True)

    # ── Charts row ─────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="section-title">Posture Distribution</div>', unsafe_allow_html=True)
        actual_counts = (participant_df["Posture Name"].value_counts()
                         .rename_axis("Posture").reset_index(name="Count"))
        fig = px.pie(actual_counts, names="Posture", values="Count", hole=0.48,
                     color="Posture", color_discrete_map=POSTURE_COLORS)
        fig.update_traces(textposition="inside", textinfo="percent")
        fig.update_layout(height=285, margin=dict(l=8,r=8,t=10,b=8),
                          showlegend=True, legend=dict(font=dict(size=9)),
                          paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Model-Predicted Posture</div>', unsafe_allow_html=True)
        pred_counts = (participant_df["Predicted Name"].dropna().value_counts()
                       .rename_axis("Posture").reset_index(name="Count"))
        if not pred_counts.empty:
            fig2 = px.pie(pred_counts, names="Posture", values="Count", hole=0.48,
                          color="Posture", color_discrete_map=POSTURE_COLORS)
            fig2.update_traces(textposition="inside", textinfo="percent")
            fig2.update_layout(height=285, margin=dict(l=8,r=8,t=10,b=8),
                               showlegend=True, legend=dict(font=dict(size=9)),
                               paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Prediction data not available.")

    with c3:
        st.markdown('<div class="section-title">CVA-Based Alignment</div>', unsafe_allow_html=True)
        angle_data = participant_df["Alignment Angle"].dropna()
        if len(angle_data) > 0:
            fig3 = px.histogram(angle_data, x="Alignment Angle", nbins=18,
                                labels={"Alignment Angle": "Angle (degrees)"})
            fig3.update_layout(height=285, margin=dict(l=40,r=10,t=10,b=40),
                               xaxis_title="Angle (degrees)", yaxis_title="Frequency",
                               paper_bgcolor="white", plot_bgcolor="white", font=dict(size=10))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Alignment angle data not available.")

    # ── Info cards row ─────────────────────────────────────────────────────────
    r1, r2, r3 = st.columns([1.15, 1, 1])

    with r1:
        st.markdown('<div class="section-title">Model Evaluation</div>', unsafe_allow_html=True)
        acc_val = f"{test_accuracy:.2f}%" if pd.notna(test_accuracy) else "—"
        st.markdown(f"""
        <div class="info-card">
            <div class="info-title">Held-Out Test Performance</div>
            <div class="info-row"><span class="info-key">Evaluation participants</span><span class="info-val">10–13</span></div>
            <div class="info-row"><span class="info-key">Test observations</span><span class="info-val">{len(test_df):,}</span></div>
            <div class="info-row"><span class="info-key">Posture classes</span><span class="info-val">5</span></div>
            <div class="info-row"><span class="info-key">Test accuracy</span><span class="info-val">{acc_val}</span></div>
        </div>""", unsafe_allow_html=True)

    with r2:
        st.markdown('<div class="section-title">Participant Summary</div>', unsafe_allow_html=True)
        if not participant_df.empty:
            counts = participant_df["Posture Name"].value_counts().to_dict()
            rows_html = "".join(
                f'<div class="info-row"><span class="info-key">{p}</span>'
                f'<span class="info-val">{counts.get(p,0)}</span></div>'
                for p in POSTURE_NAMES.values()
            )
            st.markdown(
                f'<div class="info-card"><div class="info-title">Posture Records</div>'
                f'{rows_html}</div>', unsafe_allow_html=True)

    with r3:
        st.markdown('<div class="section-title">Analysis Overview</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <div class="info-title">What this dashboard shows</div>
            <div class="info-row"><span class="info-key">Posture</span><span class="info-val">5-class analysis</span></div>
            <div class="info-row"><span class="info-key">Alignment</span><span class="info-val">CVA-based angle</span></div>
            <div class="info-row"><span class="info-key">Input source</span><span class="info-val">Existing dataset</span></div>
            <div class="info-row"><span class="info-key">Evaluation</span><span class="info-val">Participants 10–13</span></div>
            <div class="info-row"><span class="info-key">Body weight</span><span class="info-val">Not used in Phase 1</span></div>
        </div>""", unsafe_allow_html=True)

    # ── Overall test summary ────────────────────────────────────────────────────
    st.markdown('<div class="section-title" style="margin-top:18px;">Overall Test-Participant Summary</div>',
                unsafe_allow_html=True)
    o1, o2 = st.columns([1, 2])
    with o1:
        overall_counts = (test_df["Posture Name"].value_counts()
                          .rename_axis("Posture").reset_index(name="Count"))
        fig_o = px.pie(overall_counts, names="Posture", values="Count", hole=0.45,
                       color="Posture", color_discrete_map=POSTURE_COLORS)
        fig_o.update_traces(textposition="inside", textinfo="percent+label")
        fig_o.update_layout(height=250, margin=dict(l=5,r=5,t=5,b=5),
                            showlegend=False, paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig_o, use_container_width=True)
    with o2:
        tbl_data = []
        for pid in TEST_PARTICIPANTS:
            pdata = df[pd.to_numeric(df["subject"], errors="coerce").astype("Int64") == pid]
            dom_p = (pdata["Posture Name"].value_counts().idxmax()
                     if not pdata.empty else "—")
            pacc  = np.nan
            pv    = pdata[pdata["Predicted Posture"].notna()]
            if len(pv) > 0:
                pacc = (pv["Posture"].astype(str) == pv["Predicted Posture"].astype(str)).mean() * 100
            tbl_data.append({
                "Participant": pid,
                "Observations": len(pdata),
                "Dominant Posture": dom_p,
                "Accuracy (%)": f"{pacc:.1f}" if pd.notna(pacc) else "—",
            })
        st.dataframe(pd.DataFrame(tbl_data), use_container_width=True, hide_index=True)

    st.markdown('<div class="note" style="margin-top:8px;">Offline analysis uses recorded dataset '
                'landmarks; it does not use body weight.</div>', unsafe_allow_html=True)

    st.markdown(NON_CLINICAL_FOOTER, unsafe_allow_html=True)
