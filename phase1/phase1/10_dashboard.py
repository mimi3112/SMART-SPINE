import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# SMART-SPINE | Phase 1 Offline Posture Analysis Dashboard
# ============================================================

st.set_page_config(
    page_title="SMART-SPINE | Phase 1",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FEATURE_FILE = os.path.join(BASE_DIR, "results", "features.csv")
MODEL_FILE = os.path.join(BASE_DIR, "models", "posture_model.pkl")

# -----------------------------
# Posture definitions
# -----------------------------
POSTURE_NAMES = {
    "TUP": "Upright",
    "TLF": "Leaning Forward",
    "TLB": "Leaning Backward",
    "TLL": "Leaning Left",
    "TLR": "Leaning Right",
}

POSTURE_COLORS = {
    "Upright": "#3498DB",
    "Leaning Backward": "#2ECC71",
    "Leaning Forward": "#F39C12",
    "Leaning Left": "#E74C3C",
    "Leaning Right": "#9B59B6",
}

TEST_PARTICIPANTS = [10, 11, 12, 13]

# -----------------------------
# CSS
# -----------------------------
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: Arial, sans-serif;
    }

    .stApp {
        background: #f7f8fa;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 18px;
    }

    .brand {
        padding: 0 10px 16px 10px;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 16px;
    }

    .brand-title {
        font-size: 24px;
        font-weight: 700;
        color: #18212f;
        margin-bottom: 4px;
    }

    .brand-subtitle {
        font-size: 12px;
        color: #6b7280;
        line-height: 1.45;
    }

    .nav-item {
        padding: 9px 10px;
        margin: 4px 0;
        border-radius: 7px;
        font-size: 13px;
        color: #4b5563;
    }

    .nav-active {
        background: #1f73e8;
        color: white;
        font-weight: 600;
    }

    .side-box {
        border: 1px solid #e5e7eb;
        border-radius: 9px;
        padding: 12px;
        margin-top: 18px;
        background: #ffffff;
    }

    .side-label {
        color: #6b7280;
        font-size: 11px;
        margin-top: 9px;
    }

    .side-value {
        color: #263241;
        font-size: 14px;
        font-weight: 600;
    }

    .main-title {
        font-size: 28px;
        font-weight: 700;
        color: #18212f;
        line-height: 1.25;
        margin: 0;
        padding: 0;
    }

    .main-subtitle {
        font-size: 13px;
        color: #7b8491;
        margin-top: 4px;
        line-height: 1.4;
    }

    .top-line {
        margin: 13px 0 16px 0;
        border-top: 1px solid #e2e5e9;
    }

    .section-title {
        font-size: 15px;
        font-weight: 700;
        color: #263241;
        margin: 4px 0 8px 0;
    }

    .metric-card {
        background: white;
        border: 1px solid #e1e5ea;
        border-radius: 9px;
        padding: 13px 15px;
        min-height: 88px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }

    .metric-label {
        color: #687282;
        font-size: 12px;
        margin-bottom: 7px;
    }

    .metric-value {
        color: #1d4ed8;
        font-size: 25px;
        font-weight: 700;
        line-height: 1.1;
    }

    .metric-small {
        color: #7a8491;
        font-size: 10px;
        margin-top: 5px;
    }

    .panel {
        background: white;
        border: 1px solid #e1e5ea;
        border-radius: 9px;
        padding: 8px 10px 4px 10px;
        margin-top: 12px;
    }

    .info-card {
        background: white;
        border: 1px solid #e1e5ea;
        border-radius: 9px;
        padding: 14px 16px;
        min-height: 180px;
    }

    .info-title {
        font-size: 14px;
        font-weight: 700;
        color: #263241;
        margin-bottom: 10px;
    }

    .info-row {
        display: flex;
        justify-content: space-between;
        border-bottom: 1px solid #f0f1f3;
        padding: 7px 0;
        font-size: 12px;
    }

    .info-row:last-child {
        border-bottom: none;
    }

    .info-key {
        color: #737d8a;
    }

    .info-val {
        color: #273444;
        font-weight: 600;
    }

    .note {
        background: #ffffff;
        border: 1px solid #e1e5ea;
        border-radius: 8px;
        padding: 10px 13px;
        color: #697382;
        font-size: 11px;
        line-height: 1.5;
        margin-top: 12px;
    }

    .stSelectbox label {
        font-size: 12px !important;
        color: #657080 !important;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Load files
# -----------------------------
if not os.path.exists(FEATURE_FILE):
    st.error(f"Feature file not found: {FEATURE_FILE}")
    st.stop()

if not os.path.exists(MODEL_FILE):
    st.error(f"Model file not found: {MODEL_FILE}")
    st.stop()

df = pd.read_csv(FEATURE_FILE)
model_package = joblib.load(MODEL_FILE)

if isinstance(model_package, dict):
    model = model_package.get("model")
    feature_columns = model_package.get("feature_columns")
else:
    model = model_package
    feature_columns = None

if model is None:
    st.error("The saved model could not be loaded.")
    st.stop()

# Use the exact feature order saved with the model.
if feature_columns is None:
    feature_columns = [
        c for c in df.columns
        if c.endswith("_x") or c.endswith("_y") or c.endswith("_z")
    ]

feature_columns = [c for c in feature_columns if c in df.columns]

required_columns = {"subject", "upperbody_label"}
missing = required_columns - set(df.columns)

if missing:
    st.error(f"Missing required columns: {sorted(missing)}")
    st.stop()

# -----------------------------
# Prepare labels
# -----------------------------
df["Posture"] = df["upperbody_label"].astype(str)
df["Posture Name"] = df["Posture"].apply(
    lambda x: POSTURE_NAMES.get(x, x)
)

# -----------------------------
# Angle / CVA-based measure
# -----------------------------
if "cva_proxy_deg" in df.columns:
    df["Alignment Angle"] = pd.to_numeric(
        df["cva_proxy_deg"], errors="coerce"
    )
elif "cva_left_deg" in df.columns and "cva_right_deg" in df.columns:
    left = pd.to_numeric(df["cva_left_deg"], errors="coerce")
    right = pd.to_numeric(df["cva_right_deg"], errors="coerce")
    df["Alignment Angle"] = (left + right) / 2
elif "cva_deg" in df.columns:
    df["Alignment Angle"] = pd.to_numeric(
        df["cva_deg"], errors="coerce"
    )
else:
    df["Alignment Angle"] = np.nan

# -----------------------------
# Model prediction
# -----------------------------
# -----------------------------
# Model prediction
# -----------------------------
X = df[feature_columns].apply(pd.to_numeric, errors="coerce")

valid_rows = X.notna().all(axis=1)

# Create text/object arrays first.
# This avoids Pandas 3.x trying to put "TLB"/"TUP" into float64 columns.
predicted_posture = np.empty(len(df), dtype=object)
predicted_posture[:] = None

predicted_name = np.empty(len(df), dtype=object)
predicted_name[:] = None

if valid_rows.any():

    predictions = model.predict(X.loc[valid_rows])

    # Convert model output to ordinary Python strings
    predictions = [
        str(x) for x in predictions
    ]

    valid_indices = np.flatnonzero(valid_rows.to_numpy())

    for i, prediction in zip(valid_indices, predictions):
        predicted_posture[i] = prediction
        predicted_name[i] = POSTURE_NAMES.get(
            prediction,
            prediction
        )

# Assign complete object arrays at once.
# Do NOT use df.loc[...] = here.
df["Predicted Posture"] = predicted_posture
df["Predicted Name"] = predicted_name
# -----------------------------
# Participant selection
# -----------------------------
available_participants = sorted(
    pd.to_numeric(df["subject"], errors="coerce")
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)

test_available = [
    p for p in TEST_PARTICIPANTS if p in available_participants
]

participant_options = test_available if test_available else available_participants

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-title">Smart-Spine</div>
            <div class="brand-subtitle">
                Posture Analysis<br>
                Phase 1 Offline Dataset Dashboard
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="nav-item nav-active">Dashboard Overview</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="nav-item">Posture Analysis</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="nav-item">Head–Shoulder Alignment</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="nav-item">Model Evaluation</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="nav-item">Participant Summary</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="side-box">
            <div class="info-title">Dataset Info</div>

            <div class="side-label">Total Records</div>
            <div class="side-value">4,794</div>

            <div class="side-label">Participants</div>
            <div class="side-value">13</div>

            <div class="side-label">Posture Classes</div>
            <div class="side-value">5</div>

            <div class="side-label">Evaluation Split</div>
            <div class="side-value">Participants 10–13</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Header
# -----------------------------
header_left, header_right = st.columns([4, 1])

with header_left:
    st.markdown(
        '<div class="main-title">Dashboard Overview</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="main-subtitle">'
        'Phase 1 | Dataset-based posture classification and CVA-based alignment analysis'
        '</div>',
        unsafe_allow_html=True,
    )

with header_right:
    selected_participant = st.selectbox(
        "Participant",
        participant_options,
        index=0,
    )

st.markdown('<div class="top-line"></div>', unsafe_allow_html=True)

# -----------------------------
# Selected participant data
# -----------------------------
participant_df = df[
    pd.to_numeric(df["subject"], errors="coerce").astype("Int64")
    == selected_participant
].copy()

# -----------------------------
# Overall test-set data
# -----------------------------
test_df = df[
    pd.to_numeric(df["subject"], errors="coerce").isin(TEST_PARTICIPANTS)
].copy()

# Accuracy only on held-out participants
test_valid = test_df[
    test_df["Predicted Posture"].notna()
].copy()

if len(test_valid) > 0:
    test_accuracy = (
        test_valid["Posture"].astype(str)
        == test_valid["Predicted Posture"].astype(str)
    ).mean() * 100
else:
    test_accuracy = np.nan

# -----------------------------
# Top metric cards
# -----------------------------
dominant_posture = (
    participant_df["Posture Name"]
    .value_counts()
    .idxmax()
    if not participant_df.empty
    else "—"
)

avg_angle = participant_df["Alignment Angle"].mean()

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Selected Participant</div>
            <div class="metric-value">{selected_participant}</div>
            <div class="metric-small">Held-out test participant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Total Observations</div>
            <div class="metric-value">{len(participant_df):,}</div>
            <div class="metric-small">Records for selected participant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Most Common Posture</div>
            <div class="metric-value" style="font-size:20px;">
                {dominant_posture}
            </div>
            <div class="metric-small">Based on dataset labels</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m4:
    angle_text = f"{avg_angle:.1f}°" if pd.notna(avg_angle) else "—"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Average Alignment Angle</div>
            <div class="metric-value">{angle_text}</div>
            <div class="metric-small">CVA-based measure</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Row 1: three charts
# -----------------------------
c1, c2, c3 = st.columns(3)

# Actual posture distribution
with c1:
    st.markdown(
        '<div class="section-title">Posture Distribution</div>',
        unsafe_allow_html=True,
    )

    actual_counts = (
        participant_df["Posture Name"]
        .value_counts()
        .rename_axis("Posture")
        .reset_index(name="Count")
    )

    fig_actual = px.pie(
        actual_counts,
        names="Posture",
        values="Count",
        hole=0.48,
        color="Posture",
        color_discrete_map=POSTURE_COLORS,
    )

    fig_actual.update_traces(
        textposition="inside",
        textinfo="percent",
        hovertemplate="%{label}<br>Records: %{value}<br>%{percent}<extra></extra>",
    )

    fig_actual.update_layout(
        height=285,
        margin=dict(l=8, r=8, t=10, b=8),
        showlegend=True,
        legend=dict(font=dict(size=9)),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    st.plotly_chart(fig_actual, use_container_width=True)

# Predicted posture distribution
with c2:
    st.markdown(
        '<div class="section-title">Model-Predicted Posture</div>',
        unsafe_allow_html=True,
    )

    pred_counts = (
        participant_df["Predicted Name"]
        .dropna()
        .value_counts()
        .rename_axis("Posture")
        .reset_index(name="Count")
    )

    if not pred_counts.empty:
        fig_pred = px.pie(
            pred_counts,
            names="Posture",
            values="Count",
            hole=0.48,
            color="Posture",
            color_discrete_map=POSTURE_COLORS,
        )

        fig_pred.update_traces(
            textposition="inside",
            textinfo="percent",
            hovertemplate="%{label}<br>Records: %{value}<br>%{percent}<extra></extra>",
        )

        fig_pred.update_layout(
            height=285,
            margin=dict(l=8, r=8, t=10, b=8),
            showlegend=True,
            legend=dict(font=dict(size=9)),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

        st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.info("Prediction data is not available for this participant.")

# Alignment angle distribution
with c3:
    st.markdown(
        '<div class="section-title">CVA-Based Alignment</div>',
        unsafe_allow_html=True,
    )

    angle_data = participant_df["Alignment Angle"].dropna()

    if len(angle_data) > 0:
        fig_angle = px.histogram(
            angle_data,
            x="Alignment Angle",
            nbins=18,
            labels={"Alignment Angle": "Angle (degrees)"},
        )

        fig_angle.update_traces(
            hovertemplate="Angle: %{x:.1f}°<br>Frequency: %{y}<extra></extra>"
        )

        fig_angle.update_layout(
            height=285,
            margin=dict(l=40, r=10, t=10, b=40),
            xaxis_title="Angle (degrees)",
            yaxis_title="Frequency",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(size=10),
        )

        st.plotly_chart(fig_angle, use_container_width=True)
    else:
        st.info("Alignment-angle data is not available.")

# -----------------------------
# Row 2: model performance + participant info + posture summary
# -----------------------------
r1, r2, r3 = st.columns([1.15, 1, 1])

with r1:
    st.markdown(
        '<div class="section-title">Model Evaluation</div>',
        unsafe_allow_html=True,
    )

    if pd.notna(test_accuracy):
        acc_value = f"{test_accuracy:.2f}%"
    else:
        acc_value = "—"

    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-title">Held-Out Test Performance</div>
            <div class="info-row">
                <span class="info-key">Evaluation participants</span>
                <span class="info-val">10–13</span>
            </div>
            <div class="info-row">
                <span class="info-key">Test observations</span>
                <span class="info-val">{len(test_df):,}</span>
            </div>
            <div class="info-row">
                <span class="info-key">Posture classes</span>
                <span class="info-val">5</span>
            </div>
            <div class="info-row">
                <span class="info-key">Test accuracy</span>
                <span class="info-val">{acc_value}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with r2:
    st.markdown(
        '<div class="section-title">Participant Summary</div>',
        unsafe_allow_html=True,
    )

    if not participant_df.empty:
        posture_counts = (
            participant_df["Posture Name"]
            .value_counts()
            .to_dict()
        )

        rows = ""
        for posture in POSTURE_NAMES.values():
            count = posture_counts.get(posture, 0)
            rows += f"""
            <div class="info-row">
                <span class="info-key">{posture}</span>
                <span class="info-val">{count}</span>
            </div>
            """

        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-title">Posture Records</div>
                {rows}
            </div>
            """,
            unsafe_allow_html=True,
        )

with r3:
    st.markdown(
        '<div class="section-title">Analysis Overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">What this dashboard shows</div>

            <div class="info-row">
                <span class="info-key">Posture</span>
                <span class="info-val">5-class analysis</span>
            </div>

            <div class="info-row">
                <span class="info-key">Alignment</span>
                <span class="info-val">CVA-based angle</span>
            </div>

            <div class="info-row">
                <span class="info-key">Input source</span>
                <span class="info-val">Existing dataset</span>
            </div>

            <div class="info-row">
                <span class="info-key">Evaluation</span>
                <span class="info-val">Participants 10–13</span>
            </div>

            <div class="info-row">
                <span class="info-key">Body weight</span>
                <span class="info-val">Not used in Phase 1</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Overall test participant summary
# -----------------------------
st.markdown(
    '<div class="section-title" style="margin-top:18px;">Overall Test-Participant Summary</div>',
    unsafe_allow_html=True,
)

o1, o2 = st.columns([1, 2])

with o1:
    overall_counts = (
        test_df["Posture Name"]
        .value_counts()
        .rename_axis("Posture")
        .reset_index(name="Count")
    )

    fig_overall = px.pie(
        overall_counts,
        names="Posture",
        values="Count",
        hole=0.5,
        color="Posture",
        color_discrete_map=POSTURE_COLORS,
    )

    fig_overall.update_traces(
        textposition="inside",
        textinfo="percent",
    )

    fig_overall.update_layout(
        height=270,
        margin=dict(l=5, r=5, t=5, b=5),
        showlegend=True,
        legend=dict(font=dict(size=9)),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    st.plotly_chart(fig_overall, use_container_width=True)

with o2:
    if len(test_df) > 0:
        overall_angle = test_df["Alignment Angle"].mean()
        overall_min = test_df["Alignment Angle"].min()
        overall_max = test_df["Alignment Angle"].max()

        overall_angle_text = (
            f"{overall_angle:.1f}°"
            if pd.notna(overall_angle)
            else "—"
        )
        range_text = (
            f"{overall_min:.1f}° – {overall_max:.1f}°"
            if pd.notna(overall_min) and pd.notna(overall_max)
            else "—"
        )
    else:
        overall_angle_text = "—"
        range_text = "—"

    st.markdown(
        f"""
        <div class="info-card" style="min-height:270px;">
            <div class="info-title">Test-Set Overview</div>

            <div class="info-row">
                <span class="info-key">Participants</span>
                <span class="info-val">10, 11, 12, 13</span>
            </div>

            <div class="info-row">
                <span class="info-key">Total test observations</span>
                <span class="info-val">{len(test_df):,}</span>
            </div>

            <div class="info-row">
                <span class="info-key">Average alignment angle</span>
                <span class="info-val">{overall_angle_text}</span>
            </div>

            <div class="info-row">
                <span class="info-key">Observed angle range</span>
                <span class="info-val">{range_text}</span>
            </div>

            <div class="info-row">
                <span class="info-key">Evaluation approach</span>
                <span class="info-val">Participant-wise 9:4 split</span>
            </div>

            <div class="info-row">
                <span class="info-key">Phase 1 focus</span>
                <span class="info-val">Posture + alignment</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Footer note
# -----------------------------
st.markdown(
    """
    <div class="note">
        Phase 1 uses the existing dataset for posture classification and
        CVA-based head–shoulder alignment analysis. Body weight and
        cervical-load estimation are not included in this Phase 1 dashboard.
        Results are intended for research and prototype analysis, not medical diagnosis.
    </div>
    """,
    unsafe_allow_html=True,
)
