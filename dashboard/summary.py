"""
dashboard/summary.py

LLM-powered plain-language summary panel.

Two public functions:
  render_phase1_summary(participant_df, participant_id, test_accuracy)
      → 3-4 bullets about the selected participant's offline dataset results.

  render_session_summary(session_df, session_start, session_end)
      → 3-4 bullets about the current live tracking session ONLY.
         session_df must already be filtered to rows within the session window.
         If empty, shows a no-data message — does NOT fall back to historical data.

Privacy: only aggregated numbers (counts, %, averages) are sent to the LLM.
No timestamps, body weights, names, coordinates, images, or raw landmarks.

Caching: keyed on a SHA-256 fingerprint of the stats dict.
The LLM is called at most once per unique set of findings.
"""
from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Load .env if present ───────────────────────────────────────────────────────
def _get_api_key() -> str:
    """Retrieve GEMINI_API_KEY dynamically from environment or .env file."""
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)
    except ImportError:
        pass
    return os.environ.get("GEMINI_API_KEY", "").strip()


# ── Posture names (local copy — avoids circular import) ───────────────────────
_POSTURE_NAMES = {
    "TUP": "Upright", "TLF": "Leaning Forward", "TLB": "Leaning Backward",
    "TLL": "Leaning Left", "TLR": "Leaning Right",
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fingerprint(obj: dict) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _call_llm(prompt: str) -> str:
    """Call Gemini active flash model with fallback. Raises on failure."""
    key = _get_api_key()
    if not key or key == "paste_your_key_here":
        raise ValueError("Missing or default GEMINI_API_KEY in .env")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import google.generativeai as genai

    genai.configure(api_key=key)

    # Use active Gemini models that support generateContent
    models_to_try = [
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-flash-latest",
    ]
    last_err = None
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt, request_options={"timeout": 15})
            text = response.text.strip()
            if text:
                return text
        except Exception as e:
            last_err = e
            continue

    raise last_err or RuntimeError("No Gemini models responded successfully")


def _summary_card(title: str, bullets: str, source_note: str, data_note: str) -> None:
    """Render the styled summary card. Identical look for both Phase 1 and Phase 2."""
    st.markdown(
        f"""
        <div style="
            background:#ffffff;border:1px solid #e1e5ea;
            border-left:5px solid #2563eb;border-radius:10px;
            padding:18px 22px 14px 22px;margin-bottom:20px;
            box-shadow:0 2px 6px rgba(0,0,0,0.05);">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <div style="font-size:17px;font-weight:700;color:#18212f;">
                📋 {title}
            </div>
            <div style="background:#eff6ff;color:#1d4ed8;font-size:12px;font-weight:600;padding:4px 12px;border-radius:999px;border:1px solid #bfdbfe;">
                {source_note}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for line in bullets.split("\n"):
        line = line.strip()
        if line:
            st.markdown(line)
    st.markdown(
        f"<div style='font-size:11px;color:#9ca3af;margin-top:10px;border-top:1px solid #f3f4f6;padding-top:6px;'>"
        f"{data_note} &nbsp;·&nbsp; <em>Non-clinical research prototype</em></div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Participant-specific offline summary
# ──────────────────────────────────────────────────────────────────────────────

def _phase1_stats(participant_df: pd.DataFrame, participant_id: int,
                  test_accuracy: float | None) -> dict:
    """Compute stats for a single participant from features.csv."""
    if participant_df is None or participant_df.empty:
        return {"has_data": False, "participant_id": participant_id}

    stats: dict = {"has_data": True, "participant_id": participant_id}
    stats["n_observations"] = int(len(participant_df))

    # Actual posture distribution
    if "Posture Name" in participant_df.columns:
        pct = participant_df["Posture Name"].value_counts(normalize=True) * 100
        stats["actual_pct"] = {k: round(float(v), 1) for k, v in pct.items()}
        stats["dominant_actual"] = pct.index[0] if len(pct) else "Unknown"
    elif "upperbody_label" in participant_df.columns:
        pct = participant_df["upperbody_label"].map(_POSTURE_NAMES).fillna(
            participant_df["upperbody_label"]).value_counts(normalize=True) * 100
        stats["actual_pct"] = {k: round(float(v), 1) for k, v in pct.items()}
        stats["dominant_actual"] = pct.index[0] if len(pct) else "Unknown"

    # Predicted posture distribution
    if "Predicted Name" in participant_df.columns:
        pred_valid = participant_df["Predicted Name"].dropna()
        if not pred_valid.empty:
            ppct = pred_valid.value_counts(normalize=True) * 100
            stats["predicted_pct"] = {k: round(float(v), 1) for k, v in ppct.items()}
            stats["dominant_predicted"] = ppct.index[0]

    # Alignment angle
    if "Alignment Angle" in participant_df.columns:
        ang = participant_df["Alignment Angle"].dropna()
        if not ang.empty:
            stats["avg_alignment_angle"] = round(float(ang.mean()), 1)
            stats["min_alignment_angle"] = round(float(ang.min()), 1)
            stats["max_alignment_angle"] = round(float(ang.max()), 1)

    # Per-participant accuracy
    if "Posture" in participant_df.columns and "Predicted Posture" in participant_df.columns:
        valid = participant_df.dropna(subset=["Predicted Posture"])
        if not valid.empty:
            acc = (valid["Posture"].astype(str) == valid["Predicted Posture"].astype(str)).mean() * 100
            stats["participant_accuracy"] = round(float(acc), 1)

    # Overall test accuracy passed in
    if test_accuracy is not None and not pd.isna(test_accuracy):
        stats["overall_test_accuracy"] = round(float(test_accuracy), 1)

    return stats


def _phase1_prompt(stats: dict) -> str:
    s = stats
    lines = [
        "You are a friendly, plain-language posture assistant inside a research prototype app.",
        "Write exactly 3 to 4 bullet points summarising a participant's recorded posture data.",
        "",
        "Hard rules:",
        "- Use everyday language. No medical jargon (no 'CVA', no 'cervical strain', no 'classifier').",
        "- Do NOT invent any numbers not listed below.",
        "- Do NOT diagnose any medical condition.",
        "- Each bullet is one short sentence.",
        "- Start every bullet with the character •",
        "",
        f"Participant: {s['participant_id']}  |  Total recordings: {s['n_observations']}",
        "",
        "Actual posture breakdown:",
    ]
    for posture, pct in s.get("actual_pct", {}).items():
        lines.append(f"  {posture}: {pct}%")

    if "predicted_pct" in s:
        lines.append("\nModel-predicted posture breakdown:")
        for posture, pct in s["predicted_pct"].items():
            lines.append(f"  {posture}: {pct}%")

    if "avg_alignment_angle" in s:
        lines.append(
            f"\nAverage head-shoulder angle: {s['avg_alignment_angle']}° "
            f"(range: {s['min_alignment_angle']}° to {s['max_alignment_angle']}°). "
            "Higher angle = more upright; lower angle = more head-forward lean."
        )
    if "participant_accuracy" in s:
        lines.append(f"Model accuracy for this participant: {s['participant_accuracy']}%")
    if "overall_test_accuracy" in s:
        lines.append(f"Overall model accuracy on test group: {s['overall_test_accuracy']}%")

    lines += ["", "Write the 3-4 bullets now:"]
    return "\n".join(lines)


def _phase1_fallback(stats: dict) -> str:
    if not stats.get("has_data"):
        return (
            f"• No data found for participant {stats.get('participant_id', '?')}.\n"
            "• Select a participant from the dropdown to see their summary."
        )
    s = stats
    items = []

    # 1 — overall picture
    dom = s.get("dominant_actual", "Unknown")
    items.append(
        f"• Participant {s['participant_id']} has **{s['n_observations']:,} recorded frames**; "
        f"the most common position observed was **{dom}**."
    )

    # 2 — distribution detail
    non_up = {k: v for k, v in s.get("actual_pct", {}).items() if k != "Upright"}
    if non_up:
        worst = max(non_up, key=lambda k: non_up[k])
        items.append(
            f"• Among non-upright positions, **{worst}** appeared most often "
            f"({non_up[worst]:.1f}% of all frames)."
        )

    # 3 — alignment angle
    if "avg_alignment_angle" in s:
        items.append(
            f"• The average head-shoulder angle was **{s['avg_alignment_angle']}°** "
            f"(range {s['min_alignment_angle']}° to {s['max_alignment_angle']}°; "
            "a higher angle means a more upright head position)."
        )

    # 4 — model accuracy
    if "participant_accuracy" in s:
        items.append(
            f"• The posture model correctly identified this participant's position "
            f"**{s['participant_accuracy']:.1f}%** of the time."
        )
    elif "overall_test_accuracy" in s:
        items.append(
            f"• Across all test participants, the posture model achieved "
            f"**{s['overall_test_accuracy']:.1f}%** accuracy."
        )

    return "\n".join(items[:4])


def render_phase1_summary(
    participant_df: pd.DataFrame,
    participant_id: int,
    test_accuracy: float | None = None,
) -> None:
    """
    Render the participant-specific summary card for Phase 1.
    Placed below the existing charts in phase1_tab.py.
    """
    stats = _phase1_stats(participant_df, participant_id, test_accuracy)
    fp    = f"p1_{participant_id}_{_fingerprint(stats)}"
    cache: dict = st.session_state.setdefault("_summary_cache", {})
    source_cache: dict = st.session_state.setdefault("_summary_source_cache", {})

    api_key = _get_api_key()
    has_key = bool(api_key and api_key != "paste_your_key_here")

    if fp not in cache:
        if has_key:
            with st.spinner("Generating AI posture summary with Gemini..."):
                try:
                    cache[fp] = _call_llm(_phase1_prompt(stats))
                    source_cache[fp] = "✨ AI Powered (Gemini)"
                except Exception as exc:
                    print(f"[SMART-SPINE] Gemini Phase 1 error: {exc}")
                    cache[fp] = _phase1_fallback(stats)
                    source_cache[fp] = "📊 Calculated Summary"
        else:
            cache[fp] = _phase1_fallback(stats)
            source_cache[fp] = "📊 Calculated Summary"

    source_note = source_cache.get(fp, "✨ AI Powered (Gemini)" if has_key else "📊 Calculated Summary")
    data_note   = (f"Participant {participant_id} · {stats.get('n_observations', 0):,} frames"
                   if stats.get("has_data") else "No data")

    _summary_card(
        title       = f"Your Posture at a Glance — Participant {participant_id}",
        bullets     = cache[fp],
        source_note = source_note,
        data_note   = data_note,
    )


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Session-specific live summary
# ──────────────────────────────────────────────────────────────────────────────

def _session_stats(session_df: pd.DataFrame) -> dict:
    """
    Compute stats from the session-filtered DataFrame only.
    session_df must contain ONLY rows from the current session.
    An empty or None df → has_data: False.
    """
    if session_df is None or session_df.empty:
        return {"has_data": False}

    stats: dict = {"has_data": True}
    stats["n_samples"] = int(len(session_df))

    # Posture distribution
    col = "posture_name" if "posture_name" in session_df.columns else "posture"
    pct = session_df[col].value_counts(normalize=True) * 100
    stats["posture_pct"] = {k: round(float(v), 1) for k, v in pct.items()}
    stats["dominant_posture"] = pct.index[0] if len(pct) else "Unknown"

    # % upright
    if "posture" in session_df.columns:
        stats["pct_upright"] = round(
            float((session_df["posture"] == "TUP").mean() * 100), 1
        )

    # Head lean angle
    if "estimated_flexion_deg" in session_df.columns:
        flex = session_df["estimated_flexion_deg"].dropna()
        if not flex.empty:
            stats["avg_flexion_deg"] = round(float(flex.mean()), 1)
            stats["max_flexion_deg"] = round(float(flex.max()),  1)

    # Trend: first half vs second half upright %
    mid = len(session_df) // 2
    if mid >= 2 and "posture" in session_df.columns:
        first  = (session_df["posture"].iloc[:mid] == "TUP").mean() * 100
        second = (session_df["posture"].iloc[mid:] == "TUP").mean() * 100
        diff   = second - first
        stats["trend"] = "improving" if diff > 10 else ("worsening" if diff < -10 else "stable")
        stats["trend_diff_pct"] = round(float(diff), 1)

    # Longest consecutive non-upright streak
    streak = max_streak = 0
    for p in session_df.get("posture", pd.Series()):
        if str(p) != "TUP":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    stats["longest_non_upright_streak"] = max_streak

    return stats


def _session_prompt(stats: dict) -> str:
    s = stats
    lines = [
        "You are a friendly, plain-language posture coach inside a research prototype app.",
        "Write exactly 3 to 4 bullet points summarising ONE live posture tracking session.",
        "",
        "Hard rules:",
        "- Use everyday language. No medical jargon.",
        "- Do NOT invent any numbers not listed below.",
        "- Do NOT diagnose any medical condition.",
        "- Do NOT claim the app measures actual spine or bone force.",
        "- If fewer than 3 samples, say data is very limited.",
        "- Each bullet is one short sentence.",
        "- Start every bullet with the character •",
        "",
        "Session findings (this session ONLY — not historical data):",
        f"  Samples recorded this session: {s['n_samples']}",
        f"  Time sitting upright: {s.get('pct_upright', 'unknown')}%",
    ]
    for posture, pct in s.get("posture_pct", {}).items():
        lines.append(f"  {posture}: {pct}%")
    if "avg_flexion_deg" in s:
        lines.append(
            f"  Average head forward lean this session: {s['avg_flexion_deg']}° "
            "(0 = fully upright, higher = more forward lean)"
        )
    if "trend" in s:
        lines.append(
            f"  Posture trend during session: {s['trend']} "
            f"(upright % changed {s['trend_diff_pct']:+.1f}% from start to end)"
        )
    if "longest_non_upright_streak" in s:
        lines.append(
            f"  Longest run of non-upright posture in a row: "
            f"{s['longest_non_upright_streak']} observations"
        )
    if s["n_samples"] < 3:
        lines.append("  Note: very few samples — treat findings with caution.")
    lines += ["", "Write the 3-4 bullets now:"]
    return "\n".join(lines)


def _session_fallback(stats: dict) -> str:
    s = stats
    items = []

    pct_up = s.get("pct_upright", None)
    items.append(
        f"• This session recorded **{s['n_samples']} observation(s)**"
        + (f"; you were sitting upright **{pct_up}%** of the time." if pct_up is not None else ".")
    )

    non_up = {k: v for k, v in s.get("posture_pct", {}).items() if k != "Upright"}
    if non_up:
        worst = max(non_up, key=lambda k: non_up[k])
        items.append(
            f"• The most common non-upright position this session was **{worst}** "
            f"({non_up[worst]:.1f}% of observations)."
        )

    if "trend" in s:
        label = {"improving": "better", "worsening": "worse", "stable": "about the same"}.get(s["trend"], s["trend"])
        items.append(
            f"• Your posture trend went **{label}** over the course of this session "
            f"(upright time changed by {s['trend_diff_pct']:+.1f}% from start to end)."
        )
    elif "avg_flexion_deg" in s:
        items.append(
            f"• On average your head leaned forward by **{s['avg_flexion_deg']}°** "
            "during this session (0° = fully upright)."
        )

    if s["n_samples"] < 3:
        items.append(
            "• Only a small number of observations were collected — "
            "start a longer session for a more reliable picture."
        )
    elif s.get("longest_non_upright_streak", 0) >= 4:
        items.append(
            f"• There was a run of **{s['longest_non_upright_streak']} observations in a row** "
            "without upright posture — consider a short stretch break if this happens often."
        )

    return "\n".join(items[:4])


def render_session_summary(
    session_df: pd.DataFrame,
    session_label: str = "Last Session",
) -> None:
    """
    Render the session-specific summary card for Phase 2.
    session_df must ALREADY be filtered to the current session's rows only.
    If empty, shows a clear no-data message — does NOT use historical data.
    """
    if session_df is None or session_df.empty:
        _summary_card(
            title       = f"Your Posture at a Glance — {session_label}",
            bullets     = (
                "• No valid posture observations were recorded in this session.\n"
                "• Make sure the camera detected a pose during the tracking period.\n"
                "• Try starting a new session and staying visible to the camera."
            ),
            source_note = "No data",
            data_note   = session_label,
        )
        return

    stats = _session_stats(session_df)
    fp    = f"p2_{session_label}_{_fingerprint(stats)}"
    cache: dict = st.session_state.setdefault("_summary_cache", {})
    source_cache: dict = st.session_state.setdefault("_summary_source_cache", {})

    api_key = _get_api_key()
    has_key = bool(api_key and api_key != "paste_your_key_here")

    if fp not in cache:
        if has_key:
            with st.spinner("Generating AI session summary with Gemini..."):
                try:
                    cache[fp] = _call_llm(_session_prompt(stats))
                    source_cache[fp] = "✨ AI Powered (Gemini)"
                except Exception as exc:
                    print(f"[SMART-SPINE] Gemini Session error: {exc}")
                    cache[fp] = _session_fallback(stats)
                    source_cache[fp] = "📊 Calculated Summary"
        else:
            cache[fp] = _session_fallback(stats)
            source_cache[fp] = "📊 Calculated Summary"

    source_note = source_cache.get(fp, "✨ AI Powered (Gemini)" if has_key else "📊 Calculated Summary")
    data_note   = f"{stats['n_samples']} samples · {session_label}"

    _summary_card(
        title       = f"Your Posture at a Glance — {session_label}",
        bullets     = cache[fp],
        source_note = source_note,
        data_note   = data_note,
    )
