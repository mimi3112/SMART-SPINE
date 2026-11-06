# SMART-SPINE — Posture Analysis System

**SMART-SPINE** is a research prototype that uses MediaPipe body-pose landmarks to classify seated posture, estimate head-neck alignment (CVA proxy), and compute a model-based cervical load estimate in real time.

> ⚠ **Non-clinical prototype.** Results are for research and demonstration only. This is not medical advice.

Developed By: Harshita Chaudhari(Vth Sem, RBU Nagpur)
              Meenakshi Motwani(Vth Sem, RBU Nagpur)
              Tanishka Mohod(Vth Sem, RBU Nagpur)

---

## How to Launch

### Option 1 — Desktop App (Recommended)
Double-click the **`SMART-SPINE`** shortcut on your Desktop, or run:
```powershell
python desktop_app.py
```
or double-click **`run_desktop.bat`**.

This opens a native OS application window powered by Edge WebView2 (no browser tab needed). Closing the window cleanly shuts down the application.

### Option 2 — Web / Browser Mode
```powershell
streamlit run app.py
```
or double-click **`run.bat`**.

---

## Features & Tabs

### 1. Phase 1: Offline Analytics
- Participant selector (held-out test participants 10–13)
- Metric cards: participant ID, observation count, dominant posture, average CVA angle
- Donut charts: actual vs model-predicted posture distribution
- CVA-based head-shoulder angle distribution histogram
- Held-out model test evaluation (~97% test accuracy)
- **Participant-Specific Summary**: 3–4 concise bullet points generated specifically for the selected participant based on their data.

### 2. Phase 2: Live Monitor
- Start Tracking / Stop Tracking / Capture Posture Now controls
- Body weight setting from sidebar (used for cervical load scaling)
- Real-time classification badge, CVA gauge, and estimated cervical load
- Recent session observations table (auto-refreshed every 3 seconds)
- **Session-Specific Summary**: When tracking stops, generates a 3–4 bullet plain-language summary analyzing **only the observations recorded in that session**.

### 3. Posture Trends
- Date range filter and load unit toggle (kg-eq / Newtons)
- Posture over time scatter, CVA line chart, cervical load chart
- Posture distribution donut and peak load per posture
- Download filtered data as CSV

---

## Privacy
- **No camera frames or video are ever saved to disk.**
- Only numeric rows are logged to `results/live_analysis.csv`.
- AI summaries send only minimal aggregated numbers (percentages and averages). No images, coordinates, personal identifiers, or raw landmarks are ever sent.

---

## API Configuration
If no key is present, the app automatically generates accurate plain-language summaries using pure Python logic.
