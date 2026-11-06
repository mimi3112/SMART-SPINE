"""
paths.py — Project root resolver and shared path constants.
Walks upward from this file's location until it finds the project root
(identified by models/posture_model.pkl + results/).
"""
from __future__ import annotations
from pathlib import Path


def find_project_root() -> Path:
    candidate = Path(__file__).resolve().parent
    while True:
        if (candidate / "models" / "posture_model.pkl").exists() and (candidate / "results").exists():
            return candidate
        parent = candidate.parent
        if parent == candidate:
            raise RuntimeError(
                "SMART-SPINE project root not found. "
                "Expected a directory containing 'models/posture_model.pkl' and 'results/'. "
                "Searched upward from: " + str(Path(__file__).resolve().parent)
            )
        candidate = parent


PROJECT_ROOT   = find_project_root()
FEATURE_FILE   = PROJECT_ROOT / "results" / "features.csv"
MODEL_FILE     = PROJECT_ROOT / "models"  / "posture_model.pkl"
LIVE_LOG       = PROJECT_ROOT / "results" / "live_analysis.csv"
LANDMARKER_TASK = PROJECT_ROOT / "pose_landmarker_full.task"
TRACKER_LOG    = PROJECT_ROOT / "results" / "tracker.log"

# Find landmark_detection.py by searching downward
def _find_live_script() -> Path:
    for p in PROJECT_ROOT.rglob("landmark_detection.py"):
        return p
    raise FileNotFoundError("landmark_detection.py not found under " + str(PROJECT_ROOT))

LIVE_SCRIPT = _find_live_script()
