import pandas as pd
import numpy as np
import os

# ============================================================
# SMART-SPINE
# 04_biomechanics.py
#
# Phase 1:
# Project-level cervical load estimation
#
# IMPORTANT:
# - No body-weight assumption is used.
# - This is NOT a clinical force measurement.
# - It is a reference-based biomechanical estimate.
# ============================================================


# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

INPUT_FILE = "results/features.csv"
OUTPUT_FILE = "results/load_results.csv"

os.makedirs("results", exist_ok=True)


# ------------------------------------------------------------
# 2. REFERENCE POSTURE-LOAD DATA
# ------------------------------------------------------------
#
# These values represent a reference relationship between
# increasing forward neck flexion and cervical loading.
#
# They are used only as a project-level reference curve.
#
# They should NOT be described as measurements of the
# individual participant.
# ------------------------------------------------------------

REFERENCE_ANGLES_DEG = np.array([
    0,
    15,
    30,
    45,
    60
], dtype=float)

REFERENCE_LOAD_LB = np.array([
    11,
    27,
    40,
    49,
    60
], dtype=float)


# ------------------------------------------------------------
# 3. LOAD DATASET
# ------------------------------------------------------------

print("=" * 70)
print("SMART-SPINE - BIOMECHANICAL LOAD ESTIMATION")
print("=" * 70)

print("\nLoading feature dataset...")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)


# ------------------------------------------------------------
# 4. CHECK REQUIRED COLUMN
# ------------------------------------------------------------

if "cva_proxy_deg" not in df.columns:
    raise ValueError(
        "The dataset does not contain 'cva_proxy_deg'. "
        "Run 03_cva_features.py first."
    )


# ------------------------------------------------------------
# 5. CLEAN ANGLE DATA
# ------------------------------------------------------------

df["cva_proxy_deg"] = pd.to_numeric(
    df["cva_proxy_deg"],
    errors="coerce"
)

missing_angles = df["cva_proxy_deg"].isna().sum()

print(
    "Missing angle values:",
    missing_angles
)

# Remove records where the angle could not be calculated
df = df.dropna(
    subset=["cva_proxy_deg"]
).reset_index(drop=True)


# ------------------------------------------------------------
# 6. LIMIT UNREALISTIC VALUES
# ------------------------------------------------------------
#
# This is a project-level numerical safeguard.
#
# We do not allow the estimated flexion proxy to exceed
# the reference curve range of 0–60 degrees.
# ------------------------------------------------------------

# Your feature is an angle measured from the landmark geometry.
#
# For this project we use:
#
#     flexion_proxy = 90 - angle
#
# and then limit it to 0–60 degrees.
#
# This is a PROJECT PROXY, not a clinical neck-flexion angle.

df["estimated_flexion_deg"] = (
    90.0 - df["cva_proxy_deg"]
)

df["estimated_flexion_deg"] = np.clip(
    df["estimated_flexion_deg"],
    0,
    60
)


# ------------------------------------------------------------
# 7. INTERPOLATE REFERENCE CERVICAL LOAD
# ------------------------------------------------------------
#
# np.interp finds the estimated load between the known
# reference angle/load points.
#
# Example:
#
# 15 degrees  -> 27 lb
# 30 degrees  -> 40 lb
#
# If the calculated angle is 22 degrees, np.interp
# estimates a value between those two reference points.
# ------------------------------------------------------------

df["estimated_cervical_load_lb"] = np.interp(
    df["estimated_flexion_deg"],
    REFERENCE_ANGLES_DEG,
    REFERENCE_LOAD_LB
)


# ------------------------------------------------------------
# 8. CONVERT POUNDS-FORCE TO NEWTONS
# ------------------------------------------------------------

POUND_FORCE_TO_NEWTON = 4.44822

df["estimated_cervical_load_N"] = (
    df["estimated_cervical_load_lb"]
    *
    POUND_FORCE_TO_NEWTON
)


# ------------------------------------------------------------
# 9. KG-EQUIVALENT
# ------------------------------------------------------------
#
# This is NOT body weight.
#
# It simply expresses the force in an equivalent
# gravitational-force unit:
#
# kg-equivalent = force / gravitational acceleration
#
# This does NOT mean the person's head weighs this much.
# ------------------------------------------------------------

GRAVITY = 9.80665

df["estimated_cervical_load_kg_equivalent"] = (
    df["estimated_cervical_load_N"]
    /
    GRAVITY
)


# ------------------------------------------------------------
# 10. PRINT OVERALL RESULTS
# ------------------------------------------------------------

print("\nBiomechanical estimation completed.")

print("\nEstimated cervical load statistics:")

print(
    df["estimated_cervical_load_N"].describe()
)


# ------------------------------------------------------------
# 11. POSTURE-WISE ANALYSIS
# ------------------------------------------------------------

if "upperbody_label" in df.columns:

    print("\nPosture-wise cervical load:")

    posture_summary = (
        df
        .groupby("upperbody_label")
        ["estimated_cervical_load_N"]
        .agg(
            count="count",
            mean="mean",
            minimum="min",
            maximum="max",
            std="std"
        )
        .reset_index()
    )

    print(
        posture_summary.to_string(
            index=False
        )
    )

else:

    posture_summary = pd.DataFrame()


# ------------------------------------------------------------
# 12. PARTICIPANT-WISE ANALYSIS
# ------------------------------------------------------------

if "subject" in df.columns:

    print("\nParticipant-wise cervical load:")

    participant_summary = (
        df
        .groupby("subject")
        ["estimated_cervical_load_N"]
        .agg(
            count="count",
            mean="mean",
            minimum="min",
            maximum="max",
            std="std"
        )
        .reset_index()
    )

    print(
        participant_summary.to_string(
            index=False
        )
    )

else:

    participant_summary = pd.DataFrame()


# ------------------------------------------------------------
# 13. SAVE RECORD-LEVEL RESULTS
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nDetailed results saved to: {OUTPUT_FILE}"
)


# ------------------------------------------------------------
# 14. SAVE POSTURE SUMMARY
# ------------------------------------------------------------

POSTURE_SUMMARY_FILE = (
    "results/posture_load_summary.csv"
)

if not posture_summary.empty:

    posture_summary.to_csv(
        POSTURE_SUMMARY_FILE,
        index=False
    )

    print(
        "Posture summary saved to:",
        POSTURE_SUMMARY_FILE
    )


# ------------------------------------------------------------
# 15. SAVE PARTICIPANT SUMMARY
# ------------------------------------------------------------

PARTICIPANT_SUMMARY_FILE = (
    "results/participant_load_summary.csv"
)

if not participant_summary.empty:

    participant_summary.to_csv(
        PARTICIPANT_SUMMARY_FILE,
        index=False
    )

    print(
        "Participant summary saved to:",
        PARTICIPANT_SUMMARY_FILE
    )


# ------------------------------------------------------------
# 16. FINAL MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BIOMECHANICAL ANALYSIS COMPLETED")
print("=" * 70)

print(
    "\nIMPORTANT:"
)

print(
    "The cervical load values are project-level "
    "reference-based estimates."
)

print(
    "They are not direct measurements of cervical "
    "spinal force and should not be presented as "
    "clinical measurements."
)

print(
    "\nNo body-weight value was assumed or used."
)

print("=" * 70)