import pandas as pd
import numpy as np
import os

INPUT_FILE = "results/cleaned_data.csv"
OUTPUT_FILE = "results/features.csv"

os.makedirs("results", exist_ok=True)

print("=" * 60)
print("SMART-SPINE POSTURE FEATURE EXTRACTION")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("\nDataset shape:", df.shape)

# ------------------------------------------------------------
# LEFT EAR - LEFT SHOULDER
# ------------------------------------------------------------

dx_left = (
    df["left_ear_x"] -
    df["left_shoulder_x"]
)

dy_left = (
    df["left_ear_y"] -
    df["left_shoulder_y"]
)

df["cva_left_deg"] = np.degrees(
    np.arctan2(
        np.abs(dy_left),
        np.abs(dx_left)
    )
)

# ------------------------------------------------------------
# RIGHT EAR - RIGHT SHOULDER
# ------------------------------------------------------------

dx_right = (
    df["right_ear_x"] -
    df["right_shoulder_x"]
)

dy_right = (
    df["right_ear_y"] -
    df["right_shoulder_y"]
)

df["cva_right_deg"] = np.degrees(
    np.arctan2(
        np.abs(dy_right),
        np.abs(dx_right)
    )
)

# ------------------------------------------------------------
# AVERAGE CVA PROXY
# ------------------------------------------------------------

df["cva_proxy_deg"] = (
    df["cva_left_deg"] +
    df["cva_right_deg"]
) / 2

# ------------------------------------------------------------
# SHOULDER WIDTH
# ------------------------------------------------------------

df["shoulder_width"] = np.sqrt(
    (
        df["left_shoulder_x"] -
        df["right_shoulder_x"]
    ) ** 2
    +
    (
        df["left_shoulder_y"] -
        df["right_shoulder_y"]
    ) ** 2
)

# ------------------------------------------------------------
# EAR-SHOULDER DISTANCE
# ------------------------------------------------------------

left_distance = np.sqrt(
    (
        df["left_ear_x"] -
        df["left_shoulder_x"]
    ) ** 2
    +
    (
        df["left_ear_y"] -
        df["left_shoulder_y"]
    ) ** 2
)

right_distance = np.sqrt(
    (
        df["right_ear_x"] -
        df["right_shoulder_x"]
    ) ** 2
    +
    (
        df["right_ear_y"] -
        df["right_shoulder_y"]
    ) ** 2
)

df["ear_shoulder_distance_avg"] = (
    left_distance + right_distance
) / 2

# ------------------------------------------------------------
# HEAD FORWARD DISPLACEMENT
# ------------------------------------------------------------

ear_x = (
    df["left_ear_x"] +
    df["right_ear_x"]
) / 2

shoulder_x = (
    df["left_shoulder_x"] +
    df["right_shoulder_x"]
) / 2

df["head_forward_displacement"] = (
    ear_x - shoulder_x
)

# ------------------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------------------

print("\n--- CVA PROXY STATISTICS ---")
print(df["cva_proxy_deg"].describe())

print("\n--- CVA PROXY BY POSTURE ---")

print(
    df.groupby("upperbody_label")[
        "cva_proxy_deg"
    ].agg(
        ["count", "mean", "std", "min", "max"]
    )
)

# Save
df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nFeatures saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 60)
print("FEATURE EXTRACTION COMPLETED")
print("=" * 60)