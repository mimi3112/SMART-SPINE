import pandas as pd
import numpy as np

DATA_PATH = "data\\data.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nSubjects:")
print(sorted(df["subject"].unique()))

# ------------------------------------------
# Select only relevant columns
# ------------------------------------------

required_columns = [
    "subject",

    "upperbody_label",
    "lowerbody_label",

    "left_ear_x",
    "left_ear_y",
    "left_ear_z",

    "right_ear_x",
    "right_ear_y",
    "right_ear_z",

    "left_shoulder_x",
    "left_shoulder_y",
    "left_shoulder_z",

    "right_shoulder_x",
    "right_shoulder_y",
    "right_shoulder_z"
]

data = df[required_columns].copy()

print("\nSelected data shape:", data.shape)
print("\nSelected columns:")
print(data.columns.tolist())

print("\nMissing values:")
print(data.isnull().sum())

# ==========================================
# STEP 2: CALCULATE EAR AND SHOULDER MIDPOINTS
# ==========================================

data["ear_mid_x"] = (
    data["left_ear_x"] +
    data["right_ear_x"]
) / 2

data["ear_mid_y"] = (
    data["left_ear_y"] +
    data["right_ear_y"]
) / 2

data["ear_mid_z"] = (
    data["left_ear_z"] +
    data["right_ear_z"]
) / 2


data["shoulder_mid_x"] = (
    data["left_shoulder_x"] +
    data["right_shoulder_x"]
) / 2

data["shoulder_mid_y"] = (
    data["left_shoulder_y"] +
    data["right_shoulder_y"]
) / 2

data["shoulder_mid_z"] = (
    data["left_shoulder_z"] +
    data["right_shoulder_z"]
) / 2


print("\nMidpoint calculation completed.")

print(
    data[
        [
            "ear_mid_x",
            "ear_mid_y",
            "ear_mid_z",
            "shoulder_mid_x",
            "shoulder_mid_y",
            "shoulder_mid_z"
        ]
    ].head()
)

# ==========================================
# STEP 3: CALCULATE POSTURE ANGLE
# ==========================================

dx = (
    data["ear_mid_x"] -
    data["shoulder_mid_x"]
)

dy = (
    data["ear_mid_y"] -
    data["shoulder_mid_y"]
)

# Avoid division by zero
dx = dx.replace(0, np.nan)

data["posture_angle"] = np.degrees(
    np.arctan2(
        np.abs(dy),
        np.abs(dx)
    )
)

# Remove invalid values
data["posture_angle"] = data["posture_angle"].fillna(
    data["posture_angle"].median()
)

print("\nPosture angle statistics:")
print(data["posture_angle"].describe())