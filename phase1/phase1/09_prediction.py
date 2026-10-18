import pandas as pd
import numpy as np
import joblib

# ============================================================
# SMART-SPINE PREDICTION
# ============================================================

FEATURE_FILE = "results/features.csv"
MODEL_FILE = "models/posture_model.pkl"

# ------------------------------------------------------------
# POSTURE LABEL MEANINGS
# ------------------------------------------------------------

POSTURE_NAMES = {
    "TUP": "Upright",
    "TLF": "Leaning Forward",
    "TLB": "Leaning Backward",
    "TLL": "Leaning Left",
    "TLR": "Leaning Right"
}

# ------------------------------------------------------------
# BIOMECHANICAL CONSTANTS
# ------------------------------------------------------------

HEAD_MASS_FRACTION = 0.08
REFERENCE_HEAD_MASS_KG = 6.0

REFERENCE_ANGLES = np.array([
    0, 15, 30, 45, 60
])

REFERENCE_LOAD_LB = np.array([
    11, 27, 40, 49, 60
])

# ------------------------------------------------------------
# LOAD MODEL PACKAGE
# ------------------------------------------------------------

package = joblib.load(MODEL_FILE)

model = package["model"]
feature_columns = package["feature_columns"]

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(FEATURE_FILE)

# Select one example row
sample = df.iloc[[0]]

# ------------------------------------------------------------
# BODY WEIGHT INPUT
# ------------------------------------------------------------

print("=" * 60)
print("SMART-SPINE POSTURE + CERVICAL LOAD ANALYSIS")
print("=" * 60)

try:
    body_weight_kg = float(
        input(
            "\nEnter body weight (kg): "
        )
    )
except ValueError:
    print("Invalid weight.")
    exit()

if body_weight_kg <= 0:
    print("Body weight must be greater than zero.")
    exit()

# ------------------------------------------------------------
# POSTURE PREDICTION
# ------------------------------------------------------------

X_posture = sample[feature_columns]

predicted_label = model.predict(
    X_posture
)[0]

posture_name = POSTURE_NAMES.get(
    predicted_label,
    predicted_label
)

# ------------------------------------------------------------
# CVA PROXY
# ------------------------------------------------------------

cva_proxy = sample[
    "cva_proxy_deg"
].iloc[0]

# ------------------------------------------------------------
# FLEXION ESTIMATE
# ------------------------------------------------------------

estimated_flexion = (
    90.0 - cva_proxy
)

estimated_flexion = np.clip(
    estimated_flexion,
    0,
    60
)

# ------------------------------------------------------------
# BODY-WEIGHT-BASED LOAD
# ------------------------------------------------------------

estimated_head_mass = (
    body_weight_kg *
    HEAD_MASS_FRACTION
)

weight_scale = (
    estimated_head_mass /
    REFERENCE_HEAD_MASS_KG
)

reference_load_lb = np.interp(
    estimated_flexion,
    REFERENCE_ANGLES,
    REFERENCE_LOAD_LB
)

estimated_load_lb = (
    reference_load_lb *
    weight_scale
)

estimated_load_N = (
    estimated_load_lb *
    4.44822
)

# ------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("ANALYSIS RESULT")
print("=" * 60)

print(
    "\nBody weight:",
    round(body_weight_kg, 2),
    "kg"
)

print(
    "Predicted posture:",
    posture_name,
    f"({predicted_label})"
)

print(
    "CVA proxy:",
    round(cva_proxy, 2),
    "degrees"
)

print(
    "Estimated flexion:",
    round(estimated_flexion, 2),
    "degrees"
)

print(
    "Estimated cervical load:",
    round(estimated_load_N, 2),
    "N"
)

print("\nNote:")
print(
    "Cervical load is a model-based estimate, "
    "not a directly measured clinical force."
)

print(
    "CVA is a landmark-based proxy and is not "
    "a clinical diagnosis of forward head posture."
)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED")
print("=" * 60)