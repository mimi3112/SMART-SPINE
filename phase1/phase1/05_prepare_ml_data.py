import pandas as pd
import os

INPUT_FILE = "results/features.csv"
OUTPUT_FILE = "results/ml_data.csv"

print("=" * 60)
print("SMART-SPINE ML DATA PREPARATION")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

# ------------------------------------------------------------
# FEATURES FOR POSTURE CLASSIFICATION
# ------------------------------------------------------------

feature_columns = [
    col
    for col in df.columns
    if col.endswith("_x")
    or col.endswith("_y")
    or col.endswith("_z")
]

target_column = "upperbody_label"

required_columns = [
    "subject",
    target_column
] + feature_columns

ml_data = df[required_columns].copy()

print("\nInput shape:", df.shape)
print("ML dataset shape:", ml_data.shape)

print("\nNumber of landmark features:",
      len(feature_columns))

print("\nTarget:")
print(target_column)

print("\nPosture classes:")
print(
    ml_data[target_column].value_counts()
)

print("\nMissing values:",
      ml_data.isnull().sum().sum())

os.makedirs("results", exist_ok=True)

ml_data.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nML dataset saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 60)
print("ML DATA PREPARATION COMPLETED")
print("=" * 60)