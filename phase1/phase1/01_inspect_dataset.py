import pandas as pd

INPUT_FILE = "data/data.csv"

print("=" * 60)
print("SMART-SPINE DATASET INSPECTION")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully.")

print("\n--- BASIC INFORMATION ---")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n--- COLUMN NAMES ---")
for i, column in enumerate(df.columns, start=1):
    print(i, ":", column)

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print("Total missing values:", df.isnull().sum().sum())

missing = df.isnull().sum()
print("\nColumns containing missing values:")
print(missing[missing > 0])

print("\n--- DUPLICATE ROWS ---")
print("Duplicate rows:", df.duplicated().sum())

print("\n--- PARTICIPANTS ---")
print("Number of participants:", df["subject"].nunique())
print("Participant IDs:", sorted(df["subject"].unique()))

print("\nRecords per participant:")
print(df["subject"].value_counts().sort_index())

print("\n--- UPPER-BODY POSTURES ---")
print(df["upperbody_label"].value_counts())

print("\nNumber of upper-body classes:",
      df["upperbody_label"].nunique())

print("\n--- LOWER-BODY POSTURES ---")
print(df["lowerbody_label"].value_counts())

print("\nNumber of lower-body classes:",
      df["lowerbody_label"].nunique())

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- NUMERICAL SUMMARY ---")
print(df.describe())

print("\n" + "=" * 60)
print("DATASET INSPECTION COMPLETED")
print("=" * 60)