import pandas as pd
import os

INPUT_FILE = "data/data.csv"
OUTPUT_FILE = "results/cleaned_data.csv"

os.makedirs("results", exist_ok=True)

print("=" * 60)
print("SMART-SPINE DATA CLEANING")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("\nOriginal shape:", df.shape)

# Check missing values
missing = df.isnull().sum().sum()
print("Total missing values:", missing)

# Check duplicates
duplicates = df.duplicated().sum()
print("Duplicate rows:", duplicates)

# Remove duplicates
df = df.drop_duplicates().reset_index(drop=True)

print("\nCleaned shape:", df.shape)

# Save
df.to_csv(OUTPUT_FILE, index=False)

print("\nCleaned dataset saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETED")
print("=" * 60)