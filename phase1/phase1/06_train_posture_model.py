import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

INPUT_FILE = "results/cleaned_data.csv"
MODEL_FILE = "models/posture_model.pkl"
PREDICTION_FILE = "results/posture_test_predictions.csv"

print("=" * 60)
print("SMART-SPINE POSTURE CLASSIFIER")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("\nDataset shape:", df.shape)

# ------------------------------------------------------------
# PARTICIPANT-WISE SPLIT
# ------------------------------------------------------------

all_subjects = sorted(
    df["subject"].unique()
)

train_subjects = all_subjects[:9]
test_subjects = all_subjects[9:]

print("\nTraining participants:")
print(train_subjects)

print("\nTesting participants:")
print(test_subjects)

train_df = df[
    df["subject"].isin(train_subjects)
].copy()

test_df = df[
    df["subject"].isin(test_subjects)
].copy()

print("\nTraining samples:", len(train_df))
print("Testing samples:", len(test_df))

# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------

exclude_columns = [
    "subject",
    "upperbody_label",
    "lowerbody_label"
]

feature_columns = [
    col
    for col in df.columns
    if col not in exclude_columns
]

X_train = train_df[feature_columns]
y_train = train_df["upperbody_label"]

X_test = test_df[feature_columns]
y_test = test_df["upperbody_label"]

print("\nNumber of features:",
      len(feature_columns))

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")

# ------------------------------------------------------------
# TEST
# ------------------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    "\nTest Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

labels = sorted(
    df["upperbody_label"].unique()
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("\nConfusion Matrix:")
print(labels)
print(cm)

# ------------------------------------------------------------
# SAVE MODEL + FEATURE LIST
# ------------------------------------------------------------

os.makedirs("models", exist_ok=True)

model_package = {
    "model": model,
    "feature_columns": feature_columns,
    "classes": labels
}

joblib.dump(
    model_package,
    MODEL_FILE
)

print("\nModel saved to:")
print(MODEL_FILE)

# ------------------------------------------------------------
# SAVE TEST PREDICTIONS
# ------------------------------------------------------------

test_results = test_df[
    ["subject", "upperbody_label"]
].copy()

test_results["predicted_posture"] = y_pred

test_results.to_csv(
    PREDICTION_FILE,
    index=False
)

print("\nTest predictions saved to:")
print(PREDICTION_FILE)

print("\n" + "=" * 60)
print("POSTURE CLASSIFICATION COMPLETED")
print("=" * 60)