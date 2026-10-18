import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict


# ---------------------------------
# 1. Load cleaned dataset
# ---------------------------------

df = pd.read_csv("results/cleaned_data.csv")

print("Dataset shape:", df.shape)


# ---------------------------------
# 2. Define features and target
# ---------------------------------

exclude_columns = [
    "subject",
    "upperbody_label",
    "lowerbody_label"
]

feature_columns = [
    col for col in df.columns
    if col not in exclude_columns
]

X = df[feature_columns]

y = df["upperbody_label"]

groups = df["subject"]


print("\nFeatures:", X.shape[1])
print("Subjects:", groups.nunique())
print("Classes:", y.nunique())


# ---------------------------------
# 3. Create participant-wise folds
# ---------------------------------

group_kfold = GroupKFold(n_splits=5)


# ---------------------------------
# 4. Create Random Forest
# ---------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# ---------------------------------
# 5. Cross-validation
# ---------------------------------

scores = cross_val_score(
    model,
    X,
    y,
    cv=group_kfold,
    groups=groups,
    scoring="accuracy"
)


print("\n==============================")
print("PARTICIPANT-WISE EVALUATION")
print("==============================")

for i, score in enumerate(scores, start=1):
    print(f"Fold {i} Accuracy: {score:.4f}")

print("\nMean Accuracy:", scores.mean())
print("Standard Deviation:", scores.std())


# ---------------------------------
# 6. Detailed predictions
# ---------------------------------

predictions = cross_val_predict(
    model,
    X,
    y,
    cv=group_kfold,
    groups=groups
)


print("\nClassification Report:")
print(classification_report(y, predictions))


print("\nEvaluation completed.")