# train_sleep.py
"""
Train Sleep Quality classifier, save to saved_models/sleep_model.pkl
Uses: data/Sleep_health_and_lifestyle_dataset.csv
"""

import os
from app_modules.helpers import safe_read_csv, normalize_sleep_label, find_column_by_substrings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATA_PATH ="data/Sleep_health_and_lifestyle_dataset.csv"
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading:", DATA_PATH)
df = safe_read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Detect target column (sleep quality)
target = find_column_by_substrings(df.columns.tolist(), ["quality of sleep", "sleep quality", "quality_of_sleep", "sleep_quality", "sleep quality label"])
if target is None:
    raise RuntimeError("Sleep quality column not detected. Inspect CSV header and update script.")

# Candidate features (auto-detect common names)
candidates = []
for key in ["sleep duration", "sleep_hours", "sleep hours", "daily steps", "steps", "stress level", "heart rate", "bmi", "physical activity"]:
    col = find_column_by_substrings(df.columns.tolist(), [key])
    if col and col not in candidates:
        candidates.append(col)

if not candidates:
    raise RuntimeError("No feature columns detected. Inspect CSV and edit script.")

print("Target:", target)
print("Features:", candidates)

df = df.dropna(subset=[target])
y = df[target].map(normalize_sleep_label)

X = df[candidates].copy()

# Convert numeric-like columns
numeric_cols = []
for c in X.columns:
    try:
        X[c] = pd.to_numeric(X[c])
        numeric_cols.append(c)
    except Exception:
        pass

cat_cols = [c for c in X.columns if c not in numeric_cols]

for c in numeric_cols:
    X[c] = X[c].fillna(X[c].median())
for c in cat_cols:
    X[c] = X[c].fillna("Unknown").astype(str)

num_pipe = Pipeline([("scale", StandardScaler())])
cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([("num", num_pipe, numeric_cols), ("cat", cat_pipe, cat_cols)])

pipeline = Pipeline([("pre", preprocessor), ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
print("Training rows:", X_train.shape[0])
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(pipeline, os.path.join(MODEL_DIR, "sleep_model.pkl"))
print("Saved:", os.path.join(MODEL_DIR, "sleep_model.pkl"))
