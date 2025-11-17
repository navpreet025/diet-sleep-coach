# train_fitness.py
"""
Train weekly steps recommender, save to saved_models/fitness_model.pkl
Uses: data/steps_tracker_dataset.csv
"""

import os
from app_modules.helpers import safe_read_csv, find_column_by_substrings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# -----------------------------
# PATHS
# -----------------------------
DATA_PATH = "data/steps_tracker_dataset.csv"
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading:", DATA_PATH)
df = safe_read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# -----------------------------
# Detect Columns
# -----------------------------
date_col = find_column_by_substrings(df.columns.tolist(), ["date", "timestamp", "day"])
steps_col = find_column_by_substrings(df.columns.tolist(), ["step", "steps", "daily steps"])

if date_col is None or steps_col is None:
    raise RuntimeError("Could not detect date or steps columns. Inspect CSV and edit script.")

# -----------------------------
# Convert Date Column
# -----------------------------
df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")

# Fill missing dates forward
df[date_col] = df[date_col].fillna(method="ffill")

# Sort dataset
df = df.sort_values(date_col)

# -----------------------------
# Weekly Aggregation
# -----------------------------
df["week_start"] = df[date_col].dt.to_period("W").apply(lambda r: r.start_time)

weekly = df.groupby("week_start").agg(
    weekly_steps=(steps_col, "sum"),
    avg_daily_steps=(steps_col, "mean"),
    days_count=(steps_col, "count")
).reset_index().sort_values("week_start")

# Optional: distance column
distance_col = find_column_by_substrings(df.columns.tolist(), ["distance", "km"])
if distance_col:
    weekly["avg_distance_km"] = df.groupby("week_start")[distance_col].mean().values

# Target variable: next week’s steps
weekly["target_next_week"] = weekly["weekly_steps"].shift(-1)
weekly = weekly.dropna(subset=["target_next_week"])

print("Weekly rows:", weekly.shape[0])

# -----------------------------
# Features for Model
# -----------------------------
features = ["avg_daily_steps", "days_count"]
if "avg_distance_km" in weekly.columns:
    features.append("avg_distance_km")

X = weekly[features].fillna(0)
y = weekly["target_next_week"]

# -----------------------------
# Train / Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------
model_path = os.path.join(MODEL_DIR, "fitness_model.pkl")
joblib.dump(model, model_path)
print("Saved:", model_path)
