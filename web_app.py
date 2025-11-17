# web_app.py
"""
Streamlit front-end for Diet, Sleep & Fitness predictions.
Clean layout, using the exact features of trained models.
"""

import streamlit as st
import joblib
import os
import pandas as pd
from app_modules.bmi import compute_bmi, diet_recommendation

MODEL_DIR = "saved_models"

# -------------------------------
# Load models
# -------------------------------
@st.cache_resource
def load_models():
    models = {}
    models["sleep"] = joblib.load(os.path.join(MODEL_DIR, "sleep_model.pkl")) if os.path.exists(os.path.join(MODEL_DIR, "sleep_model.pkl")) else None
    models["fitness"] = joblib.load(os.path.join(MODEL_DIR, "fitness_model.pkl")) if os.path.exists(os.path.join(MODEL_DIR, "fitness_model.pkl")) else None
    return models

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="AI Diet & Wellness Coach", layout="wide")
st.title("🩺 AI Diet & Wellness Coach")

models = load_models()

# -------------------------------
# Profile & BMI Section
# -------------------------------
st.header("👤 Profile & BMI")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 8, 120, 25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    height_unit = st.radio("Height Unit", ["cm", "feet & inches"])
    if height_unit == "cm":
        height_cm = st.number_input("Height (cm)", 50, 250, 170)
    else:
        feet = st.number_input("Feet", 2, 8, 5)
        inches = st.number_input("Inches", 0, 11, 6)
        height_cm = round(feet * 30.48 + inches * 2.54, 1)

with col2:
    weight_kg = st.number_input("Weight (kg)", 10.0, 300.0, 70.0)
    goal = st.selectbox("Goal", ["maintain", "lose", "gain"])
    preference = st.selectbox("Diet Preference", ["none", "vegetarian", "vegan"])

if st.button("Compute BMI & Diet"):
    bmi = compute_bmi(weight_kg, height_cm)
    st.metric("BMI", bmi)
    st.write("**Diet Recommendation**")
    st.write(diet_recommendation(bmi, goal, preference))

st.markdown("---")

# -------------------------------
# Sleep Quality Prediction
# -------------------------------
st.header("🛌 Sleep Quality Prediction")
col1, col2, col3 = st.columns(3)

with col1:
    sleep_duration = st.slider("Sleep Duration (hours)", 0.0, 12.0, 7.0, 0.25)
    daily_steps = st.number_input("Daily Steps", 0, 50000, 6000)

with col2:
    stress_level = st.slider("Stress Level (0-10)", 0, 10, 3)
    physical_activity = st.slider("Physical Activity Level (0-10)", 0, 10, 5)

with col3:
    heart_rate = st.number_input("Heart Rate (bpm)", 30, 200, 70)
    bmi_category = st.selectbox(
        "BMI Category",
        ["Underweight", "Normal", "Overweight", "Obese"]
    )

if st.button("Predict Sleep Quality"):
    sleep_model = models.get("sleep")
    if sleep_model is None:
        st.error("Sleep model missing. Run train_sleep.py")
    else:
        X = pd.DataFrame([{
            "Sleep Duration": sleep_duration,
            "Daily Steps": daily_steps,
            "Stress Level": stress_level,
            "Heart Rate": heart_rate,
            "BMI Category": bmi_category,
            "Physical Activity Level": physical_activity
        }])
        try:
            prediction = sleep_model.predict(X)[0]
            st.success(f"Predicted Sleep Quality: **{prediction}**")
        except Exception as e:
            st.error("Prediction failed. Check training features.")
            st.write(e)

st.markdown("---")

# -------------------------------
# Fitness Goal Recommendation
# -------------------------------
st.header("🏋️ Weekly Fitness Goal Recommendation")

avg_daily_steps = st.number_input("Average Daily Steps", 0, 50000, 6000)
days_count = st.number_input("Days Count (per week)", 1, 7, 7)
avg_distance_km = st.number_input("Average Distance (km)", 0.0, 50.0, 4.0)

if st.button("Recommend Weekly Steps"):
    fitness_model = models.get("fitness")
    if fitness_model is None:
        st.error("Fitness model missing.")
    else:
        Xf = pd.DataFrame([{
            "avg_daily_steps": avg_daily_steps,
            "days_count": days_count,
            "avg_distance_km": avg_distance_km
        }])
        try:
            weekly_steps = int(fitness_model.predict(Xf)[0])
            st.info(f"Recommended Weekly Target: **{weekly_steps:,} steps**")
            st.progress(min(weekly_steps / 70000, 1.0))  # progress bar relative to 10k/day goal
            st.write(f"≈ {weekly_steps // 7:,} steps/day")
        except Exception as e:
            st.error("Fitness prediction failed.")
            st.write(e)

st.markdown("---")
st.caption("All model predictions use exact feature names from training.")










