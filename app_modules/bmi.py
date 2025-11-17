# app_modules/bmi.py
from typing import Optional

def compute_bmi(weight_kg: float, height_cm: float) -> Optional[float]:
    """Compute BMI (kg/m^2) rounded to 2 decimals. Returns None if invalid."""
    try:
        h = float(height_cm) / 100.0
        if h <= 0:
            return None
        bmi = float(weight_kg) / (h * h)
        return round(bmi, 2)
    except Exception:
        return None

def diet_recommendation(bmi: Optional[float], goal: str = "maintain", preference: str = "none") -> str:
    """Return a simple rule-based diet recommendation string."""
    if bmi is None:
        return "Invalid height or weight."

    if bmi < 18.5:
        plan = "Underweight: increase calories, focus on protein and healthy carbs; include snacks."
    elif bmi < 25:
        plan = "Normal: balanced diet with portion control and variety."
    elif bmi < 30:
        plan = "Overweight: mild calorie deficit, reduce refined carbs, increase protein & fiber."
    else:
        plan = "Obese: structured calorie deficit and medical guidance recommended."

    if goal == "lose":
        plan += " Aim for ~500 kcal/day deficit + regular exercise."
    elif goal == "gain":
        plan += " Aim for ~300–500 kcal/day surplus with strength training."

    if preference.lower() in ("vegetarian", "vegan"):
        plan += f" ({preference.title()} options: legumes, tofu, tempeh, nuts, seeds)."

    return plan
