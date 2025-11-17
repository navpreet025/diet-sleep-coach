# app_modules/helpers.py
import pandas as pd
from typing import List, Optional

def safe_read_csv(path: str) -> pd.DataFrame:
    """Read CSV and strip whitespace from column names."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def normalize_sleep_label(label) -> str:
    """Normalize sleep quality labels to 'Good','Moderate','Poor'."""
    s = str(label).lower()
    if "good" in s or "excellent" in s:
        return "Good"
    if "poor" in s or "bad" in s:
        return "Poor"
    return "Moderate"

def find_column_by_substrings(columns: List[str], substrings: List[str]) -> Optional[str]:
    """Return first column name that contains any substring (case-insensitive) or None."""
    for sub in substrings:
        for c in columns:
            if sub.lower() in c.lower():
                return c
    return None
