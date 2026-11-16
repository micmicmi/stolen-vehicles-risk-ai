"""
utils.py - פונקציות עזר כלליות
===============================
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime


def ensure_dir(path: str):
    """מוודאת שתיקייה קיימת, ויוצרת אם לא."""
    os.makedirs(path, exist_ok=True)


def log_step(step_name: str, message: str):
    """מדפיסה הודעת לוג מעוצבת."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] 📌 {step_name}: {message}")


def save_summary(summary: dict, path: str):
    """שומרת סיכום ל-JSON."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"💾 Summary saved to: {path}")


def format_number(n: float, decimals: int = 2) -> str:
    """מעצבת מספר להצגה (עם פסיקים)."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.{decimals}f}M"
    elif n >= 1_000:
        return f"{n/1_000:.{decimals}f}K"
    else:
        return f"{n:,.{decimals}f}"


def print_separator(char: str = "═", length: int = 60):
    """מדפיסה קו מפריד."""
    print(char * length)


def calculate_percentage_change(old: float, new: float) -> float:
    """מחשבת אחוז שינוי."""
    if old == 0:
        return 0.0
    return round(((new - old) / old) * 100, 2)
