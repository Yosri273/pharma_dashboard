import pandas as pd
import numpy as np

# --- NEW FUNCTION TO PREVENT DASHBOARD CRASHES ---
def safe_division(numerator, denominator):
    """
    Prevents ZeroDivisionError and handles None/NaN inputs for robust KPI calculation.
    Returns 0.0 if division is not possible.
    """
    if denominator is None or denominator == 0 or pd.isna(denominator):
        return 0.0
    
    if numerator is None or pd.isna(numerator):
        return 0.0

    try:
        result = float(numerator) / float(denominator)
        # Handle cases where division might result in infinity
        if not np.isfinite(result):
            return 0.0
        return result
    except (ZeroDivisionError, ValueError, TypeError):
        return 0.0

# --- Any other utility functions you already have (like formatting) would go here ---

def format_kpi_value(value, prefix="", suffix="", decimals=2):
    """
    Formats a numerical value as a string for the KPI cards.
    """
    try:
        formatted_value = f"{value:,.{decimals}f}"
        return f"{prefix}{formatted_value}{suffix}"
    except (ValueError, TypeError):
        return f"{prefix}0{suffix}"

# Example of another utility function that might exist
def get_trend_icon(value):
    if value > 0:
        return "▲", "color: 'green'"
    elif value < 0:
        return "▼", "color: 'red'"
    else:
        return "", "color: 'white'"