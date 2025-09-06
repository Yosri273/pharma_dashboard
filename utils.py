# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Utilities Toolbox - V21.0 (Final Master)
#
# This module provides a set of clean, well-documented, and reusable utility
# functions for the application (e.g., math, UI helpers).
# -----------------------------------------------------------------------------

import logging
from typing import Dict, Any, Union
import dash_bootstrap_components as dbc
from dash import html

logger = logging.getLogger(__name__)

def safe_division(numerator: Union[int, float], denominator: Union[int, float]) -> float:
    """
    Performs division safely, returning 0.0 if the denominator is zero.
    This prevents "division by zero" errors in KPI calculations.

    Args:
        numerator: The number to be divided.
        denominator: The number to divide by.

    Returns:
        The result of the division, or 0.0 if an error occurs.
    """
    if denominator is None or denominator == 0 or numerator is None:
        return 0.0
    try:
        return float(numerator / denominator)
    except (ValueError, TypeError):
        return 0.0


def create_placeholder_figure(message: str = "No data available") -> Dict[str, Any]:
    """
    Creates a blank Plotly figure with a custom message.
    Used as a placeholder when data for a chart is missing or empty.

    Args:
        message: The text to display on the empty chart.

    Returns:
        A dictionary representing a blank Plotly figure layout.
    """
    logger.debug(f"Creating placeholder figure with message: '{message}'")
    return {
        "layout": {
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "annotations": [{
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 16}
            }]
        }
    }


def create_kpi_body(title: str, value: str) -> dbc.CardBody:
    """
    Creates a consistent CardBody for a Key Performance Indicator (KPI).
    This reusable component ensures all KPIs have the same style and structure.

    Args:
        title: The title of the KPI (e.g., "Total Revenue").
        value: The formatted string value of the KPI (e.g., "1,234.56 SAR").

    Returns:
        A dash_bootstrap_components.CardBody object.
    """
    return dbc.CardBody([
        html.H4(title, className="card-title"),
        html.P(value, className="card-text fs-3")
    ])

