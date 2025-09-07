# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Utilities Toolbox - V26.0 (Enterprise Refactor)
#
# Logic migrated from pharma_dashboard_backup/utils.py
# This module provides reusable helper functions for the application.
# -----------------------------------------------------------------------------

import logging
from typing import Dict, Any, Union

import dash_bootstrap_components as dbc
from dash import html

# --- 1. CONFIGURE LOGGER ---
logger = logging.getLogger(__name__)


# --- 2. MATHEMATICAL UTILITIES ---

def safe_division(numerator: Union[int, float], denominator: Union[int, float]) -> float:
    """
    Performs division safely, returning 0.0 if the denominator is zero.
    """
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


# --- 3. UI & PLOTTING UTILITIES ---

def create_placeholder_figure(message: str = "No data available") -> Dict[str, Any]:
    """
    Creates a blank Plotly figure with a custom message.
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
    """
    return dbc.CardBody([
        html.H4(title, className="card-title"),
        html.P(value, className="card-text fs-3")
    ])