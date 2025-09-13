# -*- coding: utf-8 -*-
import logging
from typing import Dict, Any
import dash_bootstrap_components as dbc
from dash import html

logger = logging.getLogger(__name__)

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
    # Executive-level style: add icon, improved hierarchy
    return dbc.CardBody([
        html.H3(value, className="mb-0 fw-bold")
    ])