# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Utilities Module - V20.0
#
# This module contains small, reusable helper functions that are used
# across multiple parts of the application.
# -----------------------------------------------------------------------------

def create_placeholder_figure(message="Data Not Available"):
    """
    Creates a blank Plotly figure with a centered text message.
    Used as a placeholder when a chart cannot be generated.
    """
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

