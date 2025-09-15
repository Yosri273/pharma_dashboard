# pharma_dashboard/app/callbacks/common.py

import json
import logging
import dash
from dash import Input, Output, State, html
from datetime import datetime
import os
import dash_bootstrap_components as dbc # Added dbc import  

from services.db import get_engine
from etl.transforms import initialize_data

logger = logging.getLogger(__name__)
ALERT_STATE_FILE = "cache-directory/active_alerts.json"

def register_common_callbacks(app):
    """Registers callbacks that are not specific to a single dashboard tab."""

    @app.callback(
        Output('active-alert-banner-container', 'children'),
        Input('alert-poll-interval', 'n_intervals')
    )
    def update_active_alert_display(n):
        if not os.path.exists(ALERT_STATE_FILE):
            return []
        try:
            with open(ALERT_STATE_FILE, 'r') as f:
                content = f.read()
                if not content:
                    return []
                alert_state = json.loads(content)
        except Exception as e:
            logger.error(f"Error reading alert state file: {e}")
            return []

        banners = []
        for name, data in alert_state.items():
            if data.get("is_active"):
                ts = data.get("last_triggered", "")
                header = f"{name} (Since: {datetime.fromisoformat(ts).strftime('%Y-%m-%d %I:%M %p')})" if ts else name
                banners.append(dbc.Alert([html.H5(header), html.P(data.get("message"))], color="danger", dismissable=True, duration=45000))
        return banners

    @app.callback(
        Output("navbar-collapse", "is_open"),
        Input("navbar-toggler", "n_clicks"),
        State("navbar-collapse", "is_open"),
    )
    def toggle_navbar_collapse(n, is_open):
        if n:
            return not is_open
        return is_open

    @app.callback(
        Output('data-store-trigger', 'data'),
        Input('refresh-data-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def handle_refresh(n_clicks):
        logger.info("Refresh data button clicked.")
        engine = get_engine()
        initialize_data(engine)
        return "refreshed"

    @app.callback(
        Output('tab-content', 'children'),
        Input('tabs-controller', 'active_tab')
    )
    def render_tab_content(active_tab):
        from app.layout import (create_sales_layout, create_delivery_layout, create_customer_layout,
                                create_marketing_layout, create_profit_layout, create_predictive_layout)
        layouts = {
            "sales-tab": create_sales_layout,
            "delivery-tab": create_delivery_layout,
            "customer-tab": create_customer_layout,
            "marketing-tab": create_marketing_layout,
            "profit-tab": create_profit_layout,
            "predictive-tab": create_predictive_layout
        }
        # If a tab is not handled here, return an empty container so no
        # "Tab not found." message is shown on the UI (some tabs embed
        # their children directly in the Tabs declaration).
        return layouts.get(active_tab, lambda: html.Div())()

    # Debug: echo active_tab into a hidden div so we can verify clicks reach the server
    @app.callback(Output('tab-debug', 'children'), Input('tabs-controller', 'active_tab'))
    def echo_active_tab(active_tab):
        return str(active_tab)

    # Navigate from Welcome -> Comprehensive when the Get Started button is clicked
    @app.callback(Output('tabs-controller', 'active_tab'), Input('welcome-get-started', 'n_clicks'), prevent_initial_call=True)
    def welcome_get_started(n):
        if not n:
            return dash.no_update
        return 'comprehensive-tab'