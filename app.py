# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# E-commerce Data Analyst App - V22.0 (Mobile Optimized)
#
# This is the main entry point for the application. It has been updated to
# include the Bootstrap Icons library and the critical viewport meta tag
# to ensure true responsiveness on mobile devices.
# -----------------------------------------------------------------------------

import dash
import logging
import sys
import dash_bootstrap_components as dbc

# Import the necessary components from our new modules
from layouts import create_main_layout
from callbacks import register_callbacks
from data import initialize_data
from database import get_engine

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- 1. INITIALIZE APP AND DATA ---
logger.info("--- Starting Pharma Analytics Hub v22.0 (Mobile Optimized) ---")

# --- CRITICAL FIX: Added meta_tags with viewport for mobile responsiveness ---
# This tag tells mobile browsers to render the page at the device's width.
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ]
)
server = app.server
app.title = "Pharma Analytics Hub"

engine = get_engine()
initialize_data(engine)

# --- 2. DEFINE APP LAYOUT & REGISTER CALLBACKS ---
app.layout = create_main_layout()
register_callbacks(app)
logger.info("Application ready.")

# --- 3. RUN THE APP ---
if __name__ == '__main__':
    app.run(debug=True, port=8053)