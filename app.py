# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# E-commerce Data Analyst App - V21.1 (UI Upgrade)
#
# This is the main entry point for the application. It has been updated to
# include the Bootstrap Icons library for a more modern UI.
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
logger.info("--- Starting Pharma Analytics Hub v21.1 ---")
# --- FIX: Added Bootstrap Icons to the external stylesheets ---
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True
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
    app.run(debug=True, port=8051)

