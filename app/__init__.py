# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Pharma Analytics Hub - Main Application Package
#
# This file initializes the core Dash application object, loads all data,
# registers the layout, and connects all callbacks.
# -----------------------------------------------------------------------------

import dash
import logging
import sys
import dash_bootstrap_components as dbc

# Import components from the new modular structure
from app.layout import create_main_layout
from app.callbacks import register_all_callbacks
from etl.transforms import initialize_data
from services.db import get_engine

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- 1. INITIALIZE APP AND DATA ---
logger.info("--- Starting Pharma Analytics Hub v21.1 ---")
# Added Bootstrap Icons per original app.py
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True
)
server = app.server
app.title = "Yosri Analytics Hub"

engine = get_engine()
initialize_data(engine)

# --- 2. DEFINE APP LAYOUT & REGISTER CALLBACKS ---
app.layout = create_main_layout()
register_all_callbacks(app)
logger.info("Application ready.")