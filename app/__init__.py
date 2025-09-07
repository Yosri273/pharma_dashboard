# app/__init__.py
import os
import dash
import dash_bootstrap_components as dbc
import logging
from flask_caching import Cache
from config.settings import settings

# 1. Setup Logging
# Configure logging based on settings before doing anything else
logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.info(f"Logging configured at level: {settings.LOG_LEVEL}")

# 2. Initialize Cache
# This is crucial for performance in Dash apps
cache = Cache()

# 3. Initialize Dash App
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title=settings.APP_NAME
)

# Expose the Flask server for Gunicorn/WSGI
server = app.server

# Configure Flask server from settings
server.config.update(
    SECRET_KEY=os.urandom(24),
    CACHE_TYPE='FileSystemCache', # Example: Use filesystem for caching
    CACHE_DIR='cache-directory'
)

# 4. Initialize Cache with the app server
cache.init_app(server)
logger.info("Cache initialized.")

# 5. Import and Register Layout and Callbacks
# Import *after* app is created to avoid circular imports
from app import layout
from app import callbacks

app.layout = layout.create_layout()
callbacks.register_callbacks(app)

logger.info("Application factory completed. Layout and callbacks registered.")