# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Main entry point for the Pharma Analytics Hub.
# This script imports the fully configured app instance and runs the server.
# To run the application: python run.py
# -----------------------------------------------------------------------------

from app import app, server  # Import the app and server instances from our app package

if __name__ == '__main__':
    # We use the configuration from the original app.py
    app.run(debug=True, port=8053)