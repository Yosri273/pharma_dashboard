# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Main entry point for the Pharma Analytics Hub.
# This script imports the fully configured app instance and runs the server.
#
# MERGED: Added APScheduler from the new version to run the alerting monitor.
# -----------------------------------------------------------------------------

from app.bootstrap import app, server  # Use explicit bootstrap to avoid import-time side effects
from apscheduler.schedulers.background import BackgroundScheduler
from alerting.monitor import run_monitor_cycle # NEW: Import the alert monitor

# --- Add Scheduler for Alerting (New Feature) ---
# This scheduler will run in a background thread alongside the web server.
scheduler = BackgroundScheduler(daemon=True)

# Run the alert checks every 15 minutes.
scheduler.add_job(run_monitor_cycle, 'interval', minutes=15)
scheduler.start()
# -----------------------------------------------


if __name__ == '__main__':
    # This block now correctly runs the app that was already built by __init__.py
    print("Starting scheduler and web server...")
    try:
        # We use the configuration from the original app.py
        app.run(debug=True, port=8056)
    except (KeyboardInterrupt, SystemExit):
        print("Shutting down scheduler...")
        scheduler.shutdown()