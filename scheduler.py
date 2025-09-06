# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Enterprise Job Orchestrator - V26.0 (Refactored for Production)
#
# This module is a robust, persistent scheduler for the data pipeline.
# Key features:
# - Uses a PostgresJobStore for stateful, persistent scheduling.
# - Implements professional logging for monitoring.
# - Handles graceful shutdown signals for containerized environments.
# - Includes a heartbeat for liveness checks.
# -----------------------------------------------------------------------------

import logging
import os
import shutil
import signal
import sys
import time
from datetime import datetime

from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.blocking import BlockingScheduler
from sqlalchemy.exc import OperationalError
from tenacity import retry, stop_after_attempt, wait_fixed

# --- 1. IMPORT FROM OUR CENTRAL MODULES ---
from database import get_engine
from load_data import process_incoming_file_and_append
# We will disable the scraper for now as per our last decision,
# but the architecture is ready for it.
# from scraper import run_scraper

# --- 2. CONFIGURE LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("ETL_Scheduler")


# --- 3. DEFINE SCHEDULER JOBS ---

def run_data_loading_job():
    """
    Checks the incoming directory for new files and processes them.
    This is the core data ingestion task.
    """
    incoming_dir = "incoming_data"
    archive_dir = "archive"
    
    logger.info("--- Starting data loading job... ---")
    try:
        engine = get_engine()
        files_to_process = [f for f in os.listdir(incoming_dir) if f.endswith('.csv')]

        if not files_to_process:
            logger.info("No new files found to load.")
            return

        logger.info(f"Found {len(files_to_process)} file(s) to process: {files_to_process}")
        for filename in files_to_process:
            filepath = os.path.join(incoming_dir, filename)
            # Call the processing function from our central ETL engine
            success = process_incoming_file_and_append(filepath, engine)
            if success:
                archive_path = os.path.join(archive_dir, filename)
                shutil.move(filepath, archive_path)
                logger.info(f"  [ARCHIVED] Moved '{filename}' to '{archive_dir}'.")

    except FileNotFoundError:
        logger.warning(f"Directory not found: '{incoming_dir}'. Please create it. Skipping run.")
    except Exception as e:
        logger.error(f"  [FATAL ERROR] The data loader job failed. Error: {e}", exc_info=True)


def heartbeat_job():
    """A simple job that logs a message to confirm the scheduler is alive."""
    logger.info("Scheduler heartbeat: I am alive and running.")


# --- 4. MAIN SCHEDULER INITIALIZATION AND EXECUTION ---
if __name__ == "__main__":
    logger.info("--- Initializing Enterprise Job Orchestrator v26.0 ---")

    # --- Use a persistent Job Store ---
    # This stores job states in our main database, so if the scheduler is
    # restarted, it remembers its jobs and their last run times.
    jobstores = {
        'default': SQLAlchemyJobStore(url=get_engine().url)
    }

    scheduler = BlockingScheduler(jobstores=jobstores, timezone='Asia/Riyadh')

    # Add jobs to the scheduler with cron-style triggers
    # This job will run every day at 2:00 AM
    scheduler.add_job(
        run_data_loading_job,
        trigger='cron',
        hour=2,
        minute=0,
        id='daily_data_load',
        name='Daily ETL Processing Job',
        replace_existing=True
    )
    
    # This heartbeat job runs every 15 minutes for monitoring
    scheduler.add_job(
        heartbeat_job,
        trigger='interval',
        minutes=15,
        id='scheduler_heartbeat',
        name='Scheduler Liveness Check'
    )

    logger.info("Scheduler initialized. Current jobs:")
    scheduler.print_jobs()

    # --- Graceful Shutdown Handling ---
    def shutdown(signum, frame):
        logger.warning("Shutdown signal received. Shutting down scheduler...")
        scheduler.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)  # For Ctrl+C
    signal.signal(signal.SIGTERM, shutdown) # For Docker/Kubernetes stop signals
    
    # --- Start the Scheduler ---
    logger.info("\n--- Scheduler is now running. Press Ctrl+C to stop. ---")
    try:
        # Run the loading job once on startup for immediate feedback
        logger.info("Performing initial data load on startup...")
        run_data_loading_job()
        
        # Start the main scheduling loop
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as e:
        logger.critical(f"Scheduler failed to start or crashed. Error: {e}", exc_info=True)
        sys.exit(1)

