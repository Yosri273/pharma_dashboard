# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Automated Data Pipeline Scheduler - V21.0 (Final Master)
#
# Logic migrated from pharma_dashboard_backup/scheduler.py
# All import paths updated.
# -----------------------------------------------------------------------------

import logging
import os
import shutil
import sys
import signal
from apscheduler.schedulers.blocking import BlockingScheduler

# Import from our new central modules
from etl.ingest import process_incoming_file_and_append  # FIX: Path updated
from services.db import get_engine                       # FIX: Path updated

# Configure professional logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
INCOMING_DIR = "incoming_data"
ARCHIVE_DIR = "archive"

def process_new_files_job():
    """
    The main job that the scheduler will run.
    (Original function from scheduler.py preserved exactly)
    """
    logger.info("--- Scheduler waking up, checking for new files... ---")
    try:
        engine = get_engine()
        if not os.path.exists(INCOMING_DIR):
            logger.warning(f"'{INCOMING_DIR}' not found. Creating it.")
            os.makedirs(INCOMING_DIR)
        if not os.path.exists(ARCHIVE_DIR):
            logger.warning(f"'{ARCHIVE_DIR}' not found. Creating it.")
            os.makedirs(ARCHIVE_DIR)

        files_to_process = [f for f in os.listdir(INCOMING_DIR) if f.endswith('.csv')]

        if not files_to_process:
            logger.info("No new files found.")
            return

        logger.info(f"Found {len(files_to_process)} new file(s): {files_to_process}")
        for filename in files_to_process:
            filepath = os.path.join(INCOMING_DIR, filename)
            success = process_incoming_file_and_append(filepath, engine)
            
            if success:
                archive_path = os.path.join(ARCHIVE_DIR, filename)
                shutil.move(filepath, archive_path)
                logger.info(f"  [ARCHIVED] Moved '{filename}' to '{ARCHIVE_DIR}'.")
            else:
                logger.error(f"  [ERROR] Failed to process '{filename}'. It will remain in the incoming folder for review.")

    except Exception as e:
        logger.error(f"  [FATAL] The scheduler job failed unexpectedly. Error: {e}", exc_info=True)

if __name__ == "__main__":
    # (Original main block from scheduler.py preserved exactly)
    logger.info("--- Starting Automated Data Pipeline Scheduler v21.0 ---")
    
    scheduler = BlockingScheduler()
    scheduler.add_job(process_new_files_job, 'interval', minutes=1, id='process_incoming_files')
    
    def shutdown(signum, frame):
        logger.warning(f"Shutdown signal {signum} received. Shutting down scheduler...")
        scheduler.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info(f"Scheduler is now watching '{INCOMING_DIR}'. Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped by user.")