# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# E-commerce Data Pipeline Scheduler - V2.0 (Refactored)
#
# PURPOSE: This is the primary script for ongoing, automated data ingestion.
# It watches the 'incoming_data' folder and calls the central data engine
# (`load_data.py`) to process new files.
# -----------------------------------------------------------------------------

import time
import os
import shutil
from sqlalchemy import create_engine
from apscheduler.schedulers.blocking import BlockingScheduler

# Import the processing logic from our central data engine
from load_data import get_database_url, process_incoming_file_and_append

# --- CONFIGURATION ---
INCOMING_DIR = "incoming_data"
ARCHIVE_DIR = "archive"
SCHEDULE_MINUTES = 1 # Check for new files every 1 minute for testing

# --- SCHEDULER JOB ---
def check_for_new_files():
    """The main job that the scheduler will run."""
    print(f"[{time.ctime()}] Checking for new files in '{INCOMING_DIR}'...")
    
    try:
        engine = create_engine(get_database_url())
        files_to_process = [f for f in os.listdir(INCOMING_DIR) if f.endswith('.csv')]

        if not files_to_process:
            print("No new files found.")
            return

        print(f"Found {len(files_to_process)} new file(s): {files_to_process}")
        
        for filename in files_to_process:
            filepath = os.path.join(INCOMING_DIR, filename)
            
            # Call the central processing function
            success = process_incoming_file_and_append(filepath, engine)
            
            archive_path = os.path.join(ARCHIVE_DIR, filename)
            if success:
                shutil.move(filepath, archive_path)
                print(f"  [ARCHIVED] Moved '{filename}' to '{ARCHIVE_DIR}'.")
            else:
                print(f"  [ERROR] Did not archive '{filename}' due to processing failure.")

    except Exception as e:
        print(f"\n--- SCHEDULER ERROR ---")
        print(f"An unexpected error occurred during the scheduled job: {e}")

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    print("--- Starting Automated Data Pipeline Scheduler ---")
    print(f"Watching folder: '{INCOMING_DIR}'")
    print(f"Press Ctrl+C to stop.")
    
    # Ensure directories exist
    os.makedirs(INCOMING_DIR, exist_ok=True)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    
    scheduler = BlockingScheduler()
    scheduler.add_job(check_for_new_files, 'interval', minutes=SCHEDULE_MINUTES)
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped.")

