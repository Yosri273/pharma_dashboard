# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# E-commerce Data Pipeline Scheduler - V1.1 (Stable)
#
# This scheduler runs continuously, checking for new data files in the
# 'incoming_data' folder and processing them automatically. It does not
# include any web scraping logic.
# -----------------------------------------------------------------------------

import time
import os
import shutil
from sqlalchemy import create_engine
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime

# Import the main functions from our data processing engine
from load_data import get_database_url, process_incoming_file_and_append

# --- CONFIGURATION ---
INCOMING_DIR = "incoming_data"
ARCHIVE_DIR = "archive"
# For testing, we are using a short interval. For production, you could increase this.
SCHEDULE_MINUTES = 1

# --- SCHEDULER JOB ---
def check_for_new_files():
    """The main job that the scheduler will run."""
    print(f"\n[{datetime.now().ctime()}] --- Checking for new files in '{INCOMING_DIR}'... ---")
    
    try:
        engine = create_engine(get_database_url())
        
        # Get a list of all files in the incoming directory
        files_to_process = [f for f in os.listdir(INCOMING_DIR) if f.endswith('.csv')]

        if not files_to_process:
            print("No new files found.")
            return

        print(f"Found {len(files_to_process)} new file(s): {files_to_process}")
        
        for filename in files_to_process:
            filepath = os.path.join(INCOMING_DIR, filename)
            
            # Process the file using the imported function from our central engine
            success = process_incoming_file_and_append(filepath, engine)
            
            # Move the file to the archive folder after processing
            if success:
                archive_path = os.path.join(ARCHIVE_DIR, filename)
                shutil.move(filepath, archive_path)
                print(f"  [ARCHIVED] Moved '{filename}' to '{ARCHIVE_DIR}'.")
            else:
                print(f"  [ERROR] Did not archive '{filename}' due to processing failure.")

    except FileNotFoundError:
        print(f"  [ERROR] Directory not found: '{INCOMING_DIR}'. Please create it.")
    except Exception as e:
        print(f"\n--- SCHEDULER ERROR ---")
        print(f"An unexpected error occurred during the scheduled job: {e}")


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    print("--- Starting Automated Data Pipeline Scheduler ---")
    print(f"Watching folder: '{INCOMING_DIR}'")
    print(f"Archive folder: '{ARCHIVE_DIR}'")
    print(f"Checking for new files every {SCHEDULE_MINUTES} minute(s).")
    print("Press Ctrl+C to stop the scheduler.")

    # Ensure directories exist
    if not os.path.exists(INCOMING_DIR):
        os.makedirs(INCOMING_DIR)
    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR)
    
    scheduler = BlockingScheduler()
    # Add the job to the scheduler
    scheduler.add_job(check_for_new_files, 'interval', minutes=SCHEDULE_MINUTES)
    
    try:
        # Start the scheduler
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\nScheduler stopped by user.")

