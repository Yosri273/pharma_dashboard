# etl/schedules.py
import time
import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from etl.ingest import load_all_data
from config.settings import settings

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

def scheduled_ingestion_job():
    """Wrapper function for the scheduled job."""
    logger.info("Scheduler triggered 'load_all_data' job...")
    try:
        load_all_data()
    except Exception as e:
        logger.critical(f"Scheduled job failed with error: {e}", exc_info=True)

def start_scheduler():
    """Initializes and starts the APScheduler."""
    scheduler = BlockingScheduler()
    
    # Schedule the job to run daily at 2:00 AM
    scheduler.add_job(scheduled_ingestion_job, 'cron', hour=2, minute=0)
    
    # You can also add it to run on startup for testing
    # scheduler.add_job(scheduled_ingestion_job, 'date', run_date=datetime.now() + timedelta(seconds=5))

    logger.info("Scheduler started. Waiting for next scheduled run...")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
        scheduler.shutdown()

if __name__ == "__main__":
    # This allows you to run the scheduler as a standalone process:
    # python -m etl.schedules
    start_scheduler()