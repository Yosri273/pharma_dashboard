# scripts/bootstrap.py
import logging
from services import db
from etl import ingest
from config.settings import settings

logging.basicConfig(level=settings.LOG_LEVEL.upper())
logger = logging.getLogger("bootstrap")

def main():
    """
    Runs the initial setup for the application:
    1. Creates database tables.
    2. Performs the initial data load from CSVs.
    """
    logger.info("--- Starting Application Bootstrap Process ---")
    
    # 1. Create database tables
    logger.info("Step 1: Creating database tables...")
    try:
        db.create_tables()
        logger.info("Database tables created successfully.")
    except Exception as e:
        logger.critical(f"Failed to create database tables: {e}", exc_info=True)
        return  # Exit if tables can't be created

    # 2. Run initial data load
    logger.info("Step 2: Performing initial data load from CSVs...")
    try:
        ingest.load_all_data()
        logger.info("Initial data load completed successfully.")
    except Exception as e:
        logger.critical(f"Initial data load failed: {e}", exc_info=True)

    logger.info("--- Bootstrap Process Finished ---")

if __name__ == "__main__":
    # This allows running the script directly:
    # python scripts/bootstrap.py
    main()