# etl/ingest.py
import pandas as pd
import logging
from config.settings import settings, BASE_DIR
from services.db import get_db_engine  # Import the new engine getter
import os
from services import db # Import db module to call create_tables

logger = logging.getLogger(__name__)

def load_csv_to_db(csv_path: str, table_name: str):
    """
    Loads data from a CSV and bulk-inserts it into a PostgreSQL table.
    This replaces existing data in the table completely.
    """
    full_path = os.path.join(BASE_DIR, csv_path)
    engine = get_db_engine()
    
    try:
        df = pd.read_csv(full_path)
        # Clean column names just in case
        df.columns = [col.lower().replace(' ', '_') for col in df.columns] 

        # Ensure timestamp columns are converted before sending to DB
        if 'order_date' in df.columns:
             df['order_date'] = pd.to_datetime(df['order_date'])

        # This is the magic: one command to bulk-load the entire dataframe.
        # 'replace' automatically drops the table if it exists and creates it new
        # based on the dataframe structure. This is perfect for a full refresh.
        df.to_sql(
            table_name,
            con=engine,
            if_exists='replace', # Re-create the table every time
            index=False,         # Don't save the pandas index as a column
            method='multi'       # Use multi-row INSERTs for speed
        )
        
        logger.info(f"Successfully bulk-loaded {csv_path} to table {table_name}.")
        
    except FileNotFoundError:
        logger.error(f"Data file not found: {full_path}")
    except Exception as e:
        logger.error(f"Failed to load data from {csv_path} to {table_name}: {e}", exc_info=True)

def load_all_data():
    """
    Main ingestion function to load all data sources into the database.
    This is the target job for the scheduler AND the bootstrap script.
    """
    logger.info("Starting data ingestion job...")
    try:
        # CRITICAL: We call create_tables() first to ensure the schema exists.
        # Since df.to_sql() with 'replace' creates tables, this just acts as a safety check
        # and ensures any tables NOT from a CSV (if you add them later) also exist.
        db.create_tables() 
        
        # Load all our CSVs
        load_csv_to_db(settings.SALES_DATA_PATH, "sales")
        load_csv_to_db(settings.CUSTOMER_DATA_PATH, "customers")
        load_csv_to_db(settings.DELIVERY_DATA_PATH, "delivery")
        load_csv_to_db(settings.MARKETING_CAMPAIGNS_PATH, "marketing_campaigns")
        load_csv_to_db(settings.COMPETITOR_DATA_PATH, "competitor_data")
        load_csv_to_db(settings.MARKETING_ATTRIBUTION_PATH, "marketing_attribution")
        load_csv_to_db(settings.FUNNEL_DATA_PATH, "funnel")
        logger.info("Completed data ingestion job.")
    except Exception as e:
        logger.critical(f"Data ingestion job failed: {e}", exc_info=True)