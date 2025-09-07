# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Database Bootstrap Script
#
# Contains the bootstrap_database function from pharma_dashboard_backup/load_data.py
# This script should be run once to populate the database from master CSVs.
# -----------------------------------------------------------------------------

import pandas as pd
import sys
import os
import logging

# Import from new modular structure
# These imports assume the script is run from the project root (e.g., python scripts/bootstrap.py)
from config.settings import TABLE_CONFIG
from services.db import get_engine
from etl.ingest import normalize_headers  # Import the shared utility from its new location

# Configure professional logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)


def bootstrap_database(engine):
    """
    Loads all master CSV files, completely replacing all tables.
    (This function moved from load_data.py)
    """
    # Assume CSVs are in the root, or adjust path as needed. The original script
    # assumed CSVs were in the same dir as the script. Let's adjust to project root.
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    for table_name, config in TABLE_CONFIG.items():
        logger.info(f"--- Bootstrapping table: {table_name} ---")
        # All master CSVs (e.g., sales_data.csv) must be in the project root directory
        file_path = os.path.join(base_dir, config['filename'])
        try:
            df = pd.read_csv(file_path)
            df = normalize_headers(df, config['schema_norm'])
            
            if 'grossvalue' in df.columns and 'discountvalue' in df.columns:
                df['netsale'] = df['grossvalue'] - df['discountvalue']
            
            df.to_sql(table_name, engine, if_exists='replace', index=False)
            logger.info(f"  [SUCCESS] Table '{table_name}' created with {len(df)} rows.")
        except FileNotFoundError:
            logger.error(f"Master file not found at: {file_path}. Skipping.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while processing {config['filename']}: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    # (Main execution block from load_data.py)
    logger.info("--- Running Database Bootstrap Tool v21.0 ---")
    try:
        engine = get_engine()
        bootstrap_database(engine)
        logger.info("\n--- Database bootstrap process finished successfully ---")
    except Exception as e:
        logger.critical(f"\n--- Bootstrap failed. Error: {e}", exc_info=True)
        sys.exit(1)