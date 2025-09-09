# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Database Bootstrap Script
#
# Contains the bootstrap_database function from pharma_dashboard_backup/load_data.py
# This script should be run once to populate the database from master CSVs.
#
# BUG FIX (V21.1): Added sys.path modification to allow the script to be run
#                  directly from the command line without ModuleNotFoundErrors.
# -----------------------------------------------------------------------------

import pandas as pd
import sys
import os
import logging

# --- FIX: Add project root to Python's import path ---
# This allows the script to find and import modules from other directories
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- END FIX ---

# Import from new modular structure (now works correctly)
from config.settings import TABLE_CONFIG
from services.db import get_engine
from etl.ingest import normalize_headers

# Configure professional logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)


def bootstrap_database(engine):
    """
    Loads all master CSV files, completely replacing all tables.
    """
    # Base directory is now correctly identified as the project root
    base_dir = project_root
    
    for table_name, config in TABLE_CONFIG.items():
        logger.info(f"--- Bootstrapping table: {table_name} ---")
        file_path = os.path.join(base_dir, config['filename'])
        try:
            df = pd.read_csv(file_path)
            df = normalize_headers(df, config['schema_norm'])
            
            # This logic for 'netsale' is now deprecated as it's handled
            # in etl/transforms.py, but it's safe to keep for the bootstrap.
            if 'grossvalue' in df.columns and 'discountvalue' in df.columns:
                df['netsale'] = df['grossvalue'] - df['discountvalue']
            
            df.to_sql(table_name, engine, if_exists='replace', index=False)
            logger.info(f"  [SUCCESS] Table '{table_name}' created with {len(df)} rows.")
        except FileNotFoundError:
            logger.error(f"Master file not found at: {file_path}. Skipping. Make sure all CSV files are in the project root.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while processing {config['filename']}: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    logger.info("--- Running Database Bootstrap Tool v21.1 ---")
    try:
        engine = get_engine()
        bootstrap_database(engine)
        logger.info("\n--- Database bootstrap process finished successfully ---")
    except Exception as e:
        logger.critical(f"\n--- Bootstrap failed. Error: {e}", exc_info=True)
        sys.exit(1)
