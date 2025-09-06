# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# E-commerce Data Loader - V21.0 (Final Master)
#
# This script is the master setup tool for the database. It reads all local
# master CSV files and completely rebuilds the database tables. It also
# provides a function for the scheduler to process incremental files.
# -----------------------------------------------------------------------------

import pandas as pd
import sys
import os
import logging

# Import from our central, single-source-of-truth modules
from config import TABLE_CONFIG
from database import get_engine

# Configure professional logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

def normalize_headers(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """
    Case-insensitively renames DataFrame columns based on the schema.
    This makes the loader resilient to minor changes in CSV header casing.
    """
    header_map = {}
    cols_lower = {c.lower(): c for c in df.columns}
    for clean_name, possible_names in schema.items():
        for pname in possible_names:
            p_low = pname.lower()
            if p_low in cols_lower:
                header_map[cols_lower[p_low]] = clean_name
                break
    
    df = df.rename(columns=header_map)
    # This debug log is helpful for verifying header mapping
    logging.debug(f"Normalized headers for {list(schema.keys())[0]}: {df.columns.tolist()}")
    return df

def bootstrap_database(engine):
    """
    Loads all master CSV files from the main directory, completely replacing
    all tables in the database. This is for initial setup or a full reset.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for table_name, config in TABLE_CONFIG.items():
        logger.info(f"--- Bootstrapping table: {table_name} ---")
        file_path = os.path.join(base_dir, config['filename'])
        try:
            df = pd.read_csv(file_path)
            df = normalize_headers(df, config['schema_norm'])
            
            # Add calculated columns after cleaning headers
            if 'grossvalue' in df.columns and 'discountvalue' in df.columns:
                df['netsale'] = df['grossvalue'] - df['discountvalue']
            
            # if_exists='replace' will drop the table first if it exists
            df.to_sql(table_name, engine, if_exists='replace', index=False)
            logger.info(f"  [SUCCESS] Table '{table_name}' created with {len(df)} rows.")
        except FileNotFoundError:
            logger.error(f"Master file not found: {config['filename']}. Skipping.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while processing {config['filename']}: {e}", exc_info=True)
            raise  # Stop the entire bootstrap process if any file fails

def process_incoming_file_and_append(filepath: str, engine) -> bool:
    """
    Processes a single incoming file and appends it to the database.
    Returns True on success and False on failure.
    """
    logger.info(f"--- Processing incoming file: {filepath} ---")
    filename = os.path.basename(filepath).lower()
    table_name = None
    
    # Determine which table this file belongs to based on its prefix
    for name, config in TABLE_CONFIG.items():
        if filename.startswith(config.get('file_prefix', '')):
            table_name = name
            break
            
    if not table_name:
        logger.warning(f"Unrecognized file prefix for '{filename}'. Skipping.")
        return False

    try:
        df = pd.read_csv(filepath)
        df = normalize_headers(df, TABLE_CONFIG[table_name]['schema_norm'])
        
        if 'grossvalue' in df.columns and 'discountvalue' in df.columns:
            df['netsale'] = df['grossvalue'] - df['discountvalue']
        
        # Use if_exists='append' to add new data without deleting old data
        df.to_sql(table_name, engine, if_exists='append', index=False)
        logger.info(f"  [SUCCESS] Appended {len(df)} rows to '{table_name}'.")
        return True # Return True on success for archiving
    except Exception as e:
        logger.error(f"Failed to process and append file '{filepath}'. Error: {e}", exc_info=True)
        return False # Return False on failure

if __name__ == "__main__":
    logger.info("--- Running Database Bootstrap Tool v21.0 ---")
    try:
        engine = get_engine()
        # The get_engine function already tests the connection
        bootstrap_database(engine)
        logger.info("\n--- Database bootstrap process finished successfully ---")
    except Exception as e:
        logger.critical(f"\n--- Bootstrap failed. Error: {e}", exc_info=True)
        sys.exit(1)

