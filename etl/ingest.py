# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Data Ingestion Module
#
# Contains functions from pharma_dashboard_backup/load_data.py that are
# responsible for processing and appending new files.
# -----------------------------------------------------------------------------

import pandas as pd
import os
import logging

# Import from new modular structure
from config.settings import TABLE_CONFIG

logger = logging.getLogger(__name__)

def normalize_headers(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """
    Case-insensitively renames DataFrame columns based on the schema.
    (This function moved from load_data.py)
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
    logging.debug(f"Normalized headers for {list(schema.keys())[0]}: {df.columns.tolist()}")
    return df

def process_incoming_file_and_append(filepath: str, engine) -> bool:
    """
    Processes a single incoming file and appends it to the database.
    (This function moved from load_data.py)
    """
    logger.info(f"--- Processing incoming file: {filepath} ---")
    filename = os.path.basename(filepath).lower()
    table_name = None
    
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
        
        df.to_sql(table_name, engine, if_exists='append', index=False)
        logger.info(f"  [SUCCESS] Appended {len(df)} rows to '{table_name}'.")
        return True
    except Exception as e:
        logger.error(f"Failed to process and append file '{filepath}'. Error: {e}", exc_info=True)
        return False