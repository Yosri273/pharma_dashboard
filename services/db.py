# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Secure Data Access Layer - V21.0 (Final Master)
#
# Logic migrated from pharma_dashboard_backup/database.py
# All import paths updated.
# -----------------------------------------------------------------------------

import logging
import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from typing import Dict

# Import from our new central, single source of truth
from config.settings import settings, TABLE_CONFIG  # FIX: Path updated

logger = logging.getLogger(__name__)

_engine = None

def get_engine() -> Engine:
    """
    Creates and returns a single, cached SQLAlchemy engine instance.
    (Original function from database.py preserved exactly)
    """
    global _engine
    if _engine is None:
        db_url = settings.DATABASE_URL
        logger.info(f"Creating new database engine for host: {db_url.split('@')[-1]}")
        _engine = create_engine(db_url)
        try:
            with _engine.connect() as connection:
                logger.info("Database engine created and connection successful.")
        except Exception as e:
            logger.critical(f"Database connection failed on initial creation: {e}", exc_info=True)
            raise
    return _engine

def safe_table_exists(engine: Engine, table_name: str) -> bool:
    """
    Securely checks if a table exists using the SQLAlchemy Inspector.
    (Original function from database.py preserved exactly)
    """
    try:
        inspector = inspect(engine)
        return table_name in inspector.get_table_names()
    except Exception as e:
        logger.error(f"Failed to inspect table existence for '{table_name}': {e}", exc_info=True)
        return False

def load_data_safely(table_name: str, engine: Engine) -> pd.DataFrame:
    """
    Securely loads a full table into a pandas DataFrame.
    (Original function from database.py preserved exactly)
    """
    if table_name not in TABLE_CONFIG:
        logger.error(f"[SECURITY] Attempted to load non-whitelisted table: '{table_name}'")
        return pd.DataFrame()

    try:
        if not safe_table_exists(engine, table_name):
            logger.warning(f"Table '{table_name}' not found. Returning empty DataFrame.")
            return pd.DataFrame()

        df = pd.read_sql_table(table_name, engine)
        df.columns = [col.lower() for col in df.columns]
        logger.info(f"Successfully loaded {len(df)} rows from table '{table_name}'.")
        return df
    except Exception as e:
        logger.error(f"Could not load table '{table_name}'. Error: {e}", exc_info=True)
        return pd.DataFrame()

def refresh_all_data(engine: Engine) -> Dict[str, pd.DataFrame]:
    """
    Loads all tables defined in the config into a dictionary of DataFrames.
    (Original function from database.py preserved exactly)
    """
    logger.info("--- Refreshing all data from database ---")
    dataframes = {}
    for table_name in TABLE_CONFIG.keys():
        dataframes[table_name] = load_data_safely(table_name, engine)
    logger.info("--- Data refresh complete ---")
    return dataframes