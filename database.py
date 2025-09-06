# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Central Database Module - V20.0
#
# Handles all direct interactions with the PostgreSQL database, including a
# secure data loading function that prevents SQL injection by validating
# table names and using the SQLAlchemy Inspector.
# -----------------------------------------------------------------------------

import pandas as pd
from sqlalchemy import create_engine, inspect

# Import from our central configuration module
from config import get_database_url, TABLE_CONFIG

def get_engine():
    """Creates and returns a SQLAlchemy engine."""
    return create_engine(get_database_url())

def load_data_safely(table_name, engine):
    """
    Tries to load a table securely; returns an empty DataFrame on failure.
    Validates the table_name against a predefined list to prevent SQL injection.
    """
    # Validate that the requested table_name is a known, safe table from our config.
    if table_name not in TABLE_CONFIG:
        print(f"  [SECURITY WARNING] Attempted to load an unknown table: '{table_name}'. Aborting.")
        return pd.DataFrame()

    try:
        # Use the SQLAlchemy Inspector to safely check for table existence
        inspector = inspect(engine)
        if not inspector.has_table(table_name):
            print(f"  [WARNING] Table '{table_name}' not found. Skipping.")
            return pd.DataFrame()
        
        # Now that we know the table_name is safe, use pandas' secure reader
        df = pd.read_sql_table(table_name, engine)
        df.columns = [col.lower() for col in df.columns]
        print(f"  [SUCCESS] Loaded {table_name} data.")
        return df
    except Exception as e:
        print(f"  [ERROR] Could not load table '{table_name}'. Error: {e}")
        return pd.DataFrame()

def refresh_all_data(engine):
    """Reloads all known dataframes from the database."""
    print("\n--- Refreshing all data from database ---")
    # Use the keys from our central config as the list of tables to refresh
    dataframes = {
        table_name: load_data_safely(table_name, engine)
        for table_name in TABLE_CONFIG.keys()
    }
    print("--- Data refresh complete ---\n")
    return dataframes

    