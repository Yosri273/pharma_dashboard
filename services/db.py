# services/db.py
import pandas as pd
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from config.settings import settings

# Setup logger
logger = logging.getLogger(__name__)

# Create a single, reusable engine. SQLAlchemy handles connection pooling.
# This engine is created once when the module is imported, using your DATABASE_URL.
try:
    engine = create_engine(settings.DATABASE_URL)
    logger.info("Database engine created successfully.")
except Exception as e:
    logger.critical(f"Failed to create database engine: {e}")
    raise

def get_db_engine() -> Engine:
    """Returns the globally configured SQLAlchemy engine."""
    return engine

def create_tables():
    """
    Executes DDL statements to create all necessary application tables.
    This uses the same PostgreSQL-compatible DDL from your original file.
    """
    # These DDL commands are compatible with PostgreSQL
    ddl_commands = [
        """
        CREATE TABLE IF NOT EXISTS sales (
            order_id TEXT PRIMARY KEY,
            order_date TIMESTAMP,
            customer_id TEXT,
            product_id TEXT,
            product_name TEXT,
            quantity INTEGER,
            price_per_unit REAL,
            total_price REAL,
            region TEXT
        );
        """,
        "CREATE TABLE IF NOT EXISTS customers (customer_id TEXT PRIMARY KEY, segment TEXT);",
        "CREATE TABLE IF NOT EXISTS delivery (order_id TEXT PRIMARY KEY, status TEXT);",
        "CREATE TABLE IF NOT EXISTS marketing_campaigns (campaign_id TEXT PRIMARY KEY, spend REAL);",
        "CREATE TABLE IF NOT EXISTS competitor_data (product_id TEXT PRIMARY KEY, competitor_price REAL);",
        "CREATE TABLE IF NOT EXISTS marketing_attribution (order_id TEXT PRIMARY KEY, channel TEXT);",
        "CREATE TABLE IF NOT EXISTS funnel (stage TEXT, count INTEGER);"
    ]
    
    try:
        # The 'with' block handles the connection and transaction automatically
        with engine.connect() as connection:
            for command in ddl_commands:
                connection.execute(text(command))
            connection.commit()
        logger.info("Database tables verified/created successfully.")
    except Exception as e:
        logger.error(f"Error creating tables: {e}", exc_info=True)
        raise

def fetch_data(query: str) -> pd.DataFrame:
    """
    Executes a SELECT query using the engine and returns a DataFrame.
    This is much cleaner and safer.
    """
    try:
        # pd.read_sql handles the connection automatically using the SQLAlchemy engine
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        logger.error(f"Error fetching data with query ({query}): {e}")
        return pd.DataFrame() # Return empty DF on failure