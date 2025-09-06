# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Data Processing Engine - V21.0 (Final Master)
#
# This module is the central engine for all data loading, preparation, and
# enrichment. It is designed to be modular, resilient, and scalable.
# -----------------------------------------------------------------------------

import logging
import pandas as pd
from sqlalchemy.engine import Engine
from typing import Dict
from datetime import datetime, timedelta

# Import from our central modules
from database import refresh_all_data
from utils import safe_division

logger = logging.getLogger(__name__)

# This global dictionary will act as an in-memory data store for the app.
DATA: Dict[str, pd.DataFrame] = {}

# --- 2. DATA ENRICHMENT & TRANSFORMATION FUNCTIONS ---

def _enrich_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enriches the raw sales data with calculated columns for analysis."""
    logger.info("Enriching sales data...")
    if df.empty:
        return df
    
    # FIX (J): Use .get() for safer column access to prevent KeyErrors
    df['netsale'] = df.get('grossvalue', 0) - df.get('discountvalue', 0)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['week'] = df['timestamp'].dt.to_period('W').astype(str)
    df['month'] = df['timestamp'].dt.to_period('M').astype(str)
    return df

def _calculate_customer_segments(customers_df: pd.DataFrame, sales_df: pd.DataFrame) -> pd.DataFrame:
    """Performs RFM analysis and dynamically segments customers."""
    logger.info("Calculating customer segments...")
    if customers_df.empty or sales_df.empty:
        return pd.DataFrame()

    customers_df['joindate'] = pd.to_datetime(customers_df['joindate'])
    
    # FIX (F): Added missing .reset_index() to the RFM calculation
    rfm_df = sales_df.groupby('customerid').agg(
        last_purchase_date=('timestamp', 'max'),
        frequency=('orderid', 'nunique'),
        monetary=('netsale', 'sum')
    ).reset_index()
    
    current_date = datetime.now()
    rfm_df['recency'] = (current_date - rfm_df['last_purchase_date']).dt.days
    
    analysis_df = pd.merge(customers_df, rfm_df, on='customerid', how='left')

    def get_status(row):
        join_recency = (current_date - row['joindate']).days
        if join_recency <= 90: return 'New'
        if pd.isna(row['recency']): return 'Never Purchased'
        if row['recency'] <= 90: return 'Active'
        if 90 < row['recency'] <= 180: return 'Dormant (At-Risk)'
        return 'Churn Risk'

    analysis_df['status'] = analysis_df.apply(get_status, axis=1)
    return analysis_df

# --- 3. MAIN INITIALIZATION FUNCTION ---

def initialize_data(engine: Engine) -> None:
    """
    Main orchestrator to load all raw data from the database and then call the
    various enrichment and transformation functions. Results are stored in the
    global DATA dictionary.
    """
    global DATA
    DATA = refresh_all_data(engine)

    # Sequentially enrich and create analysis dataframes
    DATA['sales'] = _enrich_sales_data(DATA.get('sales', pd.DataFrame()))
    DATA['customer_analysis_df'] = _calculate_customer_segments(
        DATA.get('customers', pd.DataFrame()),
        DATA.get('sales', pd.DataFrame())
    )
    # In a full enterprise app, other enrichment functions for profit,
    # marketing, etc., would be called here.

