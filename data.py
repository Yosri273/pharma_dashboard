# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Data Processing Engine - V21.2 (Vectorization Fix)
#
# This version corrects an error where a single-value utility (safe_division)
# was being incorrectly applied to entire Pandas columns (Series).
# Replaced all vectorized division calls with np.where for safe, element-wise ops.
# -----------------------------------------------------------------------------

import logging
import pandas as pd
import numpy as np  # <-- ADD THIS IMPORT
from sqlalchemy.engine import Engine
from typing import Dict
from datetime import datetime, timedelta

# Import from our central modules
from database import refresh_all_data, safe_table_exists
from utils import safe_division  # We still need this for any single-value calcs

logger = logging.getLogger(__name__)

# This global dictionary will act as an in-memory data store for the app.
DATA: Dict[str, pd.DataFrame] = {}

# --- HELPER FUNCTIONS FOR DATA ENRICHMENT ---

def _enrich_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enriches the raw sales data with calculated columns for analysis."""
    logger.info("Enriching sales data...")
    if df.empty:
        return df
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

def _enrich_delivery_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enriches raw delivery data with a proper 'date' column for filtering."""
    logger.info("Enriching delivery data...")
    if df.empty:
        return df
    df['orderdate'] = pd.to_datetime(df['orderdate'])
    df['date'] = df['orderdate'].dt.date
    df['actualdeliverydate'] = pd.to_datetime(df['actualdeliverydate'])
    df['delivery_time_days'] = (df['actualdeliverydate'] - df['orderdate']).dt.days
    df['promiseddate'] = pd.to_datetime(df['promiseddate'])
    df['on_time'] = df['actualdeliverydate'] <= df['promiseddate']
    return df

def _calculate_price_comparison(sales_df: pd.DataFrame, competitor_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates our prices vs. average competitor prices for the Market Intel tab."""
    logger.info("Calculating price comparison dataframe...")
    if sales_df.empty or competitor_df.empty:
        return pd.DataFrame()

    our_price_agg = sales_df.groupby(['productid', 'productname']).agg(
        total_netsale=('netsale', 'sum'),
        total_qty=('quantity', 'sum')
    ).reset_index()
    
    # --- FIX: Replaced safe_division with np.where for vectorized division ---
    our_price_agg['our_price'] = np.where(
        our_price_agg['total_qty'] == 0, 
        0, 
        our_price_agg['total_netsale'] / our_price_agg['total_qty']
    )
    
    comp_price_agg = competitor_df.groupby(['productid', 'productname'])['price'].mean().reset_index()
    comp_price_agg = comp_price_agg.rename(columns={'price': 'avg_competitor_price'})

    price_comparison_df = pd.merge(
        our_price_agg[['productid', 'productname', 'our_price']],
        comp_price_agg,
        on=['productid', 'productname'],
        how='outer'
    )
    
    price_comparison_df = price_comparison_df.fillna(0)
    price_comparison_df['price_difference'] = price_comparison_df['our_price'] - price_comparison_df['avg_competitor_price']
    return price_comparison_df

def _calculate_campaign_performance(sales_df: pd.DataFrame, campaigns_df: pd.DataFrame, attribution_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates ROAS and CPA for the Marketing tab."""
    logger.info("Calculating campaign performance dataframe...")
    if sales_df.empty or campaigns_df.empty or attribution_df.empty:
        return pd.DataFrame()

    sales_subset = sales_df[['orderid', 'netsale']].drop_duplicates()
    attributed_sales = pd.merge(attribution_df, sales_subset, on='orderid', how='left')

    campaign_performance = attributed_sales.groupby('campaignid').agg(
        netsale=('netsale', 'sum'),
        conversions=('orderid', 'nunique')
    ).reset_index()

    campaign_analysis_df = pd.merge(campaigns_df, campaign_performance, on='campaignid', how='left')
    
    campaign_analysis_df['netsale'] = campaign_analysis_df['netsale'].fillna(0)
    campaign_analysis_df['conversions'] = campaign_analysis_df['conversions'].fillna(0)
    
    # --- FIX: Replaced safe_division with np.where for vectorized KPIs ---
    campaign_analysis_df['roas'] = np.where(
        campaign_analysis_df['totalcost'] == 0, 
        0, 
        campaign_analysis_df['netsale'] / campaign_analysis_df['totalcost']
    )
    campaign_analysis_df['cpa'] = np.where(
        campaign_analysis_df['conversions'] == 0, 
        0, 
        campaign_analysis_df['totalcost'] / campaign_analysis_df['conversions']
    )
    
    return campaign_analysis_df

def _calculate_profit_analysis(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Creates the profit analysis dataframe based on sales data."""
    logger.info("Calculating profit analysis dataframe...")
    if sales_df.empty:
        return pd.DataFrame()
        
    profit_df = sales_df.copy()
    profit_df['net_profit'] = profit_df['netsale'] - profit_df['costofgoodssold']
    
    # --- FIX: Replaced safe_division with np.where for vectorized margin calc ---
    profit_df['profit_margin'] = np.where(
        profit_df['netsale'] == 0, 
        0, 
        (profit_df['net_profit'] / profit_df['netsale']) * 100
    )
    
    profit_df['total_cost'] = profit_df['costofgoodssold']
    
    return profit_df

def _load_prediction_data(engine: Engine) -> pd.DataFrame:
    """Safely attempts to load the customer churn predictions table."""
    logger.info("Attempting to load customer prediction data...")
    table_name = "customer_churn_predictions"
    try:
        if safe_table_exists(engine, table_name):
            df = pd.read_sql_table(table_name, engine)
            logger.info(f"Successfully loaded {len(df)} rows from '{table_name}'.")
            return df
        else:
            logger.warning(f"Prediction table '{table_name}' not found. Predictive tab will be empty.")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Could not load prediction table '{table_name}'. Error: {e}", exc_info=True)
        return pd.DataFrame()


# --- MAIN INITIALIZATION FUNCTION ---

def initialize_data(engine: Engine) -> None:
    """
    Main orchestrator to load all raw data from the database and then call the
    various enrichment and transformation functions. Results are stored in the
    global DATA dictionary.
    
    --- FIX (V21.3) ---
    Removed 'global DATA' and the re-assignment 'DATA = ...'.
    This now mutates the existing DATA dict (using .clear() and .update()) 
    so that all other modules that imported it can see the changes.
    """
    
    # 1. Load all raw data
    # Do NOT re-assign DATA. Mutate the dictionary that all other modules imported.
    raw_data = refresh_all_data(engine)
    DATA.clear()
    DATA.update(raw_data)

    # 2. Sequentially enrich and create analysis dataframes
    
    # Sales (Base for many others)
    if 'sales' in DATA:
        DATA['sales'] = _enrich_sales_data(DATA.get('sales', pd.DataFrame()))

    # Customers
    if 'customers' in DATA and 'sales' in DATA:
        DATA['customer_analysis_df'] = _calculate_customer_segments(
            DATA.get('customers', pd.DataFrame()),
            DATA.get('sales', pd.DataFrame())
        )

    # Deliveries
    if 'deliveries' in DATA:
        DATA['deliveries'] = _enrich_delivery_data(DATA.get('deliveries', pd.DataFrame()))

    # Market Intel
    if 'sales' in DATA and 'competitors' in DATA:
        DATA['price_comparison_df'] = _calculate_price_comparison(
            DATA.get('sales', pd.DataFrame()),
            DATA.get('competitors', pd.DataFrame())
        )

    # Marketing
    if 'sales' in DATA and 'marketing_campaigns' in DATA and 'marketing_attribution' in DATA:
        DATA['campaign_performance_df'] = _calculate_campaign_performance(
            DATA.get('sales', pd.DataFrame()),
            DATA.get('marketing_campaigns', pd.DataFrame()),
            DATA.get('marketing_attribution', pd.DataFrame())
        )

    # Profit
    if 'sales' in DATA:
        DATA['profit_df'] = _calculate_profit_analysis(DATA.get('sales', pd.DataFrame()))

    # Predictive
    DATA['predictions_df'] = _load_prediction_data(engine)
    
    logger.info("Data initialization and all enrichments complete. DATA dict is now populated.")