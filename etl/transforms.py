# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Data Processing Engine - V23.0 (Advanced Chart Analytics)
#
# ADDED: New function _calculate_advanced_analytics to create aggregated
#        datasets for new, complex visualizations like the Profitability
#        Waterfall and CLV by Acquisition Channel charts.
# -----------------------------------------------------------------------------

import logging
import pandas as pd
import numpy as np
from sqlalchemy.engine import Engine
from typing import Dict, Any
from datetime import datetime, timedelta

# Import from our new central modules
from services.db import refresh_all_data, safe_table_exists
from app.utils import safe_division

logger = logging.getLogger(__name__)

# This global dictionary will act as an in-memory data store for the app.
DATA: Dict[str, Any] = {}

# --- HELPER FUNCTIONS FOR DATA ENRICHMENT ---
# ... (_enrich_sales_data, _calculate_customer_segments, etc. remain unchanged) ...
def _enrich_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enriches the raw sales data with calculated columns for analysis."""
    logger.info("Enriching sales data...")
    if df.empty: return df
    df['netsale'] = df.get('grossvalue', 0) - df.get('discountvalue', 0)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['week'] = df['timestamp'].dt.to_period('W').astype(str)
    df['month'] = df['timestamp'].dt.to_period('M').astype(str)
    return df

def _calculate_customer_segments(customers_df: pd.DataFrame, sales_df: pd.DataFrame) -> pd.DataFrame:
    """Performs RFM analysis and dynamically segments customers."""
    logger.info("Calculating customer segments...")
    if customers_df.empty or sales_df.empty: return pd.DataFrame()
    customers_df['joindate'] = pd.to_datetime(customers_df['joindate'])
    rfm_df = sales_df.groupby('customerid').agg(last_purchase_date=('timestamp', 'max'), frequency=('orderid', 'nunique'), monetary=('netsale', 'sum')).reset_index()
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
    if df.empty: return df
    df['orderdate'] = pd.to_datetime(df['orderdate'])
    df['date'] = df['orderdate'].dt.date
    df['actualdeliverydate'] = pd.to_datetime(df['actualdeliverydate'])
    df['delivery_time_days'] = (df['actualdeliverydate'] - df['orderdate']).dt.days
    df['promiseddate'] = pd.to_datetime(df['promiseddate'])
    df['on_time'] = df['actualdeliverydate'] <= df['promiseddate']
    return df

def _calculate_campaign_performance(sales_df: pd.DataFrame, campaigns_df: pd.DataFrame, attribution_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates ROAS and CPA for the Marketing tab."""
    logger.info("Calculating campaign performance dataframe...")
    if sales_df.empty or campaigns_df.empty or attribution_df.empty: return pd.DataFrame()
    sales_subset = sales_df[['orderid', 'netsale']].drop_duplicates()
    attributed_sales = pd.merge(attribution_df, sales_subset, on='orderid', how='left')
    campaign_performance = attributed_sales.groupby('campaignid').agg(netsale=('netsale', 'sum'), conversions=('orderid', 'nunique')).reset_index()
    campaign_analysis_df = pd.merge(campaigns_df, campaign_performance, on='campaignid', how='left')
    campaign_analysis_df['netsale'] = campaign_analysis_df['netsale'].fillna(0)
    campaign_analysis_df['conversions'] = campaign_analysis_df['conversions'].fillna(0)
    campaign_analysis_df['roas'] = np.where(campaign_analysis_df['totalcost'] == 0, 0, campaign_analysis_df['netsale'] / campaign_analysis_df['totalcost'])
    campaign_analysis_df['cpa'] = np.where(campaign_analysis_df['conversions'] == 0, 0, campaign_analysis_df['totalcost'] / campaign_analysis_df['conversions'])
    return campaign_analysis_df

def _calculate_profit_analysis(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Creates the profit analysis dataframe based on sales data."""
    logger.info("Calculating profit analysis dataframe...")
    if sales_df.empty: return pd.DataFrame()
    profit_df = sales_df.copy()
    profit_df['net_profit'] = profit_df['netsale'] - profit_df['costofgoodssold']
    profit_df['profit_margin'] = np.where(profit_df['netsale'] == 0, 0, (profit_df['net_profit'] / profit_df['netsale']) * 100)
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

def _calculate_retention_and_synthesis_kpis(customer_analysis_df: pd.DataFrame, campaign_performance_df: pd.DataFrame, predictions_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates high-level strategic KPIs that synthesize data from multiple sources."""
    logger.info("Calculating new synthesis KPIs (Retention, CLV:CAC)...")
    kpis = {"retention_rate": 0, "repeat_purchase_rate": 0, "clv_cac_ratio": 0}
    if not customer_analysis_df.empty:
        total_customers, new_customers, active_customers = len(customer_analysis_df), (customer_analysis_df['status'] == 'New').sum(), (customer_analysis_df['status'] == 'Active').sum()
        established_base = total_customers - new_customers
        if established_base > 0: kpis["retention_rate"] = (active_customers / established_base) * 100
        customers_with_purchases = customer_analysis_df[customer_analysis_df['frequency'] > 0]
        multi_purchase_customers = customers_with_purchases[customers_with_purchases['frequency'] > 1].shape[0]
        if not customers_with_purchases.empty: kpis["repeat_purchase_rate"] = (multi_purchase_customers / len(customers_with_purchases)) * 100
    if not campaign_performance_df.empty and not predictions_df.empty:
        if 'churn_probability' in predictions_df.columns and 'Estimated_LTV' in predictions_df.columns:
            total_spend, total_conversions = campaign_performance_df['totalcost'].sum(), campaign_performance_df['conversions'].sum()
            avg_cpa = safe_division(total_spend, total_conversions)
            active_customers_mask = predictions_df['churn_probability'] <= 0.5
            avg_ltv = predictions_df.loc[active_customers_mask, 'Estimated_LTV'].mean()
            if avg_cpa > 0 and not pd.isna(avg_ltv): kpis["clv_cac_ratio"] = safe_division(avg_ltv, avg_cpa)
        else:
            logger.warning("Predictions table missing required columns. CLV:CAC calculation skipped.")
    logger.info(f"New KPI Results: {kpis}")
    return kpis

# --- NEW: Function for Advanced Chart Data ---
def _calculate_advanced_analytics(
    sales_df: pd.DataFrame,
    delivery_df: pd.DataFrame,
    campaign_perf_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    attribution_df: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Creates aggregated DataFrames for complex, cross-domain charts.
    """
    logger.info("Calculating advanced analytics for new charts...")
    analytics = {}

    # 1. Data for Profitability Waterfall
    if not sales_df.empty and not delivery_df.empty and not campaign_perf_df.empty:
        gross_revenue = sales_df['grossvalue'].sum()
        total_discounts = sales_df['discountvalue'].sum()
        total_cogs = sales_df['costofgoodssold'].sum()
        total_delivery_cost = delivery_df['deliverycost'].sum()
        total_marketing_spend = campaign_perf_df['totalcost'].sum()
        
        analytics['waterfall_data'] = pd.DataFrame({
            'measure': ['Gross Revenue', 'Discounts', 'COGS', 'Delivery Costs', 'Marketing Spend', 'Net Profit'],
            'amount': [gross_revenue, -total_discounts, -total_cogs, -total_delivery_cost, -total_marketing_spend, 0] # Net profit is calculated in waterfall chart
        })
    
    # 2. Data for CLV by Acquisition Channel
    if not predictions_df.empty and not attribution_df.empty and not campaign_perf_df.empty:
        if 'Estimated_LTV' in predictions_df.columns:
            # Join attribution to campaigns to get channel, then to predictions to get LTV
            attr_campaigns = pd.merge(attribution_df, campaign_perf_df[['campaignid', 'channel']], on='campaignid')
            # Each order might have multiple attributions, get first touchpoint channel
            first_touch_channel = attr_campaigns.drop_duplicates('orderid', keep='first')
            # Join this with sales to get customerid
            order_customers = pd.merge(first_touch_channel, sales_df[['orderid', 'customerid']].drop_duplicates(), on='orderid')
            # Finally, join with predictions to get LTV
            customer_ltv = pd.merge(order_customers, predictions_df[['customerid', 'Estimated_LTV']], on='customerid')
            
            analytics['clv_by_channel'] = customer_ltv.groupby('channel')['Estimated_LTV'].mean().reset_index().sort_values('Estimated_LTV', ascending=False)

    logger.info(f"Advanced analytics calculation complete. Generated data for: {list(analytics.keys())}")
    return analytics

# --- MAIN INITIALIZATION FUNCTION (MODIFIED) ---
def initialize_data(engine: Engine) -> None:
    """
    Main orchestrator to load all raw data and call transformation functions.
    """
    raw_data = refresh_all_data(engine)
    DATA.clear()
    DATA.update(raw_data)

    if 'sales' in DATA:
        DATA['sales'] = _enrich_sales_data(DATA.get('sales', pd.DataFrame()))
    if 'customers' in DATA and 'sales' in DATA:
        DATA['customer_analysis_df'] = _calculate_customer_segments(DATA.get('customers', pd.DataFrame()), DATA.get('sales', pd.DataFrame()))
    if 'deliveries' in DATA:
        DATA['deliveries'] = _enrich_delivery_data(DATA.get('deliveries', pd.DataFrame()))
    if 'sales' in DATA and 'marketing_campaigns' in DATA and 'marketing_attribution' in DATA:
        DATA['campaign_performance_df'] = _calculate_campaign_performance(DATA.get('sales', pd.DataFrame()), DATA.get('marketing_campaigns', pd.DataFrame()), DATA.get('marketing_attribution', pd.DataFrame()))
    if 'sales' in DATA:
        DATA['profit_df'] = _calculate_profit_analysis(DATA.get('sales', pd.DataFrame()))
    
    DATA['predictions_df'] = _load_prediction_data(engine)
    
    DATA['synthesis_kpis'] = _calculate_retention_and_synthesis_kpis(
        DATA.get('customer_analysis_df', pd.DataFrame()),
        DATA.get('campaign_performance_df', pd.DataFrame()),
        DATA.get('predictions_df', pd.DataFrame())
    )
    
    # --- NEW: Call advanced analytics function ---
    DATA['advanced_analytics'] = _calculate_advanced_analytics(
        DATA.get('sales', pd.DataFrame()),
        DATA.get('deliveries', pd.DataFrame()),
        DATA.get('campaign_performance_df', pd.DataFrame()),
        DATA.get('predictions_df', pd.DataFrame()),
        DATA.get('marketing_attribution', pd.DataFrame())
    )
    
    logger.info("Data initialization and all enrichments complete. DATA dict is now populated.")

