# models/features.py
"""
Generates ML feature sets (DataFrames) for predictive models.
- Time-series data for demand forecasting.
- RFM (Recency, Frequency, Monetary) features for churn and LTV.
"""

import pandas as pd
from datetime import datetime


def _ensure_col(df: pd.DataFrame, candidates, canonical):
    """Ensure DataFrame has a canonical column name by mapping from candidates.

    candidates can be a string or iterable of possible names. If none found,
    the function leaves df unchanged.
    """
    if df is None or df.empty:
        return df
    if isinstance(candidates, str):
        candidates = [candidates]
    for c in candidates:
        if c in df.columns and canonical not in df.columns:
            df.rename(columns={c: canonical}, inplace=True)
            break
    return df

def get_daily_sales_timeseries(sales_df: pd.DataFrame, category: str = 'all', channel: str = 'all') -> pd.DataFrame:
    """
    Aggregates sales data into the Prophet-required format (ds, y) daily.
    
    Filters by category or channel if provided.
    """
    df = sales_df.copy()

    # Normalize column names to lower_snake to handle many source formats
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]

    # Accept both normalized lower_snake and legacy names via _ensure_col
    _ensure_col(df, ['timestamp', 'time', 'date'], 'timestamp')
    _ensure_col(df, ['grossvalue', 'gross_value'], 'grossvalue')
    _ensure_col(df, ['category'], 'category')
    _ensure_col(df, ['channel'], 'channel')

    if 'timestamp' not in df.columns or 'grossvalue' not in df.columns:
        return pd.DataFrame(columns=['ds', 'y'])

    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Filter if necessary
    if category != 'all' and 'category' in df.columns:
        df = df[df['category'] == category]
    if channel != 'all' and 'channel' in df.columns:
        df = df[df['channel'] == channel]

    # Aggregate to daily sum of grossvalue
    daily_sales = df.set_index('timestamp').resample('D')['grossvalue'].sum().reset_index()

    # Rename for Prophet
    daily_sales = daily_sales.rename(columns={"timestamp": "ds", "grossvalue": "y"})
    
    # Remove last day as it might be incomplete
    if len(daily_sales) > 1:
        daily_sales = daily_sales.iloc[:-1]
        
    return daily_sales


def build_rfm_features(sales_df: pd.DataFrame, customer_df: pd.DataFrame, analysis_date: datetime) -> pd.DataFrame:
    """
    Builds Recency, Frequency, and Monetary (RFM) features for churn modeling.
    Also calculates customer tenure.
    """
    sales = sales_df.copy()
    customers = customer_df.copy()

    # Normalize incoming column names to lower_snake for robust mapping
    sales.columns = [str(c).strip().lower().replace(' ', '_') for c in sales.columns]
    customers.columns = [str(c).strip().lower().replace(' ', '_') for c in customers.columns]

    # Map possible column names to canonical ones
    _ensure_col(sales, ['timestamp', 'time', 'date'], 'timestamp')
    _ensure_col(sales, ['orderid', 'order_id'], 'orderid')
    _ensure_col(sales, ['customerid', 'customer_id'], 'customerid')
    # Accept several monetary field aliases commonly used across sources
    _ensure_col(sales, ['grossvalue', 'gross_value', 'netsale', 'net_sale', 'amount', 'price', 'netamount'], 'grossvalue')

    _ensure_col(customers, ['customerid', 'customer_id'], 'customerid')
    _ensure_col(customers, ['joindate', 'join_date'], 'joindate')

    # If required columns are missing, return an empty features df with expected columns
    if sales.empty or customers.empty or 'timestamp' not in sales.columns or 'customerid' not in sales.columns:
        empty = pd.DataFrame(columns=['customerid', 'Recency', 'Frequency', 'Monetary', 'Tenure', 'Churned'])
        return empty

    sales['timestamp'] = pd.to_datetime(sales['timestamp'])
    customers['joindate'] = pd.to_datetime(customers['joindate'])

    # 1. Recency: Days since last purchase for each customer
    recency_df = sales.groupby('customerid')['timestamp'].max().reset_index()
    recency_df['Recency'] = (analysis_date - recency_df['timestamp']).dt.days

    # 2. Frequency: Count of unique orders
    if 'orderid' in sales.columns:
        frequency_df = sales.groupby('customerid')['orderid'].nunique().reset_index()
    else:
        frequency_df = sales.groupby('customerid').size().reset_index(name='order_count')
        frequency_df.columns = ['customerid', 'Frequency']
    frequency_df.columns = ['customerid', 'Frequency']

    # 3. Monetary: Sum of grossvalue
    monetary_df = sales.groupby('customerid')['grossvalue'].sum().reset_index()
    monetary_df.columns = ['customerid', 'Monetary']

    # Merge RFM
    rfm = recency_df.merge(frequency_df, on='customerid', how='left').merge(monetary_df, on='customerid', how='left')

    # Join with base customer data (ensure customerid canonical)
    customers = customers.rename(columns={c: c.lower() if c.isupper() else c for c in customers.columns})
    if 'customerid' in customers.columns:
        features_df = customers.merge(rfm, on='customerid', how='left')
    else:
        features_df = rfm.copy()

    # 4. Tenure: Days since customer joined
    # Ensure we operate on a Series; if 'joindate' missing, create a Series filled with analysis_date
    if 'joindate' in features_df.columns:
        join_series = pd.to_datetime(features_df['joindate'], errors='coerce')
    else:
        join_series = pd.Series([analysis_date] * len(features_df))

    tenure_td = analysis_date - join_series
    # tenure_td may contain NaT values; coerce to Timedelta with 0 where necessary
    features_df['Tenure'] = tenure_td.apply(lambda x: int(x.days) if pd.notna(x) else 0)

    # Handle NaNs for customers who joined but never purchased
    features_df['Recency'] = features_df['Recency'].fillna(features_df['Tenure'])
    features_df['Frequency'] = features_df['Frequency'].fillna(0)
    features_df['Monetary'] = features_df['Monetary'].fillna(0)

    # Define Churn using business heuristic of 90 days
    features_df['Churned'] = (features_df['Recency'] > 90).astype(int)

    # Normalize output column names to CamelCase expected by some UI pieces
    features_df = features_df.rename(columns={
        'customerid': 'customerid',
        'Recency': 'Recency',
        'Frequency': 'Frequency',
        'Monetary': 'Monetary',
        'Tenure': 'Tenure'
    })

    return features_df