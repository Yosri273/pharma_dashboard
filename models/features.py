# models/features.py
"""
Generates ML feature sets (DataFrames) for predictive models.
- Time-series data for demand forecasting.
- RFM (Recency, Frequency, Monetary) features for churn and LTV.
"""

import pandas as pd
from datetime import datetime

def get_daily_sales_timeseries(sales_df: pd.DataFrame, category: str = 'all', channel: str = 'all') -> pd.DataFrame:
    """
    Aggregates sales data into the Prophet-required format (ds, y) daily.
    
    Filters by category or channel if provided.
    """
    df = sales_df.copy()
    
    # Ensure timestamp is datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Filter if necessary
    if category != 'all':
        df = df[df['Category'] == category]
    if channel != 'all':
        df = df[df['Channel'] == channel]

    # Aggregate to daily sum of GrossValue
    daily_sales = df.set_index('Timestamp').resample('D')['GrossValue'].sum().reset_index()
    
    # Rename for Prophet
    daily_sales = daily_sales.rename(columns={"Timestamp": "ds", "GrossValue": "y"})
    
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

    sales['Timestamp'] = pd.to_datetime(sales['Timestamp'])
    customers['JoinDate'] = pd.to_datetime(customers['JoinDate'])

    # 1. Recency: Days since last purchase for each customer
    recency_df = sales.groupby('CustomerID')['Timestamp'].max().reset_index()
    recency_df['Recency'] = (analysis_date - recency_df['Timestamp']).dt.days
    
    # 2. Frequency: Count of unique orders
    frequency_df = sales.groupby('CustomerID')['OrderID'].nunique().reset_index()
    frequency_df.columns = ['CustomerID', 'Frequency']

    # 3. Monetary: Sum of GrossValue
    monetary_df = sales.groupby('CustomerID')['GrossValue'].sum().reset_index()
    monetary_df.columns = ['CustomerID', 'Monetary']

    # Merge RFM
    rfm = recency_df.merge(frequency_df, on='CustomerID', how='left').merge(monetary_df, on='CustomerID', how='left')

    # Join with base customer data
    features_df = customers.merge(rfm, on='CustomerID', how='left')

    # 4. Tenure: Days since customer joined
    features_df['Tenure'] = (analysis_date - features_df['JoinDate']).dt.days
    
    # Handle NaNs for customers who joined but never purchased
    features_df['Recency'] = features_df['Recency'].fillna(features_df['Tenure']) # Never purchased = recency is their tenure
    features_df['Frequency'] = features_df['Frequency'].fillna(0)
    features_df['Monetary'] = features_df['Monetary'].fillna(0)

    # Define Churn. 
    # Business Logic: If a customer hasn't purchased in 90 days, they are considered churned (Target Variable).
    # This is a common heuristic; you can tune this window.
    features_df['Churned'] = (features_df['Recency'] > 90).astype(int)

    return features_df