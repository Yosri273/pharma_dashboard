import pandas as pd
import os

def get_unified_data():
    base_path = os.path.join(os.path.dirname(__file__), 'sample_csvs')
    # Load CSVs
    sales = pd.read_csv(os.path.join(base_path, 'sales_data.csv')) if os.path.exists(os.path.join(base_path, 'sales_data.csv')) else pd.DataFrame()
    customers = pd.read_csv(os.path.join(base_path, 'customer_data.csv')) if os.path.exists(os.path.join(base_path, 'customer_data.csv')) else pd.DataFrame()
    delivery = pd.read_csv(os.path.join(base_path, 'delivery_data.csv')) if os.path.exists(os.path.join(base_path, 'delivery_data.csv')) else pd.DataFrame()
    marketing = pd.read_csv(os.path.join(base_path, 'marketing_attribution.csv')) if os.path.exists(os.path.join(base_path, 'marketing_attribution.csv')) else pd.DataFrame()
    funnel = pd.read_csv(os.path.join(base_path, 'funnel_data.csv')) if os.path.exists(os.path.join(base_path, 'funnel_data.csv')) else pd.DataFrame()
    competitor = pd.read_csv(os.path.join(base_path, 'competitor_data.csv')) if os.path.exists(os.path.join(base_path, 'competitor_data.csv')) else pd.DataFrame()
    # Merge sales with customers
    if not sales.empty and not customers.empty:
        sales = sales.merge(customers, how='left', left_on='customerid', right_on='customerid')
    # Merge sales with delivery
    if not sales.empty and not delivery.empty:
        sales = sales.merge(delivery, how='left', left_on='orderid', right_on='orderid')
    # Merge sales with marketing
    if not sales.empty and not marketing.empty:
        sales = sales.merge(marketing, how='left', left_on='orderid', right_on='orderid')
    # Merge sales with funnel
    if not sales.empty and not funnel.empty:
        sales = sales.merge(funnel, how='left', left_on='orderid', right_on='orderid')
    # Merge sales with competitor
    if not sales.empty and not competitor.empty:
        sales = sales.merge(competitor, how='left', left_on='productid', right_on='productid')
    # Handle missing values
    sales.fillna(0, inplace=True)
    # Type conversions (example: dates)
    for col in sales.columns:
        if 'date' in col:
            try:
                sales[col] = pd.to_datetime(sales[col], errors='coerce')
            except Exception:
                pass
    return sales
