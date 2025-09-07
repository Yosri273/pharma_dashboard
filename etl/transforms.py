# etl/transforms.py
import pandas as pd
from models.domain import DashboardKPIs

def process_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies core transformations to the raw sales DataFrame.
    Converts types, handles NaNs, and engineers new features.
    """
    if df.empty:
        return pd.DataFrame()
        
    df_processed = df.copy()
    df_processed['order_date'] = pd.to_datetime(df_processed['order_date'])
    df_processed['total_price'] = pd.to_numeric(df_processed['total_price'], errors='coerce')
    df_processed.dropna(subset=['order_date', 'total_price'], inplace=True)
    
    # Example feature: 'month_year' for aggregation
    df_processed['month_year'] = df_processed['order_date'].dt.to_period('M').astype(str)
    return df_processed

def get_kpis(df: pd.DataFrame) -> DashboardKPIs:
    """
    Calculates key performance indicators from the processed sales data.
    Returns a validated DashboardKPIs domain model.
    """
    if df.empty:
        # Return default values in the model
        return DashboardKPIs(total_sales=0.0, avg_order_value=0.0, total_orders=0)

    total_sales = df['total_price'].sum()
    total_orders = df['order_id'].nunique()
    avg_order_value = total_sales / total_orders if total_orders > 0 else 0.0
    
    # Placeholder for conversion rate if funnel data were joined
    # conversion_rate = ... 

    return DashboardKPIs(
        total_sales=total_sales,
        avg_order_value=avg_order_value,
        total_orders=total_orders
        # conversion_rate=conversion_rate
    )

def get_sales_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates sales data by date for time-series plotting.
    """
    if df.empty:
        return pd.DataFrame(columns=['order_date', 'total_price'])
        
    sales_ot = df.groupby(df['order_date'].dt.date)['total_price'].sum().reset_index()
    sales_ot.columns = ['order_date', 'total_sales'] # Rename for clarity
    return sales_ot.sort_values(by='order_date')

def get_top_products(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Finds the top N products by total sales revenue.
    """
    if df.empty:
        return pd.DataFrame(columns=['product_name', 'total_sales'])

    top_prod = df.groupby('product_name')['total_price'].sum().nlargest(top_n).reset_index()
    top_prod.columns = ['product_name', 'total_sales']
    return top_prod

def get_sales_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates sales data by region for pie chart plotting.
    """
    if df.empty:
        return pd.DataFrame(columns=['region', 'total_sales'])
        
    region_sales = df.groupby('region')['total_price'].sum().reset_index()
    region_sales.columns = ['region', 'total_sales']
    return region_sales