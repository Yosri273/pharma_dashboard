# tests/test_transforms.py
import pytest
import pandas as pd
from datetime import datetime
from etl import transforms
from models.domain import DashboardKPIs

# A pytest fixture creates reusable test data
@pytest.fixture
def sample_sales_df() -> pd.DataFrame:
    """Creates a sample processed DataFrame for testing transforms."""
    data = {
        'order_id': ['A1', 'A2', 'A3'],
        'order_date': [
            datetime(2023, 1, 1), 
            datetime(2023, 1, 5), 
            datetime(2023, 1, 10)
        ],
        'product_name': ['Widget A', 'Widget B', 'Widget A'],
        'total_price': [100.0, 50.5, 150.0],
        'region': ['North', 'South', 'North'],
    }
    return pd.DataFrame(data)

def test_get_kpis(sample_sales_df):
    """Tests the get_kpis transform function."""
    kpis = transforms.get_kpis(sample_sales_df)
    
    assert isinstance(kpis, DashboardKPIs)
    assert kpis.total_sales == pytest.approx(300.5)
    assert kpis.total_orders == 3
    assert kpis.avg_order_value == pytest.approx(100.16666)

def test_get_kpis_empty_df():
    """Tests that get_kpis handles an empty DataFrame gracefully."""
    empty_df = pd.DataFrame(columns=['order_id', 'total_price'])
    kpis = transforms.get_kpis(empty_df)
    
    assert isinstance(kpis, DashboardKPIs)
    assert kpis.total_sales == 0.0
    assert kpis.total_orders == 0
    assert kpis.avg_order_value == 0.0

def test_get_top_products(sample_sales_df):
    """Tests the top products aggregation."""
    top_prod = transforms.get_top_products(sample_sales_df, top_n=2)
    
    assert len(top_prod) == 2
    assert top_prod.iloc[0]['product_name'] == 'Widget A'
    assert top_prod.iloc[0]['total_sales'] == 250.0
    assert top_prod.iloc[1]['product_name'] == 'Widget B'

def test_get_sales_by_region(sample_sales_df):
    """Tests the region aggregation."""
    region_sales = transforms.get_sales_by_region(sample_sales_df)
    
    assert len(region_sales) == 2
    north_sales = region_sales[region_sales['region'] == 'North']['total_sales'].values[0]
    south_sales = region_sales[region_sales['region'] == 'South']['total_sales'].values[0]
    
    assert north_sales == 250.0
    assert south_sales == 50.5