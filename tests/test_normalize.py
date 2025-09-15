import pandas as pd
from etl.normalize import normalize_data_dict


def test_normalize_sales_columns():
    df = pd.DataFrame({
        'Order_ID': ['o1', 'o2'],
        'Customer_ID': ['c1', 'c2'],
        'Gross Value': [100, 200],
        'Time': ['2020-01-01', '2020-01-02']
    })
    data = {'sales': df}
    out = normalize_data_dict(data)
    sales = out['sales']
    assert 'orderid' in sales.columns
    assert 'customerid' in sales.columns
    assert 'grossvalue' in sales.columns
    assert 'timestamp' in sales.columns


def test_normalize_customers_columns():
    df = pd.DataFrame({
        'Customer_ID': ['c1', 'c2'],
        'Join Date': ['2021-01-01', '2021-02-01'],
        'Segment': ['retail', 'vip']
    })
    data = {'customers': df}
    out = normalize_data_dict(data)
    cust = out['customers']
    assert 'customerid' in cust.columns
    assert 'joindate' in cust.columns
    assert 'segment' in cust.columns
