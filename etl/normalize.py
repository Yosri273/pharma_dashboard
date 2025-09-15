"""
Data normalization utilities: canonicalize column names from multiple sources so
downstream feature builders and models see consistent schema.

Provide a single entry point `normalize_data_dict(raw)` that returns a shallow
copy of the dict with DataFrames normalized in-place.
"""
from typing import Dict
import pandas as pd


def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    # normalize to lowercase underscores
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    return df


def _normalize_sales(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = _lower_cols(df)
    # common name variants -> canonical
    col_map = {
        'timestamp': 'timestamp',
        'time': 'timestamp',
        'date': 'date',
        'orderid': 'orderid',
        'order_id': 'orderid',
        'customerid': 'customerid',
        'customer_id': 'customerid',
        'productname': 'productname',
        'product_name': 'productname',
        'grossvalue': 'grossvalue',
        'gross_value': 'grossvalue',
        'netsale': 'netsale',
        'net_sale': 'netsale',
        'costofgoodssold': 'costofgoodssold',
        'cost_of_goods_sold': 'costofgoodssold',
        'discountvalue': 'discountvalue',
        'discount_value': 'discountvalue',
        'city': 'city',
        'locationid': 'locationid',
        'location_id': 'locationid',
        'channel': 'channel',
        'category': 'category'
    }
    for alt, canon in col_map.items():
        if alt in df.columns and canon not in df.columns:
            df.rename(columns={alt: canon}, inplace=True)

    # Ensure timestamp is datetime when present
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception:
            pass

    return df


def _normalize_customers(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = _lower_cols(df)
    col_map = {
        'customerid': 'customerid',
        'customer_id': 'customerid',
        'joindate': 'joindate',
        'join_date': 'joindate',
        'join_date_utc': 'joindate',
        'segment': 'segment',
        'city': 'city',
        'nps_score': 'nps_score'
    }
    for alt, canon in col_map.items():
        if alt in df.columns and canon not in df.columns:
            df.rename(columns={alt: canon}, inplace=True)
    if 'joindate' in df.columns:
        try:
            df['joindate'] = pd.to_datetime(df['joindate'])
        except Exception:
            pass
    return df


def _normalize_deliveries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = _lower_cols(df)
    col_map = {
        'orderid': 'orderid',
        'order_id': 'orderid',
        'orderdate': 'orderdate',
        'order_date': 'orderdate',
        'actualdeliverydate': 'actualdeliverydate',
        'actual_delivery_date': 'actualdeliverydate',
        'promiseddate': 'promiseddate',
        'promised_date': 'promiseddate',
        'deliverycost': 'deliverycost',
        'delivery_cost': 'deliverycost',
        'vehicletype': 'vehicletype',
        'vehicle_type': 'vehicletype',
        'driverid': 'driverid',
        'driver_id': 'driverid'
    }
    for alt, canon in col_map.items():
        if alt in df.columns and canon not in df.columns:
            df.rename(columns={alt: canon}, inplace=True)
    return df


def normalize_data_dict(raw: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Return a shallow copy of raw dict with normalized DataFrames for known keys.

    Keys handled: sales, customers, deliveries, marketing_campaigns, marketing_attribution
    """
    out = dict(raw)
    # sales
    if 'sales' in out:
        out['sales'] = _normalize_sales(out.get('sales', pd.DataFrame()))
    # customers
    if 'customers' in out:
        out['customers'] = _normalize_customers(out.get('customers', pd.DataFrame()))
    # deliveries
    if 'deliveries' in out:
        out['deliveries'] = _normalize_deliveries(out.get('deliveries', pd.DataFrame()))
    # marketing_campaigns & attribution
    if 'marketing_campaigns' in out:
        out['marketing_campaigns'] = _lower_cols(out.get('marketing_campaigns', pd.DataFrame()))
    if 'marketing_attribution' in out:
        out['marketing_attribution'] = _lower_cols(out.get('marketing_attribution', pd.DataFrame()))

    return out
