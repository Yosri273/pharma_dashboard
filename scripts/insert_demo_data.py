"""Generate rich demo datasets for comprehensive tab KPIs and insert into DB.
This script writes to the legacy *_data table names that `app.comprehensive_analysis.data_sources` expects.
It's idempotent by default: it inserts rows with a demo prefix so you can detect/remove them later.

New: supports a `--clean-demo` flag to remove demo rows (rows with 'demo_' prefixes) across text-like columns.
"""
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta, timezone
import random
import argparse

# Ensure project root on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from services.db import get_engine
from sqlalchemy import inspect, text

RND = random.Random(42)

PLATFORMS = ['Facebook', 'Google', 'Email', 'Affiliates', 'Organic', 'Direct', 'Referral', 'Social']
DEVICES = ['Desktop', 'Mobile', 'Tablet', 'App']
CITIES = ['Riyadh', 'Jeddah', 'Dammam', 'Mecca', 'Medina']

def generate_web_mobile_analytics(n_days=60, sessions_per_day=200):
    rows = []
    now = datetime.now(timezone.utc)
    for d in range(n_days):
        date = (now - timedelta(days=d)).date()
        for s in range(sessions_per_day):
            session_id = f"demo_s_{d}_{s}"
            device = RND.choice(DEVICES)
            source = RND.choice(['organic','paid','referral','direct','social','app'])
            pageviews = max(1, int(np.random.poisson(3)))
            session_duration = max(5, int(np.random.exponential(120)))
            bounce = RND.random() < 0.35
            conversion = RND.random() < 0.05
            customerid = f"demo_c_{RND.randint(1,200)}" if RND.random() < 0.6 else None
            rows.append({
                'date': date,
                'session_id': session_id,
                'user_id': customerid,
                'source': source,
                'medium': 'paid' if source=='paid' else source,
                'device': device,
                'pageviews': pageviews,
                'session_duration': session_duration,
                'bounce': bounce,
                'conversion': int(conversion)
            })
    return pd.DataFrame(rows)


def generate_ad_platform_data(num_rows=200):
    rows = []
    for i in range(num_rows):
        platform = RND.choice(['Facebook','Google','Email','Affiliates'])
        impressions = RND.randint(1000, 20000)
        clicks = int(impressions * RND.uniform(0.01, 0.08))
        spend = round(clicks * RND.uniform(0.2, 1.5),2)
        conversions = int(clicks * RND.uniform(0.01, 0.1))
        rows.append({
            'platform': platform,
            'campaign_id': f'demo_cmp_{RND.randint(1,20)}',
            'impressions': impressions,
            'clicks': clicks,
            'spend': spend,
            'conversions': conversions
        })
    return pd.DataFrame(rows)


def generate_sales(num_orders=1000):
    rows = []
    now = datetime.now(timezone.utc)
    for i in range(num_orders):
        order_id = f"demo_ord_{i}"
        ts = now - timedelta(days=RND.randint(0,60), minutes=RND.randint(0,1440))
        customer = f"demo_c_{RND.randint(1,500)}"
        product = f"P-{RND.randint(1,50)}"
        product_name = f"Demo Product {RND.randint(1,50)}"
        qty = RND.randint(1,3)
        gross = round(RND.uniform(10, 300) * qty,2)
        discount = round(gross * RND.choice([0,0.05,0.1,0.15]),2)
        netsale = gross - discount
        cogs = round(netsale * RND.uniform(0.4,0.7),2)
        city = RND.choice(CITIES)
        channel = RND.choice(['Online','App','Store'])
        orderstatus = RND.choice(['Completed','Returned','Cancelled']) if RND.random() < 0.05 else 'Completed'
        rows.append({
            'OrderID': order_id,
            'Timestamp': ts.isoformat(),
            'ProductID': product,
            'ProductName': product_name,
            'Category': 'Demo',
            'Quantity': qty,
            'GrossValue': gross,
            'DiscountValue': discount,
            'NetSale': netsale,
            'CostOfGoodsSold': cogs,
            'CustomerID': customer,
            'City': city,
            'LocationID': f'L-{RND.randint(1,10)}',
            'Channel': channel,
            'OrderStatus': orderstatus
        })
    return pd.DataFrame(rows)


def generate_customers(num_customers=500):
    rows = []
    now = datetime.now(timezone.utc)
    for i in range(num_customers):
        cid = f"demo_c_{i+1}"
        join = (now - timedelta(days=RND.randint(30,1000))).date().isoformat()
        segment = RND.choice(['New','Repeat','VIP','Loyal'])
        rows.append({'CustomerID': cid, 'JoinDate': join, 'City': RND.choice(CITIES), 'Segment': segment})
    return pd.DataFrame(rows)


def generate_deliveries(sales_df):
    rows = []
    for idx, row in sales_df.iterrows():
        orderid = row['OrderID']
        orderdate = pd.to_datetime(row['Timestamp']).date()
        promised = orderdate + timedelta(days=RND.randint(1,5))
        actual = promised + timedelta(days=RND.choice([0,0,0,1,2]))
        status = 'Delivered' if RND.random() < 0.95 else 'Delayed'
        cost = round(RND.uniform(3,15),2)
        rows.append({'DeliveryID': f'demo_del_{idx}', 'OrderID': orderid, 'OrderDate': orderdate.isoformat(), 'PromisedDate': promised.isoformat(), 'ActualDeliveryDate': actual.isoformat(), 'Status': status, 'DriverID': f'DR-{RND.randint(1,20)}', 'VehicleType': RND.choice(['Bike','Van','Truck']), 'City': row['City'], 'DeliveryCost': cost})
    return pd.DataFrame(rows)


def generate_funnel(n_weeks=10):
    rows = []
    now = datetime.now(timezone.utc)
    for w in range(n_weeks):
        week = (now - timedelta(weeks=w)).strftime('%Y-%W')
        visits = RND.randint(2000,5000)
        carts = int(visits * RND.uniform(0.05,0.15))
        orders = int(carts * RND.uniform(0.2,0.5))
        deliveries = int(orders * RND.uniform(0.9,1.0))
        rows.append({'Week': week, 'Visits': visits, 'Carts': carts, 'Orders': orders, 'Delivery': deliveries})
    return pd.DataFrame(rows)


def generate_support_tickets(num=200):
    rows = []
    now = datetime.now(timezone.utc)
    issues = ['Late Delivery', 'Damaged Item', 'Payment Issue', 'Wrong Item', 'App Bug']
    for i in range(num):
        ticket = f'demo_t_{i}'
        date = (now - timedelta(days=RND.randint(0,60))).date().isoformat()
        issue = RND.choice(issues)
        status = RND.choice(['Open','Closed','Resolved'])
        resolution = RND.randint(1,72) if status!='Open' else None
        cust = f'demo_c_{RND.randint(1,500)}'
        city = RND.choice(CITIES)
        rows.append({'ticket_id': ticket, 'date': date, 'issue_type': issue, 'status': status, 'resolution_time': resolution, 'customerid': cust, 'city': city})
    return pd.DataFrame(rows)


def generate_competitor_data(num=50):
    rows = []
    now = datetime.now(timezone.utc)
    for i in range(num):
        rows.append({'Date': (now - timedelta(days=RND.randint(0,60))).date().isoformat(), 'Competitor': RND.choice(['RivalCo','OtherCo','CopyCat']), 'ProductID': f'P-{RND.randint(1,50)}', 'ProductName': f'Demo Product {RND.randint(1,50)}', 'Price': round(RND.uniform(5,400),2), 'OnPromotion': RND.choice([True, False])})
    return pd.DataFrame(rows)


def write_to_db(engine):
    # Generate datasets
    web = generate_web_mobile_analytics(n_days=60, sessions_per_day=300)
    mobile = generate_web_mobile_analytics(n_days=60, sessions_per_day=200)
    ad = generate_ad_platform_data(500)
    sales = generate_sales(2500)
    customers = generate_customers(1000)
    deliveries = generate_deliveries(sales)
    funnel = generate_funnel(16)
    support = generate_support_tickets(800)
    competitor = generate_competitor_data(500)

    # Write to legacy table names used by loader
    targets = {
        'web_analytics': web,
        'mobile_analytics': mobile,
        'ad_platform_data': ad,
        'sales_data': sales,
        'crm_data': customers.sample(300) if len(customers)>300 else customers,
        'customer_data': customers,
        'delivery_data': deliveries,
        'funnel_data': funnel,
        'support_tickets': support,
        'competitor_data': competitor
    }

    for table, df in targets.items():
        # Attempt a safe write: only include columns that actually exist in the
        # target database table. This avoids SQL errors when demo columns don't
        # match the deployed schema.
        try:
            inspector = inspect(engine)
            if inspector.has_table(table):
                cols_info = inspector.get_columns(table)
                existing_cols = [c['name'] for c in cols_info]
                # Preserve column order where possible
                matched_cols = [c for c in df.columns if c in existing_cols]
                if not matched_cols:
                    print(f"No matching columns to write for {table}, skipping (found {len(df.columns)} demo cols, table has {len(existing_cols)} cols)")
                    continue
                df_write = df[matched_cols].copy()

                # Attempt lightweight type coercion for boolean and date-like columns
                for col_meta in cols_info:
                    col_name = col_meta['name']
                    if col_name not in df_write.columns:
                        continue
                    col_type = str(col_meta.get('type', '')).lower()
                    try:
                        if 'bool' in col_type:
                            # convert 0/1/int to booleans
                            df_write[col_name] = df_write[col_name].astype(bool)
                        elif 'date' in col_type and not pd.api.types.is_datetime64_any_dtype(df_write[col_name]):
                            df_write[col_name] = pd.to_datetime(df_write[col_name], errors='coerce').dt.date
                        elif 'timestamp' in col_type or 'datetime' in col_type:
                            df_write[col_name] = pd.to_datetime(df_write[col_name], errors='coerce')
                    except Exception:
                        # Best-effort coercion; if it fails, fall back to raw values and let DB raise
                        pass

                df_write.to_sql(table, engine, if_exists='append', index=False)
                print(f"Wrote {len(df_write)} rows to {table} (subset of {len(df)} demo rows)")
            else:
                # Table doesn't exist in DB — create it using df's schema
                df.to_sql(table, engine, if_exists='append', index=False)
                print(f"Created & wrote {len(df)} rows to new table {table}")
        except Exception as e:
            print(f"Failed to write {table}: {e}")


def clean_demo(engine, tables=None):
    """Remove demo rows from tables by deleting rows where any text-like column starts with 'demo_'.
    If `tables` is None, operate on a sensible default set used by the inserter.
    """
    inspector = inspect(engine)
    if tables is None:
        tables = ['web_analytics','mobile_analytics','ad_platform_data','sales_data','crm_data','customer_data','delivery_data','funnel_data','support_tickets','competitor_data']

    for table in tables:
        try:
            if not inspector.has_table(table):
                print(f"Table {table} not present, skipping clean")
                continue
            cols_info = inspector.get_columns(table)
            text_cols = [c['name'] for c in cols_info if any(t in str(c.get('type','')).lower() for t in ('char','text','varchar'))]
            if not text_cols:
                print(f"No text-like columns to match demo rows in {table}, skipping")
                continue
            # build WHERE clause
            clauses = [f"{col} LIKE 'demo_%'" for col in text_cols]
            where = ' OR '.join(clauses)
            sql = f"DELETE FROM {table} WHERE {where};"
            with engine.begin() as conn:
                res = conn.execute(text(sql))
                # rowcount may be DB-dependent
                rc = getattr(res, 'rowcount', None)
                print(f"Deleted {rc if rc is not None else 'unknown'} demo rows from {table}")
        except Exception as e:
            print(f"Failed cleaning {table}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Insert or clean demo data for the pharma dashboard')
    parser.add_argument('--clean-demo', action='store_true', help='Remove demo rows from target tables and exit')
    args = parser.parse_args()

    engine = get_engine()
    if args.clean_demo:
        clean_demo(engine)
        print('Demo cleanup complete.')
        sys.exit(0)

    write_to_db(engine)
    print('Demo dataset generation complete.')
