# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# E-commerce Data Loader - V12.0 (Secure Cloud Ready)
#
# This version is optimized for cloud deployment. It automatically handles
# SSL connections for secure cloud databases like Render.
# -----------------------------------------------------------------------------

import pandas
from sqlalchemy import create_engine
import sys
import os

# --- 1. DATABASE CONFIGURATION ---
DATABASE_URL = os.environ.get('DATABASE_URL')

if not DATABASE_URL:
    # Fallback for local development if the environment variable isn't set
    print("DATABASE_URL not found. Falling back to local Docker connection.")
    DB_NAME = 'pharma_db'
    DB_USER = 'mohamedyousri'
    DB_HOST = 'host.docker.internal'
    DATABASE_URL = f"postgresql://{DB_USER}@{DB_HOST}:5432/{DB_NAME}"
else:
    # When connecting to a cloud provider, ensure SSL is used.
    # Heroku and Render need this for external connections.
    if "render.com" in DATABASE_URL or "heroku.com" in DATABASE_URL:
        DATABASE_URL += "?sslmode=require"

# --- 2. DEFINE DATA SCHEMAS AND FILE MAPPINGS ---
SALES_SCHEMA = {
    'orderid': ['OrderID', 'Order ID', 'TransactionID'],
    'timestamp': ['Timestamp', 'DateTime', 'OrderDate'],
    'productid': ['ProductID', 'Product ID', 'SKU'],
    'productname': ['ProductName', 'Product Name'],
    'category': ['Category', 'ProductCategory'],
    'quantity': ['Quantity', 'Qty'],
    'grossvalue': ['GrossValue', 'Gross Sale'],
    'discountvalue': ['DiscountValue', 'Discount'],
    'customerid': ['CustomerID', 'Customer ID', 'UserID'],
    'city': ['City', 'LocationCity'],
    'locationid': ['LocationID', 'StoreID']
}

DELIVERY_SCHEMA = {
    'deliveryid': ['DeliveryID', 'ShipmentID'],
    'orderid': ['OrderID', 'Order ID'],
    'orderdate': ['OrderDate', 'DateOrdered'],
    'promiseddate': ['PromisedDate', 'Promised Delivery Date'],
    'actualdeliverydate': ['ActualDeliveryDate', 'DeliveredOn'],
    'status': ['Status', 'DeliveryStatus'],
    'deliverypartner': ['DeliveryPartner', 'Carrier'],
    'city': ['City', 'DeliveryCity']
}

CUSTOMER_SCHEMA = {
    'customerid': ['CustomerID', 'Customer ID', 'UserID'],
    'joindate': ['JoinDate', 'RegistrationDate'],
    'city': ['City', 'CustomerCity'],
    'segment': ['Segment', 'CustomerSegment']
}

COMPETITOR_SCHEMA = {
    'date': ['Date', 'SnapshotDate'],
    'competitor': ['Competitor', 'CompetitorName'],
    'productid': ['ProductID', 'SKU'],
    'productname': ['ProductName'],
    'price': ['Price', 'CompetitorPrice'],
    'onpromotion': ['OnPromotion', 'IsPromoted']
}

TABLE_CONFIG = {
    "sales": {"schema": SALES_SCHEMA, "filepath": "sales_data.csv"},
    "deliveries": {"schema": DELIVERY_SCHEMA, "filepath": "delivery_data.csv"},
    "customers": {"schema": CUSTOMER_SCHEMA, "filepath": "customer_data.csv"},
    "competitors": {"schema": COMPETITOR_SCHEMA, "filepath": "competitor_data.csv"}
}

# --- 3. DATA PROCESSING FUNCTIONS ---

def normalize_headers(df, schema):
    header_map = {}
    for clean_name, possible_names in schema.items():
        for possible_name in possible_names:
            if possible_name in df.columns:
                header_map[possible_name] = clean_name
                break 
    df = df.rename(columns=header_map)
    missing_cols = set(schema.keys()) - set(df.columns)
    if missing_cols:
        print(f"  [WARNING] The file is missing required columns: {missing_cols}.")
    return df[[col for col in schema.keys() if col in df.columns]]

def process_table(table_name, config, engine):
    print(f"\n--- Processing table: {table_name} ---")
    filepath = config['filepath']
    schema = config['schema']
    try:
        print(f"Reading from '{filepath}'...")
        df = pandas.read_csv(filepath)
    except FileNotFoundError:
        print(f"  [ERROR] File not found: '{filepath}'.")
        return
    except pandas.errors.EmptyDataError:
        print(f"  [ERROR] The file '{filepath}' is empty.")
        return
    except Exception as e:
        print(f"  [ERROR] An unexpected error occurred reading the file: {e}")
        return

    print("Normalizing column headers...")
    df = normalize_headers(df, schema)

    if table_name == 'sales':
        if 'grossvalue' in df.columns and 'discountvalue' in df.columns:
            df['grossvalue'] = pandas.to_numeric(df['grossvalue'], errors='coerce')
            df['discountvalue'] = pandas.to_numeric(df['discountvalue'], errors='coerce')
            df['netsale'] = df['grossvalue'] - df['discountvalue']
            
    print(f"Loading {len(df)} rows into PostgreSQL table '{table_name}'...")
    try:
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"  [SUCCESS] Table '{table_name}' loaded successfully.")
    except Exception as e:
        print(f"  [ERROR] Could not load data into PostgreSQL. Error: {e}")

# --- 4. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    try:
        print(f"Attempting to connect to database...")
        engine = create_engine(DATABASE_URL)
        with engine.connect() as connection:
            pass
        print("Database connection successful.")
    except Exception as e:
        print("\n--- DATABASE CONNECTION FAILED ---")
        print("Could not connect to the database. See error below:")
        print(f"ERROR DETAILS: {e}")
        sys.exit(1)

    for table_name, config in TABLE_CONFIG.items():
        process_table(table_name, config, engine)
        
    print("\n--- Data loading process finished ---")

