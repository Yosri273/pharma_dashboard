# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# E-commerce Data Loader - V10.1 (Docker Networking)
#
# This version updates the database connection string for consistency.
# -----------------------------------------------------------------------------

import pandas
from sqlalchemy import create_engine
import sys

# --- 1. DATABASE CONFIGURATION ---
DB_NAME = 'pharma_db'
DB_USER = 'mohamedyousri' # IMPORTANT: Change if your Mac username is different
# --- Use Docker's special DNS name for consistency, works outside container too ---
DB_HOST = 'host.docker.internal' 
DB_CONNECTION_STRING = f"postgresql://{DB_USER}@{DB_HOST}:5432/{DB_NAME}"

# --- 2. DEFINE DATA SCHEMAS AND FILE MAPPINGS ---
# (The rest of the file is unchanged)
SALES_SCHEMA = {
    'orderid': ['OrderID', 'Order ID', 'TransactionID'],
    'timestamp': ['Timestamp', 'DateTime', 'OrderDate'],
    'productid': ['ProductID', 'Product ID', 'SKU'],
    'productname': ['ProductName', 'Product Name'],
    'category': ['Category', 'ProductCategory'],
    'quantity': ['Quantity', 'Qty'],
    'price': ['Price', 'UnitPrice'],
    'city': ['City', 'LocationCity'],
    'locationid': ['LocationID', 'StoreID'],
    'grossvalue': ['GrossValue', 'Gross Sale'],
    'discountvalue': ['DiscountValue', 'Discount'],
    'customerid': ['CustomerID', 'Customer ID', 'UserID']
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
        print(f"  [WARNING] The file is missing the following required columns: {missing_cols}. They will be absent from the database.")
        
    return df[[col for col in schema.keys() if col in df.columns]]

def process_table(table_name, config, engine):
    print(f"\n--- Processing table: {table_name} ---")
    filepath = config['filepath']
    schema = config['schema']

    try:
        print(f"Reading from '{filepath}'...")
        df = pandas.read_csv(filepath)
    except FileNotFoundError:
        print(f"  [ERROR] File not found: '{filepath}'. Please make sure it is in the same folder as the script.")
        return
    except pandas.errors.EmptyDataError:
        print(f"  [ERROR] The file '{filepath}' is empty. Please ensure it contains data.")
        return
    except Exception as e:
        print(f"  [ERROR] An unexpected error occurred: {e}")
        return

    print("Normalizing column headers...")
    df = normalize_headers(df, schema)

    if table_name == 'sales':
        if 'grossvalue' in df.columns and 'discountvalue' in df.columns:
            df['netsale'] = df['grossvalue'] - df['discountvalue']
        else:
            df['netsale'] = df['price'] * df['quantity']
            
    print(f"Loading {len(df)} rows into PostgreSQL table '{table_name}'...")
    try:
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"  [SUCCESS] Table '{table_name}' loaded successfully.")
    except Exception as e:
        print(f"  [ERROR] Could not load data into PostgreSQL. Error: {e}")

# --- 4. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    try:
        engine = create_engine(DB_CONNECTION_STRING)
        with engine.connect() as connection:
            pass
    except Exception as e:
        print("--- DATABASE CONNECTION FAILED ---")
        print(f"Could not connect to the PostgreSQL database '{DB_NAME}'.")
        print("Please ensure that:")
        print("1. Postgres.app is running on your Mac.")
        print(f"2. A database named '{DB_NAME}' exists.")
        print(f"3. The username in the script ('{DB_USER}') matches your Mac username.")
        sys.exit(1)

    for table_name, config in TABLE_CONFIG.items():
        process_table(table_name, config, engine)
        
    print("\n--- Data loading process finished ---")

