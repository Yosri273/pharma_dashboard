# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# E-commerce Data Processing Engine - V16.0 (with Marketing Data)
#
# This version adds the ability to process and load marketing campaign and
# attribution data, enabling a full-funnel view of performance.
# -----------------------------------------------------------------------------

import pandas
from sqlalchemy import create_engine
import sys
import os
import re

# --- 1. DATABASE CONFIGURATION ---
def get_database_url():
    """Gets the correct database URL from environment variables or local fallback."""
    DATABASE_URL = os.environ.get('DATABASE_URL')
    if not DATABASE_URL:
        print("DATABASE_URL not found. Falling back to local connection.")
        DB_NAME = 'pharma_db'
        DB_USER = 'mohamedyousri'
        DB_HOST = 'localhost'
        DATABASE_URL = f"postgresql://{DB_USER}@{DB_HOST}:5432/{DB_NAME}"
    else:
        DATABASE_URL = re.sub(r"^(postgres|https)://", "postgresql://", DATABASE_URL)
        if "render.com" in DATABASE_URL and "?sslmode=require" not in DATABASE_URL:
            DATABASE_URL += "?sslmode=require"
    return DATABASE_URL

# --- 2. DEFINE DATA SCHEMAS ---
SALES_SCHEMA = {
    'orderid': ['OrderID'], 'timestamp': ['Timestamp'], 'productid': ['ProductID'],
    'productname': ['ProductName'], 'category': ['Category'], 'quantity': ['Quantity'],
    'grossvalue': ['GrossValue'], 'discountvalue': ['DiscountValue'],
    'costofgoodssold': ['CostOfGoodsSold'], 'customerid': ['CustomerID'], 'city': ['City'],
    'locationid': ['LocationID'], 'channel': ['Channel'], 'orderstatus': ['OrderStatus']
}
FUNNEL_SCHEMA = { 'week': ['Week'], 'visits': ['Visits'], 'carts': ['Carts'], 'orders': ['Orders'] }
DELIVERY_SCHEMA = {
    'deliveryid': ['DeliveryID'], 'orderid': ['OrderID'], 'orderdate': ['OrderDate'],
    'promiseddate': ['PromisedDate'], 'actualdeliverydate': ['ActualDeliveryDate'],
    'status': ['Status'], 'deliverypartner': ['DeliveryPartner'], 'city': ['City'],
    'deliverycost': ['DeliveryCost']
}
CUSTOMER_SCHEMA = { 'customerid': ['CustomerID'], 'joindate': ['JoinDate'], 'city': ['City'], 'segment': ['Segment'] }
COMPETITOR_SCHEMA = { 'date': ['Date'], 'competitor': ['Competitor'], 'productid': ['ProductID'], 'productname': ['ProductName'], 'price': ['Price'], 'onpromotion': ['OnPromotion'] }
# NEW SCHEMAS
CAMPAIGN_SCHEMA = {
    'campaignid': ['CampaignID'], 'campaignname': ['CampaignName'], 'channel': ['Channel'],
    'startdate': ['StartDate'], 'enddate': ['EndDate'], 'totalcost': ['TotalCost'],
    'impressions': ['Impressions'], 'clicks': ['Clicks']
}
ATTRIBUTION_SCHEMA = { 'orderid': ['OrderID'], 'campaignid': ['CampaignID'] }

TABLE_CONFIG = {
    "sales": {"schema": SALES_SCHEMA, "filename": "sales_data.csv"},
    "deliveries": {"schema": DELIVERY_SCHEMA, "filename": "delivery_data.csv"},
    "customers": {"schema": CUSTOMER_SCHEMA, "filename": "customer_data.csv"},
    "competitors": {"schema": COMPETITOR_SCHEMA, "filename": "competitor_data.csv"},
    "sales_funnel": {"schema": FUNNEL_SCHEMA, "filename": "funnel_data.csv"},
    "marketing_campaigns": {"schema": CAMPAIGN_SCHEMA, "filename": "marketing_campaigns.csv"}, # NEW
    "marketing_attribution": {"schema": ATTRIBUTION_SCHEMA, "filename": "marketing_attribution.csv"} # NEW
}

# --- 3. CORE LOGIC ---
def normalize_headers(df, schema):
    """Renames DataFrame columns to a clean, consistent format based on the schema."""
    header_map = {}
    for clean_name, possible_names in schema.items():
        for possible_name in possible_names:
            if possible_name in df.columns:
                header_map[possible_name] = clean_name
                break
    return df.rename(columns=header_map)

def bootstrap_database(engine):
    """Loads all master CSV files from the main directory, completely replacing tables."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for table_name, config in TABLE_CONFIG.items():
        print(f"--- Bootstrapping table: {table_name} ---")
        file_path = os.path.join(base_dir, config['filename'])
        try:
            df = pandas.read_csv(file_path)

            # 🧹 Clean the column names to avoid hidden errors
            df.columns = (
                df.columns
                .str.strip()                            # remove spaces
                .str.lower()                            # lowercase
                .str.replace('\ufeff', '', regex=True)  # remove hidden BOM chars
            )

            df = normalize_headers(df, config['schema'])
            
            # Add calculated columns after cleaning headers
            if table_name == 'sales':
                df['netsale'] = df['grossvalue'] - df['discountvalue']
            
            df.to_sql(table_name, engine, if_exists='replace', index=False)
            print(f"  [SUCCESS] Table '{table_name}' created with {len(df)} rows.")
        except FileNotFoundError:
            print(f"  [ERROR] Master file not found: {config['filename']}. Skipping.")
        except Exception as e:
            print(f"  [ERROR] An unexpected error occurred while processing {config['filename']}: {e}")

def process_incoming_file_and_append(filepath, engine):
    """
    Processes a single incoming file and appends it to the appropriate table.
    This function is called by the scheduler.
    """
    # This is a placeholder for the logic handled by the scheduler script.
    # In a more advanced architecture, this function would contain the logic
    # to read a single file, determine its type, and append it to the DB.
    print(f"Placeholder: Processing and appending file {filepath}")
    pass

# --- 4. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    print("--- Running Database Bootstrap Tool ---")
    print("This tool will completely rebuild all database tables from the master CSV files.")
    
    db_url = get_database_url()
    try:
        engine = create_engine(db_url)
        # Test the connection before proceeding
        with engine.connect() as connection:
            print("Database connection successful.")
    except Exception as e:
        print(f"\n--- DATABASE CONNECTION FAILED ---")
        print(f"Could not connect to the PostgreSQL database.")
        print(f"Attempted URL: {db_url}")
        print(f"Error details: {e}")
        sys.exit(1)

    bootstrap_database(engine)
    
    print("\n--- Database bootstrap process finished ---")

