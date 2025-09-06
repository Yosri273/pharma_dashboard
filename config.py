# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Central Configuration Module - V20.0
#
# Single source of truth for database connections and data schemas.
# Solves code duplication and provides a robust connection URL parser.
# -----------------------------------------------------------------------------

import os
import re

# --- 1. DATABASE CONFIGURATION ---
def get_database_url():
    """
    Gets the correct database URL from environment variables, handling various
    formats and ensuring it's compatible with SQLAlchemy.
    """
    DATABASE_URL = os.environ.get('DATABASE_URL')
    if not DATABASE_URL:
        # Fallback for local development
        print("DATABASE_URL not found. Falling back to local connection.")
        DB_NAME = 'pharma_db'
        # Safely get username, default to a common value if not found
        DB_USER = os.environ.get('USER', 'mohamedyousri')
        DB_HOST = 'localhost'
        return f"postgresql://{DB_USER}@{DB_HOST}:5432/{DB_NAME}"

    # Robustly handle different URL schemes (postgres:// and postgresql://)
    DATABASE_URL = re.sub(r'^postgres(ql)?:\/\/', 'postgresql://', DATABASE_URL)

    # Add SSL requirement for Render databases
    if "render.com" in DATABASE_URL and "?sslmode=require" not in DATABASE_URL:
        DATABASE_URL += "?sslmode=require"
        
    return DATABASE_URL

# --- 2. DEFINE DATA SCHEMAS & TABLE CONFIG ---
TABLE_CONFIG = {
    "sales": {
        "schema": {
            'orderid': ['OrderID'], 'timestamp': ['Timestamp'], 'productid': ['ProductID'],
            'productname': ['ProductName'], 'category': ['Category'], 'quantity': ['Quantity'],
            'grossvalue': ['GrossValue'], 'discountvalue': ['DiscountValue'],
            'costofgoodssold': ['CostOfGoodsSold'], 'customerid': ['CustomerID'], 'city': ['City'],
            'locationid': ['LocationID'], 'channel': ['Channel'], 'orderstatus': ['OrderStatus']
        },
        "filename": "sales_data.csv", "file_prefix": "sales_"
    },
    "deliveries": {
        "schema": {
            'deliveryid': ['DeliveryID'], 'orderid': ['OrderID'], 'orderdate': ['OrderDate'],
            'promiseddate': ['PromisedDate'], 'actualdeliverydate': ['ActualDeliveryDate'],
            'status': ['Status'], 'deliverypartner': ['DeliveryPartner'], 'city': ['City'],
            'deliverycost': ['DeliveryCost']
        },
        "filename": "delivery_data.csv", "file_prefix": "delivery_"
    },
    "customers": {
        "schema": {
            'customerid': ['CustomerID'], 'joindate': ['JoinDate'],
            'city': ['City'], 'segment': ['Segment']
        },
        "filename": "customer_data.csv", "file_prefix": "customer_"
    },
    "competitors": {
        "schema": {
            'date': ['Date'], 'competitor': ['Competitor'], 'productid': ['ProductID'],
            'productname': ['ProductName'], 'price': ['Price'], 'onpromotion': ['OnPromotion']
        },
        "filename": "competitor_data.csv", "file_prefix": "competitor_"
    },
    "sales_funnel": {
        "schema": { 'week': ['Week'], 'visits': ['Visits'], 'carts': ['Carts'], 'orders': ['Orders'] },
        "filename": "funnel_data.csv", "file_prefix": "funnel_"
    },
    "marketing_campaigns": {
        "schema": {
            'campaignid': ['CampaignID'], 'campaignname': ['CampaignName'], 'channel': ['Channel'],
            'startdate': ['StartDate'], 'enddate': ['EndDate'], 'totalcost': ['TotalCost'],
            'impressions': ['Impressions'], 'clicks': ['Clicks']
        },
        "filename": "marketing_campaigns.csv", "file_prefix": "marketing_"
    },
    "marketing_attribution": {
        "schema": { 'orderid': ['OrderID'], 'campaignid': ['CampaignID'] },
        "filename": "marketing_attribution.csv", "file_prefix": "attribution_"
    }
}

