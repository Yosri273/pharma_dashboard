# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Central Configuration Module - V21.2 (Predictive Pathing Fix)
#
# Moved path constants AFTER pydantic instantiation to resolve parser conflict.
# -----------------------------------------------------------------------------

import os
import re
import logging
import sys
from pydantic_settings import BaseSettings
from typing import Dict, List, Any
from dotenv import load_dotenv  # <-- 1. ADD THIS IMPORT

# --- ADD THIS LINE ---
load_dotenv()  # This finds and loads the .env file from the project root

# --- 1. Pydantic Settings Class ---
# (Original code from config.py)
class Settings(BaseSettings):
    """Defines all application settings and their types via environment variables."""
    DB_USER: str = "mohamedyousri"
    DB_PASSWORD: str = ""
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "pharma_db"
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

    @property
    def DATABASE_URL(self) -> str:
        """Constructs the final database URL, applying necessary fixes."""
        if 'DATABASE_URL' in os.environ:
            db_url = os.environ['DATABASE_URL']
            db_url = re.sub(r'^postgres(?!\w)', 'postgresql', db_url)
            if "render.com" in db_url and "?sslmode=require" not in db_url:
                db_url += "?sslmode=require"
            return db_url
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

settings = Settings()

# --- NEW: Path Constants (Correct Location) ---
# Define module-level path constants AFTER the Pydantic class is instantiated.
# This prevents the Pydantic parser conflict error.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_STORE_PATH = os.path.join(BASE_DIR, 'model_store')


# --- 2. Centralized Data Schemas ---
# (Original code from config.py)
SALES_SCHEMA_NORM: Dict[str, List[str]] = {
    'orderid': ['OrderID', 'order_id'], 'timestamp': ['Timestamp', 'order_date'], 
    'productid': ['ProductID', 'product_sku'], 'productname': ['ProductName'],
    'category': ['Category'], 'quantity': ['Quantity', 'qty'], 'grossvalue': ['GrossValue'], 
    'discountvalue': ['DiscountValue'], 'costofgoodssold': ['CostOfGoodsSold', 'cogs'], 
    'customerid': ['CustomerID', 'user_id'], 'city': ['City', 'store_city'], 
    'locationid': ['LocationID', 'store_id'], 'channel': ['Channel'], 'orderstatus': ['OrderStatus', 'status']
}
DELIVERY_SCHEMA_NORM: Dict[str, List[str]] = {
    'deliveryid': ['DeliveryID'], 'orderid': ['OrderID'], 'orderdate': ['OrderDate'], 
    'promiseddate': ['PromisedDate'], 'actualdeliverydate': ['ActualDeliveryDate'], 
    'status': ['Status'], 'deliverypartner': ['DeliveryPartner'], 'city': ['City'], 
    'deliverycost': ['DeliveryCost']
}
CUSTOMER_SCHEMA_NORM: Dict[str, List[str]] = { 'customerid': ['CustomerID'], 'joindate': ['JoinDate'], 'city': ['City'], 'segment': ['Segment'] }
FUNNEL_SCHEMA_NORM: Dict[str, List[str]] = { 'week': ['Week'], 'visits': ['Visits'], 'carts': ['Carts'], 'orders': ['Orders'] }
COMPETITOR_SCHEMA_NORM: Dict[str, List[str]] = { 'date': ['Date'], 'competitor': ['Competitor'], 'productid': ['ProductID'], 'productname': ['ProductName'], 'price': ['Price'], 'onpromotion': ['OnPromotion'] }
CAMPAIGN_SCHEMA_NORM: Dict[str, List[str]] = {
    'campaignid': ['CampaignID'], 'campaignname': ['CampaignName'], 'channel': ['Channel'], 
    'startdate': ['StartDate'], 'enddate': ['EndDate'], 'totalcost': ['TotalCost'], 
    'impressions': ['Impressions'], 'clicks': ['Clicks']
}
ATTRIBUTION_SCHEMA_NORM: Dict[str, List[str]] = { 'orderid': ['OrderID'], 'campaignid': ['CampaignID'] }

TABLE_CONFIG: Dict[str, Dict[str, Any]] = {
    "sales": {"schema_norm": SALES_SCHEMA_NORM, "filename": "sales_data.csv", "file_prefix": "sales_"},
    "deliveries": {"schema_norm": DELIVERY_SCHEMA_NORM, "filename": "delivery_data.csv", "file_prefix": "delivery_"},
    "customers": {"schema_norm": CUSTOMER_SCHEMA_NORM, "filename": "customer_data.csv", "file_prefix": "customer_"},
    "sales_funnel": {"schema_norm": FUNNEL_SCHEMA_NORM, "filename": "funnel_data.csv", "file_prefix": "funnel_"},
    "marketing_campaigns": {"schema_norm": CAMPAIGN_SCHEMA_NORM, "filename": "marketing_campaigns.csv", "file_prefix": "campaigns_"},
    "marketing_attribution": {"schema_norm": ATTRIBUTION_SCHEMA_NORM, "filename": "marketing_attribution.csv", "file_prefix": "attribution_"}
}

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.info(f"Configuration loaded. DB Target: {settings.DATABASE_URL.split('@')[-1]}")
logger.info(f"Model storage path set to: {MODEL_STORE_PATH}") # This log will now correctly use the constant

# --- NEW: SMTP & REPORTING AUTOMATION SETTINGS ---
# These settings are used by the automated report distribution service.

# SMTP Server Configuration (Example using Gmail/Google Workspace)
# For production, NEVER hardcode these. Set them as environment variables.
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER = os.environ.get("SMTP_USER", None) # e.g., "your-service-account@your-company.com"
# For Google, this is an "App Password", not your real password
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", None) 

# Automated Report Recipient Lists
# Define recipients as a comma-separated string in environment variables
_default_recipients = "regional.manager@pharma.com,ceo@pharma.com"
REPORT_RECIPIENTS_LIST = os.environ.get("REPORT_RECIPIENTS", _default_recipients).split(",")
