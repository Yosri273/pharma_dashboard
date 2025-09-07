# config/settings.py
import os
from pydantic_settings import BaseSettings
from typing import Literal

# Get the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Settings(BaseSettings):
    """
    Application-wide settings managed by Pydantic.
    Values are loaded from environment variables or a .env file.
    """
    APP_NAME: str = "E-Commerce Analytics Dashboard"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    
    # Database settings
    DATABASE_URL: str = f"sqlite:///{os.path.join(BASE_DIR, 'sales.db')}"

    # Paths to data files (relative to project root)
    # In a real enterprise app, these might be S3 URIs or API endpoints
    SALES_DATA_PATH: str = "sales_data.csv"
    CUSTOMER_DATA_PATH: str = "customer_data.csv"
    DELIVERY_DATA_PATH: str = "delivery_data.csv"
    MARKETING_CAMPAIGNS_PATH: str = "marketing_campaigns.csv"
    COMPETITOR_DATA_PATH: str = "competitor_data.csv"
    MARKETING_ATTRIBUTION_PATH: str = "marketing_attribution.csv"
    FUNNEL_DATA_PATH: str = "funnel_data.csv"

    class Config:
        # This tells Pydantic to look for a .env file in the project root
        env_file = os.path.join(BASE_DIR, ".env")
        env_file_encoding = 'utf-8'

# Create a single, importable instance of the settings
settings = Settings()