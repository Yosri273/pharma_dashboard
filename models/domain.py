# models/domain.py
from pydantic import BaseModel, Field
from typing import Optional

class DashboardKPIs(BaseModel):
    """
    A Pydantic model representing the key KPIs for the dashboard.
    This provides type validation and a clear data contract between
    the transformation logic and the application layer.
    """
    total_sales: float = Field(..., description="Total revenue from all sales.")
    avg_order_value: float = Field(..., description="Average value of a single order.")
    total_orders: int = Field(..., description="The total count of unique orders.")
    conversion_rate: Optional[float] = Field(None, description="The customer conversion rate.")

# Add any other business-specific models here.
# For example, you could create models for SalesOrder, Customer, etc.
# if you were building a full API, but for this app, the KPI model is key.