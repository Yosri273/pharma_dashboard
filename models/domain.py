# models/domain.py
from pydantic import BaseModel
from typing import Optional

class PharmaDashboardKPIs(BaseModel):
    """
    A validated data model for ALL KPIs calculated for the overview dashboard.
    This replaces passing around a large, unsafe dictionary.
    """
    total_sales: float = 0.0
    total_orders: int = 0
    aov: float = 0.0
    customer_count: int = 0
    avg_delivery_time: Optional[float] = None
    conversion_rate: Optional[float] = None
    total_spend: float = 0.0
    cpa: float = 0.0
    roas: float = 0.0
    # Backwards-compatible field expected by older callers/tests
    avg_order_value: float = 0.0


# Backwards-compatible alias expected by some tests
DashboardKPIs = PharmaDashboardKPIs