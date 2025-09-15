# pharma_dashboard/app/callbacks/__init__.py

from .common import register_common_callbacks
from .sales import register_sales_callbacks
from .delivery import register_delivery_callbacks
from .customer import register_customer_callbacks
from .marketing import register_marketing_callbacks
from .profit import register_profit_callbacks
from .predictive import register_predictive_callbacks
from . import thresholds

def register_all_callbacks(app):
    """
    Registers all callbacks for the application.
    """
    register_common_callbacks(app)
    register_sales_callbacks(app)
    register_delivery_callbacks(app)
    register_customer_callbacks(app)
    register_marketing_callbacks(app)
    register_profit_callbacks(app)
    register_predictive_callbacks(app)
    # Import the thresholds callbacks module to ensure its @callback
    # decorators are executed and the modal load/save callbacks are registered.
    # The module registers callbacks on import via decorator side-effects.
    # No explicit register function is required.
    # Register comprehensive analysis callbacks
    from app.comprehensive_analysis.callbacks import register_callbacks as register_comprehensive_callbacks
    register_comprehensive_callbacks(app)

    