"""
KPI Check Functions (Corrected and Final Version)

This module contains all specific business logic functions for checking KPIs.

CRITICAL FIXES APPLIED:
1.  Use 'pathlib' to build absolute paths to data files. This fixes the FileNotFoundError
    by no longer relying on a fragile relative path when the script is run by a scheduler.
2.  Data is now read *inside each check function* instead of once at the global level.
    This is ESSENTIAL to ensure the monitor checks against FRESH data on every run,
    not stale data loaded only at startup.
3.  Added robust error handling within each function to gracefully manage missing or empty
    data files without crashing the monitoring process.
"""
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# --- Dynamic & Robust Path Configuration ---
# Define the project root directory. This assumes the 'alerting' directory is one level
# down from the project root (e.g., pharma_dashboard/alerting/). The .parent.parent
# navigates up two levels to get to 'pharma_dashboard/'.
try:
    # This is the most reliable method when running as a script.
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # This provides a fallback if __file__ is not available (e.g., in an interactive session).
    PROJECT_ROOT = Path.cwd()

# Define absolute paths to your data files to prevent FileNotFoundError.
SALES_DATA_PATH = PROJECT_ROOT / "sales_data.csv"
DELIVERY_DATA_PATH = PROJECT_ROOT / "delivery_data.csv"
CUSTOMER_DATA_PATH = PROJECT_ROOT / "customer_data.csv"

# --- Example KPI Check Functions (Now reading fresh data on each execution) ---

def check_daily_sales_drop(percentage_drop_threshold: float) -> tuple[bool, str]:
    """
    Checks if yesterday's total sales dropped by more than the threshold
    compared to the previous 7-day average. Reads fresh data on every check.
    """
    try:
        sales_df = pd.read_csv(SALES_DATA_PATH, parse_dates=["order_date"])
    except FileNotFoundError:
        # Gracefully handle if the file doesn't exist.
        return (False, f"SKIPPED: Data file not found at {SALES_DATA_PATH}")
    except pd.errors.EmptyDataError:
        # Gracefully handle if the file is empty.
        return (False, "")

    today = pd.to_datetime(datetime.now().date())
    yesterday = today - timedelta(days=1)
    week_ago = today - timedelta(days=8)  # 7 days prior to yesterday

    sales_yesterday = sales_df[sales_df['order_date'] == yesterday]['sales_amount'].sum()
    
    sales_last_7_days = sales_df[
        (sales_df['order_date'] >= week_ago) & (sales_df['order_date'] < yesterday)
    ]
    
    # Check for empty dataframe before division
    avg_7_day_sales = sales_last_7_days['sales_amount'].sum() / 7.0 if not sales_last_7_days.empty else 0

    if avg_7_day_sales == 0:
        return (False, "")  # Avoid divide-by-zero; not enough data to compare.

    percent_change = ((sales_yesterday - avg_7_day_sales) / avg_7_day_sales) * 100.0
    
    if percent_change < -percentage_drop_threshold:
        msg = (
            f"Critical Sales Drop Detected: Yesterday's sales (${sales_yesterday:,.2f}) "
            f"were {percent_change:,.1f}% lower than the 7-day average (${avg_7_day_sales:,.2f})."
        )
        return (True, msg)
        
    return (False, "")

def check_delayed_deliveries(delay_hours_threshold: int, priority_level: str) -> tuple[bool, str]:
    """
    Checks for active deliveries with a specific priority level
    that are delayed beyond the defined hour threshold. Reads fresh data on every check.
    """
    try:
        delivery_df = pd.read_csv(DELIVERY_DATA_PATH, parse_dates=["estimated_delivery", "actual_delivery"])
    except FileNotFoundError:
        return (False, f"SKIPPED: Data file not found at {DELIVERY_DATA_PATH}")
    except pd.errors.EmptyDataError:
        return (False, "")

    now = pd.to_datetime(datetime.now())
    pending_deliveries = delivery_df[
        (delivery_df['actual_delivery'].isnull()) & 
        (delivery_df['estimated_delivery'] < now) &
        (delivery_df['priority'] == priority_level)
    ]

    if pending_deliveries.empty:
        return (False, "")

    # Create a copy to avoid a 'SettingWithCopyWarning' from pandas.
    pending_deliveries = pending_deliveries.copy()
    pending_deliveries['delay_hours'] = (now - pending_deliveries['estimated_delivery']).dt.total_seconds() / 3600
    
    critical_delays = pending_deliveries[pending_deliveries['delay_hours'] > delay_hours_threshold]
    
    count_critical = len(critical_delays)
    if count_critical > 0:
        avg_delay = critical_delays['delay_hours'].mean()
        msg = (
            f"Logistics Alert: {count_critical} '{priority_level}' priority deliveries are "
            f"delayed by more than {delay_hours_threshold} hours (Avg Delay: {avg_delay:.1f} hours)."
        )
        return (True, msg)

    return (False, "")

def check_inactive_high_value_customers(inactive_days_threshold: int, value_segment: str) -> tuple[bool, str]:
    """
    Checks for customers in a high-value segment who have not
    placed an order in the specified number of days. Reads fresh data on every check.
    """
    try:
        customer_df = pd.read_csv(CUSTOMER_DATA_PATH, parse_dates=["last_order_date"])
    except FileNotFoundError:
        return (False, f"SKIPPED: Data file not found at {CUSTOMER_DATA_PATH}")
    except pd.errors.EmptyDataError:
        return (False, "")
        
    threshold_date = datetime.now() - timedelta(days=inactive_days_threshold)
    
    inactive_target_customers = customer_df[
        (customer_df['segment'] == value_segment) &
        (customer_df['last_order_date'] < threshold_date)
    ]
    
    count_inactive = len(inactive_target_customers)
    if count_inactive > 0:
        # Get a sample of customer names for the alert body for quick reference.
        customer_list = ", ".join(inactive_target_customers['customer_name'].head(5).tolist())
        if count_inactive > 5:
            customer_list += f"... (and {count_inactive - 5} others)"

        msg = (
            f"Customer Churn Risk: {count_inactive} '{value_segment}' customers "
            f"have been inactive for over {inactive_days_threshold} days. "
            f"Customers: {customer_list}."
        )
        return (True, msg)

    return (False, "")