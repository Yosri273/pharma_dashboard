# app/callbacks.py
import logging
from dash import Dash, Input, Output, State, no_update
import pandas as pd
from datetime import date
from dateutil.parser import parse

# Import from our new decoupled modules
from services import db
from etl import transforms
from app import layout  # Import layout to get access to plotting functions

logger = logging.getLogger(__name__)

def register_callbacks(app: Dash):
    """Registers all application callbacks."""

    @app.callback(
        [
            Output('kpi-total-sales', 'children'),
            Output('kpi-avg-order-value', 'children'),
            Output('kpi-total-orders', 'children'),
            Output('kpi-conversion-rate', 'children'),
            Output('graph-sales-over-time', 'figure'),
            Output('graph-top-products', 'figure'),
            Output('graph-sales-by-region', 'figure'),
        ],
        [Input('date-picker-range', 'start_date'),
         Input('date-picker-range', 'end_date')]
    )
    def update_dashboard(start_date_str: str, end_date_str: str):
        """
        Main callback to update the entire dashboard based on the date range.
        This function is now a 'controller' that calls services and transforms.
        """
        logger.info(f"Callback triggered. Date range: {start_date_str} to {end_date_str}")

        try:
            # 1. FETCH DATA (from services.db)
            # Fetch all data once. Filtering should happen in pandas.
            # In a larger app, you would parameterize the SQL query.
            raw_sales_df = db.fetch_data("SELECT * FROM sales")
            
            if raw_sales_df.empty:
                logger.warning("No data fetched from database. Returning empty graphs.")
                # Return empty state
                fig_empty = layout.create_time_series_chart(pd.DataFrame(), "")
                return "N/A", "N/A", "N/A", "N/A", fig_empty, fig_empty, fig_empty

            # 2. TRANSFORM DATA (from etl.transforms)
            processed_df = transforms.process_sales_data(raw_sales_df)
            
            # Filter data based on date picker
            if start_date_str and end_date_str:
                start_date = parse(start_date_str).date()
                end_date = parse(end_date_str).date()
                
                mask = (processed_df['order_date'].dt.date >= start_date) & \
                       (processed_df['order_date'].dt.date <= end_date)
                filtered_df = processed_df[mask]
            else:
                filtered_df = processed_df.copy() # Use all data if no dates

            # 3. GET BUSINESS LOGIC OUTPUTS (from etl.transforms)
            kpis = transforms.get_kpis(filtered_df)
            sales_ot_df = transforms.get_sales_over_time(filtered_df)
            top_prod_df = transforms.get_top_products(filtered_df)
            region_sales_df = transforms.get_sales_by_region(filtered_df)

            # 4. CREATE PRESENTATION (from app.layout helpers)
            # Format KPI strings using data from the Pydantic model
            kpi_sales_str = f"${kpis.total_sales:,.2f}"
            kpi_avg_val_str = f"${kpis.avg_order_value:,.2f}"
            kpi_orders_str = f"{kpis.total_orders:,}"
            kpi_conv_str = f"{kpis.conversion_rate:.1%}" if kpis.conversion_rate else "N/A"

            # Create figures
            fig_sales_ot = layout.create_time_series_chart(sales_ot_df, "Sales Over Time")
            fig_top_prod = layout.create_bar_chart(top_prod_df, 'product_name', 'total_sales', "Top 5 Products")
            fig_region_pie = layout.create_pie_chart(region_sales_df, 'region', 'total_sales', "Sales by Region")

            # 5. RETURN OUTPUTS to the layout
            return (
                kpi_sales_str,
                kpi_avg_val_str,
                kpi_orders_str,
                kpi_conv_str,
                fig_sales_ot,
                fig_top_prod,
                fig_region_pie
            )
            
        except Exception as e:
            logger.error(f"Error in update_dashboard callback: {e}", exc_info=True)
            # Return no_update to avoid crashing the dashboard on a data error
            return [no_update] * 7