# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Interactive Callbacks Module - V23.1 (Predictive Fallback Update)
#
# Replaced churn dash callback with a master layout-rendering callback
# that handles model checking, fallback UI, and a new training trigger.
# Added synthetic data generation and in-app training job function.
# -----------------------------------------------------------------------------

import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback_context, dcc, html, dash_table, callback, no_update
from dash.exceptions import PreventUpdate
from datetime import datetime, timedelta
import numpy as np
import joblib
import os
import sys
import random # For synthetic data

# Import from our new central modules
from etl.transforms import DATA, initialize_data  # FIX: Path updated
from services.db import get_engine                  # FIX: Path updated
from app.utils import create_kpi_body, create_placeholder_figure # FIX: Path updated
from app.layout import create_kpi_card, create_graph_card, create_datatable_card # Import layout helpers

# --- MODIFIED IMPORT FOR PDF REPORTING ---
# Replaced specific functions with the single generic generator
from app.reporting import generate_pdf_report
# --- END MODIFIED IMPORT ---

# --- NEW IMPORTS FOR PREDICTIVE ANALYTICS ---
from services.storage import load_model_artifact
from models.predictors import DemandForecaster, ChurnPredictor # Need class definitions
from models.features import build_rfm_features

logger = logging.getLogger(__name__)

# --- NEW: DEFINE MODEL STORAGE PATHS ---
# Add project root to path to allow loading modules like 'models'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODEL_STORE_PATH = os.path.join(PROJECT_ROOT, "model_store")
if not os.path.exists(MODEL_STORE_PATH):
    os.makedirs(MODEL_STORE_PATH)
    logger.info(f"Created model store directory: {MODEL_STORE_PATH}")

CHURN_MODEL_PATH = os.path.join(MODEL_STORE_PATH, 'churn_predictor_main.joblib')
CHURN_METRICS_PATH = os.path.join(MODEL_STORE_PATH, 'churn_metrics.joblib')
FORECAST_MODEL_PATH = os.path.join(MODEL_STORE_PATH, 'demand_forecaster_main.joblib')


def register_callbacks(app):
    """Registers all application callbacks."""

    # --- MAIN CALLBACKS ---
    # (toggle_navbar_collapse, handle_refresh, render_tab_content... remain unchanged)
    
    @callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
    def toggle_navbar_collapse(n, is_open):
        """Callback to toggle the mobile hamburger menu."""
        if n:
            return not is_open
        return is_open

    @app.callback(
        Output('data-store-trigger', 'data'),
        Input('refresh-data-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def handle_refresh(n_clicks):
        """Refreshes all data from the database when the refresh button is clicked."""
        logger.info("Refresh data button clicked. Re-initializing all data.")
        engine = get_engine()
        initialize_data(engine)
        return "refreshed"

    @app.callback(Output('tab-content', 'children'), Input('tabs-controller', 'active_tab'))
    def render_tab_content(active_tab):
        """Renders the content for the selected tab."""
        # FIX: Path updated
        from app.layout import (create_sales_layout, create_delivery_layout, create_customer_layout,
                             create_competitor_layout, create_marketing_layout, create_profit_layout,
                             create_predictive_layout)
        layouts = {
            "sales-tab": create_sales_layout, "delivery-tab": create_delivery_layout,
            "customer-tab": create_customer_layout,
            "marketing-tab": create_marketing_layout, "profit-tab": create_profit_layout,
            "predictive-tab": create_predictive_layout  # This now points to our new advanced layout
        }
        return layouts.get(active_tab, lambda: html.H4("Tab not found."))()


    # --- ALL OTHER DASHBOARD CALLBACKS (SALES, DELIVERY, CUSTOMER, ETC) ---
    
    # --- Sales Dashboard Callback (Unchanged from prev step) ---
    @app.callback(
        [Output('kpi-total-revenue', 'children'), Output('kpi-gross-margin', 'children'),
         Output('kpi-net-profit', 'children'), Output('kpi-total-orders', 'children'),
         Output('kpi-aov', 'children'), Output('kpi-return-rate', 'children'),
         Output('sales-funnel-chart', 'figure'), Output('sales-over-time-chart', 'figure'),
         Output('sales-by-category-chart', 'figure'), Output('top-products-chart', 'figure'),
         Output('sales-by-channel-chart', 'figure'), Output('sales-by-city-chart', 'figure'),
         Output('sales-by-branch-chart', 'figure')],
        [Input('sales-apply-btn', 'n_clicks')],
        [State('channel-filter-dropdown', 'value'),
         State('sales-date-picker', 'start_date'),
         State('sales-date-picker', 'end_date'),
         State('time-agg-selector', 'value'),
         State('sales-region-filter', 'value'),
         State('sales-category-filter', 'value')]
    )
    def update_sales_dashboard(n_clicks, selected_channel, start_date, end_date, time_agg, selected_regions, selected_categories):
        if n_clicks is None:
            raise PreventUpdate

        sales_df = DATA.get('sales', pd.DataFrame())
        funnel_df = DATA.get('sales_funnel', pd.DataFrame())
        if sales_df.empty or not start_date or not end_date:
            raise PreventUpdate

        # --- Filtering Logic ---
        start_date_obj, end_date_obj = pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()
        date_mask = (sales_df['date'] >= start_date_obj) & (sales_df['date'] <= end_date_obj)
        channel_mask = (sales_df['channel'] == selected_channel) if selected_channel != 'All' else True
        
        region_mask_active = 'All' not in selected_regions and bool(selected_regions)
        region_mask = sales_df['city'].isin(selected_regions) if region_mask_active else True

        category_mask_active = 'All' not in selected_categories and bool(selected_categories)
        category_mask = sales_df['category'].isin(selected_categories) if category_mask_active else True

        filtered_sales = sales_df.loc[date_mask & channel_mask & region_mask & category_mask]
        
        if filtered_sales.empty:
            empty_kpi = create_kpi_body("No Data", "-")
            placeholder_fig = create_placeholder_figure("No data for selected filters")
            return [empty_kpi]*6 + [placeholder_fig]*7

        # --- KPI Calculation ---
        total_revenue = filtered_sales['netsale'].sum()
        total_cogs = filtered_sales['costofgoodssold'].sum()
        net_profit = total_revenue - total_cogs
        gross_margin = (net_profit / total_revenue * 100) if total_revenue > 0 else 0
        total_orders = filtered_sales['orderid'].nunique()
        aov = total_revenue / total_orders if total_orders > 0 else 0
        returned_orders = filtered_sales[filtered_sales['orderstatus'] == 'Returned']['orderid'].nunique()
        return_rate = (returned_orders / total_orders * 100) if total_orders > 0 else 0

        kpi_revenue = create_kpi_body("Total Revenue", f"{total_revenue:,.2f} SAR")
        kpi_margin = create_kpi_body("Gross Margin", f"{gross_margin:.2f}%")
        kpi_profit = create_kpi_body("Net Profit", f"{net_profit:,.2f} SAR")
        kpi_orders = create_kpi_body("Total Orders", f"{total_orders:,}")
        kpi_aov = create_kpi_body("Avg Order Value", f"{aov:,.2f} SAR")
        kpi_return = create_kpi_body("Return Rate", f"{return_rate:.2f}%")

        # --- Figure Generation ---
        funnel_fig = create_placeholder_figure("Funnel Data Not Available")
        unfiltered_view = not (selected_channel != 'All' or region_mask_active or category_mask_active)
        if not funnel_df.empty and unfiltered_view:
            completed = filtered_sales[filtered_sales['orderstatus'] == 'Completed']['orderid'].nunique()
            funnel_fig = go.Figure(go.Funnel(
                y=["Visits", "Carts", "Total Orders", "Fulfilled"], 
                x=[funnel_df['visits'].sum(), funnel_df['carts'].sum(), total_orders, completed], 
                textinfo="value+percent initial"
            )).update_layout(title_text="Sales Funnel")
        elif not unfiltered_view:
             funnel_fig = create_placeholder_figure("Funnel view disabled when filters are active")

        agg_col = 'date' if time_agg not in ['week', 'month'] else time_agg
        time_grouped = filtered_sales.groupby(agg_col)['netsale'].sum().reset_index()
        sales_over_time_fig = px.line(time_grouped, x=agg_col, y='netsale', title=f'Net Sales Trend ({time_agg.capitalize()})')
        
        category_sales = filtered_sales.groupby('category')['netsale'].sum().reset_index()
        sales_by_cat_fig = px.pie(category_sales, names='category', values='netsale', title='Sales by Category', hole=0.3)
        
        product_sales = filtered_sales.groupby('productname')['netsale'].sum().nlargest(10).reset_index()
        top_prod_fig = px.bar(product_sales, x='netsale', y='productname', orientation='h', title='Top 10 Products').update_layout(yaxis={'categoryorder':'total ascending'})
        
        channel_sales = filtered_sales.groupby('channel')['netsale'].sum().reset_index()
        sales_by_channel_fig = px.pie(channel_sales, names='channel', values='netsale', title='Sales by Channel', hole=0.3)
        
        city_sales = filtered_sales.groupby('city')['netsale'].sum().nlargest(10).reset_index()
        sales_by_city_fig = px.bar(city_sales, x='netsale', y='city', orientation='h', title='Top 10 Cities by Sales').update_layout(yaxis={'categoryorder':'total ascending'})
        
        branch_sales = filtered_sales.groupby('locationid')['netsale'].sum().nlargest(10).reset_index()
        sales_by_branch_fig = px.bar(branch_sales, x='netsale', y='locationid', orientation='h', title='Top 10 Pharmacy Branches by Sales').update_layout(yaxis={'categoryorder':'total ascending'})

        return kpi_revenue, kpi_margin, kpi_profit, kpi_orders, kpi_aov, kpi_return, funnel_fig, sales_over_time_fig, sales_by_cat_fig, top_prod_fig, sales_by_channel_fig, sales_by_city_fig, sales_by_branch_fig

    # --- MODIFIED SALES EXPORT CALLBACK ---
    @app.callback(
        Output('download-dashboard-pdf', 'data', allow_duplicate=True),
        Input('sales-export-btn', 'n_clicks'),
        [State('channel-filter-dropdown', 'value'),
         State('sales-date-picker', 'start_date'),
         State('sales-date-picker', 'end_date'),
         State('time-agg-selector', 'value'),
         State('sales-region-filter', 'value'),
         State('sales-category-filter', 'value')],
        prevent_initial_call=True
    )
    def export_sales_pdf(n_clicks, selected_channel, start_date, end_date, time_agg, selected_regions, selected_categories):
        if n_clicks is None:
            raise PreventUpdate

        sales_df = DATA.get('sales', pd.DataFrame())
        if sales_df.empty or not start_date or not end_date:
            raise PreventUpdate

        # --- 1. (DUPLICATED) Filtering Logic ---
        start_date_obj, end_date_obj = pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()
        date_mask = (sales_df['date'] >= start_date_obj) & (sales_df['date'] <= end_date_obj)
        channel_mask = (sales_df['channel'] == selected_channel) if selected_channel != 'All' else True
        
        region_mask_active = 'All' not in selected_regions and bool(selected_regions)
        region_mask = sales_df['city'].isin(selected_regions) if region_mask_active else True

        category_mask_active = 'All' not in selected_categories and bool(selected_categories)
        category_mask = sales_df['category'].isin(selected_categories) if category_mask_active else True
        
        filtered_sales = sales_df.loc[date_mask & channel_mask & region_mask & category_mask]

        if filtered_sales.empty:
            raise PreventUpdate

        # --- 2. (DUPLICATED) KPI Calculation ---
        total_revenue = filtered_sales['netsale'].sum()
        total_cogs = filtered_sales['costofgoodssold'].sum()
        net_profit = total_revenue - total_cogs
        gross_margin = (net_profit / total_revenue * 100) if total_revenue > 0 else 0
        total_orders = filtered_sales['orderid'].nunique()
        aov = total_revenue / total_orders if total_orders > 0 else 0
        returned_orders = filtered_sales[filtered_sales['orderstatus'] == 'Returned']['orderid'].nunique()
        return_rate = (returned_orders / total_orders * 100) if total_orders > 0 else 0

        kpi_data = {
            "Total Revenue": f"{total_revenue:,.2f} SAR",
            "Gross Margin": f"{gross_margin:.2f}%",
            "Net Profit": f"{net_profit:,.2f} SAR",
            "Total Orders": f"{total_orders:,}",
            "Avg Order Value": f"{aov:,.2f} SAR",
            "Return Rate": f"{return_rate:.2f}%"
        }

        # --- 3. (DUPLICATED) Figure Generation ---
        agg_col = 'date' if time_agg not in ['week', 'month'] else time_agg
        time_grouped = filtered_sales.groupby(agg_col)['netsale'].sum().reset_index()
        sales_over_time_fig = px.line(time_grouped, x=agg_col, y='netsale', title=f'Net Sales Trend ({time_agg.capitalize()})')
        
        category_sales = filtered_sales.groupby('category')['netsale'].sum().reset_index()
        sales_by_cat_fig = px.pie(category_sales, names='category', values='netsale', title='Sales by Category', hole=0.3)
        
        product_sales = filtered_sales.groupby('productname')['netsale'].sum().nlargest(10).reset_index() # Used for table
        
        channel_sales = filtered_sales.groupby('channel')['netsale'].sum().reset_index()
        sales_by_channel_fig = px.pie(channel_sales, names='channel', values='netsale', title='Sales by Channel', hole=0.3)
        
        city_sales = filtered_sales.groupby('city')['netsale'].sum().nlargest(10).reset_index()
        sales_by_city_fig = px.bar(city_sales, x='netsale', y='city', orientation='h', title='Top 10 Cities by Sales').update_layout(yaxis={'categoryorder':'total ascending'})
        
        # --- 4. PREPARE DATA FOR NEW GENERIC FUNCTION ---
        figures_list = [sales_over_time_fig, sales_by_cat_fig, sales_by_channel_fig, sales_by_city_fig]
        main_table_df = product_sales # Use the Top 10 Products table
        report_title = "Sales Dashboard Report"
        table_title = "Top 10 Products by Revenue"

        filter_context = {
            "Start Date": start_date,
            "End Date": end_date,
            "Regions": selected_regions,
            "Categories": selected_categories,
            "Channel": selected_channel
        }
        
        # --- 5. CALL NEW FUNCTION AND SEND BYTES ---
        pdf_bytes_io = generate_pdf_report(
            kpi_data, filter_context, main_table_df, figures_list,
            report_title=report_title, table_title=table_title
        )
        
        filename = f"Sales_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
        return dcc.send_bytes(pdf_bytes_io.getvalue(), filename)
    # --- END MODIFIED CALLBACK ---


    # --- Delivery Dashboard Callback (Unchanged from prev step) ---
    @app.callback(
        [Output('kpi-on-time-delivery', 'children'), Output('kpi-failed-delivery', 'children'),
         Output('kpi-avg-delivery-time', 'children'), Output('kpi-avg-delivery-cost', 'children'),
         Output('delivery-pipeline-chart', 'figure'), Output('avg-time-by-city-chart', 'figure'),
         Output('partner-performance-chart', 'figure')],
        [Input('delivery-apply-btn', 'n_clicks')],
        [State('delivery-partner-filter', 'value'),
         State('delivery-date-picker', 'start_date'),
         State('delivery-date-picker', 'end_date'),
         State('delivery-region-filter', 'value')]
    )
    def update_delivery_dashboard(n_clicks, selected_partner, start_date, end_date, selected_regions):
        if n_clicks is None:
            raise PreventUpdate
            
        delivery_df = DATA.get('deliveries', pd.DataFrame())
        if delivery_df.empty or not start_date or not end_date:
            raise PreventUpdate
            
        # --- Filtering Logic ---
        start_date_obj, end_date_obj = pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()
        date_mask = (delivery_df['date'] >= start_date_obj) & (delivery_df['date'] <= end_date_obj)
        partner_mask = (delivery_df['deliverypartner'] == selected_partner) if selected_partner != 'All' else True
        
        if 'All' in selected_regions or not selected_regions:
            region_mask = True
        else:
            region_mask = delivery_df['city'].isin(selected_regions)

        filtered_df = delivery_df.loc[date_mask & partner_mask & region_mask].copy()

        if filtered_df.empty:
            empty_kpi = create_kpi_body("No Data", "-")
            placeholder_fig = create_placeholder_figure("No data for selected filters")
            return [empty_kpi]*4 + [placeholder_fig]*3
            
        # --- KPI Calculation ---
        total_deliveries = len(filtered_df)
        on_time_rate = (filtered_df['on_time'].sum() / total_deliveries * 100) if total_deliveries > 0 else 0
        failed_rate = ((filtered_df['status'] == 'Failed').sum() / total_deliveries * 100) if total_deliveries > 0 else 0
        avg_delivery_time = filtered_df['delivery_time_days'].mean()
        avg_delivery_cost = filtered_df['deliverycost'].mean()

        kpi_on_time = create_kpi_body("On-Time Rate", f"{on_time_rate:.2f}%")
        kpi_failed = create_kpi_body("Failed Delivery Rate", f"{failed_rate:.2f}%")
        kpi_avg_time = create_kpi_body("Avg. Delivery Time", f"{avg_delivery_time:.2f} Days")
        kpi_avg_cost = create_kpi_body("Avg. Cost per Delivery", f"{avg_delivery_cost:,.2f} SAR")
        
        # --- Figure Generation ---
        status_order = ['Pending', 'Shipped', 'Delivered', 'Failed']
        pipeline_counts = filtered_df['status'].value_counts().reindex(status_order).fillna(0)
        pipeline_fig = px.bar(pipeline_counts, x=pipeline_counts.index, y=pipeline_counts.values, title='Live Delivery Pipeline', labels={'x': 'Status', 'y': 'Number of Orders'})
        
        time_by_city = filtered_df.groupby('city')['delivery_time_days'].mean().reset_index()
        time_by_city_fig = px.bar(time_by_city, x='city', y='delivery_time_days', title='Average Delivery Time by City', labels={'delivery_time_days': 'Average Days'})
        
        partner_perf = filtered_df.groupby('deliverypartner')['on_time'].mean().reset_index()
        partner_perf['on_time'] *= 100
        partner_perf_fig = px.bar(partner_perf.sort_values('on_time'), x='on_time', y='deliverypartner', orientation='h', title='On-Time Rate by Partner')

        return kpi_on_time, kpi_failed, kpi_avg_time, kpi_avg_cost, pipeline_fig, time_by_city_fig, partner_perf_fig

    # --- MODIFIED DELIVERY EXPORT CALLBACK ---
    @app.callback(
        Output('download-dashboard-pdf', 'data', allow_duplicate=True),
        Input('delivery-export-btn', 'n_clicks'),
        [State('delivery-partner-filter', 'value'),
         State('delivery-date-picker', 'start_date'),
         State('delivery-date-picker', 'end_date'),
         State('delivery-region-filter', 'value')],
        prevent_initial_call=True
    )
    def export_delivery_pdf(n_clicks, selected_partner, start_date, end_date, selected_regions):
        if n_clicks is None:
            raise PreventUpdate
            
        delivery_df = DATA.get('deliveries', pd.DataFrame())
        if delivery_df.empty or not start_date or not end_date:
            raise PreventUpdate
            
        # --- 1. (DUPLICATED) Filtering Logic ---
        start_date_obj, end_date_obj = pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()
        date_mask = (delivery_df['date'] >= start_date_obj) & (delivery_df['date'] <= end_date_obj)
        partner_mask = (delivery_df['deliverypartner'] == selected_partner) if selected_partner != 'All' else True
        if 'All' in selected_regions or not selected_regions:
            region_mask = True
        else:
            region_mask = delivery_df['city'].isin(selected_regions)
        filtered_df = delivery_df.loc[date_mask & partner_mask & region_mask].copy()

        if filtered_df.empty:
            raise PreventUpdate
            
        # --- 2. (DUPLICATED) KPI Calculation ---
        total_deliveries = len(filtered_df)
        on_time_rate = (filtered_df['on_time'].sum() / total_deliveries * 100) if total_deliveries > 0 else 0
        failed_rate = ((filtered_df['status'] == 'Failed').sum() / total_deliveries * 100) if total_deliveries > 0 else 0
        avg_delivery_time = filtered_df['delivery_time_days'].mean()
        avg_delivery_cost = filtered_df['deliverycost'].mean()

        kpi_data = {
            "On-Time Rate": f"{on_time_rate:.2f}%",
            "Failed Delivery Rate": f"{failed_rate:.2f}%",
            "Avg Delivery Time": f"{avg_delivery_time:.2f} Days",
            "Avg Delivery Cost": f"{avg_delivery_cost:,.2f} SAR"
        }
        
        # --- 3. (DUPLICATED) Figure Generation ---
        status_order = ['Pending', 'Shipped', 'Delivered', 'Failed']
        pipeline_counts = filtered_df['status'].value_counts().reindex(status_order).fillna(0)
        pipeline_fig = px.bar(pipeline_counts, x=pipeline_counts.index, y=pipeline_counts.values, title='Live Delivery Pipeline', labels={'x': 'Status', 'y': 'Number of Orders'})
        
        time_by_city = filtered_df.groupby('city')['delivery_time_days'].mean().reset_index()
        time_by_city_fig = px.bar(time_by_city, x='city', y='delivery_time_days', title='Average Delivery Time by City', labels={'delivery_time_days': 'Average Days'})
        
        partner_perf = filtered_df.groupby('deliverypartner')['on_time'].mean().reset_index()
        partner_perf['on_time'] *= 100
        partner_perf_fig = px.bar(partner_perf.sort_values('on_time'), x='on_time', y='deliverypartner', orientation='h', title='On-Time Rate by Partner')

        # --- 4. PREPARE DATA FOR NEW GENERIC FUNCTION ---
        figures_list = [pipeline_fig, time_by_city_fig, partner_perf_fig]
        main_table_df = partner_perf.sort_values('on_time', ascending=False)
        report_title = "Logistics & Delivery Report"
        table_title = "Partner On-Time Performance"

        filter_context = {
            "Start Date": start_date,
            "End Date": end_date,
            "Regions": selected_regions,
            "Partner": selected_partner
        }
        
        # --- 5. CALL NEW FUNCTION AND SEND BYTES ---
        pdf_bytes_io = generate_pdf_report(
            kpi_data, filter_context, main_table_df, figures_list,
            report_title=report_title, table_title=table_title
        )
        
        filename = f"Logistics_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
        return dcc.send_bytes(pdf_bytes_io.getvalue(), filename)
    # --- END MODIFIED CALLBACK ---

    
    # --- Customer Dashboard Callback (Unchanged from prev step) ---
    @app.callback(
        [Output('kpi-total-customers', 'children'), Output('kpi-active-customers', 'children'),
         Output('kpi-dormant-customers', 'children'), Output('kpi-churn-risk', 'children'),
         Output('customer-status-dist-chart', 'figure'), Output('customer-data-table', 'data'),
         Output('customer-data-table', 'columns')],
        [Input('customer-apply-btn', 'n_clicks')],
        [State('customer-list-selector', 'value'),
         State('customer-date-picker', 'start_date'),
         State('customer-date-picker', 'end_date'),
         State('customer-region-filter', 'value'),
         State('customer-segment-filter', 'value')]
    )
    def update_customer_dashboard(n_clicks, selected_list, start_date, end_date, selected_regions, selected_segments):
        if n_clicks is None:
            raise PreventUpdate
            
        customer_analysis_df = DATA.get('customer_analysis_df', pd.DataFrame())
        if customer_analysis_df.empty or not start_date or not end_date:
            placeholder = create_placeholder_figure("Customer Data Not Available")
            empty_kpi = create_kpi_body("No Data", "-")
            return [empty_kpi]*4 + [placeholder, [], []]

        # --- Filtering Logic ---
        start_dt = pd.to_datetime(start_date).date()
        end_dt = pd.to_datetime(end_date).date()
        date_mask = (customer_analysis_df['joindate'].dt.date >= start_dt) & (customer_analysis_df['joindate'].dt.date <= end_dt)
        
        if 'All' in selected_regions or not selected_regions:
            region_mask = True
        else:
            region_mask = customer_analysis_df['city'].isin(selected_regions)

        if 'All' in selected_segments or not selected_segments:
            segment_mask = True
        else:
            segment_mask = customer_analysis_df['segment'].isin(selected_segments)

        dff = customer_analysis_df[date_mask & region_mask & segment_mask]

        if dff.empty:
            placeholder = create_placeholder_figure("No data for selected filters")
            empty_kpi = create_kpi_body("No Data", "-")
            return [empty_kpi]*4 + [placeholder, [], []]

        # --- KPI Calculation ---
        status_counts = dff['status'].value_counts()
        total_cust, active_cust = len(dff), status_counts.get('Active', 0)
        dormant_cust, churn_risk_cust = status_counts.get('Dormant (At-Risk)', 0), status_counts.get('Churn Risk', 0)
        
        kpi_total = create_kpi_body("Total Customers", f"{total_cust:,}")
        kpi_active = create_kpi_body("Active Customers", f"{active_cust:,}")
        kpi_dormant = create_kpi_body("Dormant Customers", f"{dormant_cust:,}")
        kpi_churn = create_kpi_body("High Churn Risk", f"{churn_risk_cust:,}")
        
        status_dist_fig = px.pie(status_counts, names=status_counts.index, values=status_counts.values, title='Customer Status Distribution (Filtered)', hole=0.3)
        
        # --- Table Logic ---
        table_df = pd.DataFrame()
        if selected_list == 'top_value':
            table_df = dff.sort_values('monetary', ascending=False).head(20)[['customerid', 'city', 'segment', 'monetary', 'frequency', 'recency']]
        elif selected_list == 'churn_risk':
            table_df = dff[dff['status'] == 'Churn Risk'][['customerid', 'city', 'segment', 'recency', 'last_purchase_date']]
        elif selected_list == 'new':
            table_df = dff[dff['status'] == 'New'][['customerid', 'city', 'segment', 'joindate']]
        else:
            table_df = dff.head(20)[['customerid', 'city', 'segment', 'monetary', 'frequency', 'recency']]
            
        columns = [{"name": i, "id": i} for i in table_df.columns]
        data = table_df.to_dict('records')
        
        return kpi_total, kpi_active, kpi_dormant, kpi_churn, status_dist_fig, data, columns

    # --- MODIFIED CUSTOMER EXPORT CALLBACK ---
    @app.callback(
        Output('download-dashboard-pdf', 'data', allow_duplicate=True),
        Input('customer-export-btn', 'n_clicks'),
        [State('customer-list-selector', 'value'),
         State('customer-date-picker', 'start_date'),
         State('customer-date-picker', 'end_date'),
         State('customer-region-filter', 'value'),
         State('customer-segment-filter', 'value')],
        prevent_initial_call=True
    )
    def export_customer_pdf(n_clicks, selected_list, start_date, end_date, selected_regions, selected_segments):
        if n_clicks is None:
            raise PreventUpdate
            
        customer_analysis_df = DATA.get('customer_analysis_df', pd.DataFrame())
        if customer_analysis_df.empty or not start_date or not end_date:
            raise PreventUpdate

        # --- 1. (DUPLICATED) Filtering Logic ---
        start_dt = pd.to_datetime(start_date).date()
        end_dt = pd.to_datetime(end_date).date()
        date_mask = (customer_analysis_df['joindate'].dt.date >= start_dt) & (customer_analysis_df['joindate'].dt.date <= end_dt)
        if 'All' in selected_regions or not selected_regions:
            region_mask = True
        else:
            region_mask = customer_analysis_df['city'].isin(selected_regions)
        if 'All' in selected_segments or not selected_segments:
            segment_mask = True
        else:
            segment_mask = customer_analysis_df['segment'].isin(selected_segments)
        dff = customer_analysis_df[date_mask & region_mask & segment_mask]

        if dff.empty:
            raise PreventUpdate

        # --- 2. (DUPLICATED) KPI Calculation ---
        status_counts = dff['status'].value_counts()
        total_cust, active_cust = len(dff), status_counts.get('Active', 0)
        dormant_cust, churn_risk_cust = status_counts.get('Dormant (At-Risk)', 0), status_counts.get('Churn Risk', 0)
        
        list_title_str = selected_list.replace("_", " ").title()
        kpi_data = {
            "Total Customers": f"{total_cust:,}",
            "Active Customers": f"{active_cust:,}",
            "Dormant Customers": f"{dormant_cust:,}",
            "High Churn Risk": f"{churn_risk_cust:,}",
            "Selected List": list_title_str
        }

        # --- 3. (DUPLICATED) Figure & Table Generation ---
        status_dist_fig = px.pie(status_counts, names=status_counts.index, values=status_counts.values, title='Customer Status Distribution (Filtered)', hole=0.3)
        
        table_df = pd.DataFrame()
        cols = ['customerid', 'city', 'segment', 'monetary', 'frequency', 'recency']
        if selected_list == 'top_value':
            table_df = dff.sort_values('monetary', ascending=False).head(50)[cols]
        elif selected_list == 'churn_risk':
            cols = ['customerid', 'city', 'segment', 'recency', 'last_purchase_date']
            table_df = dff[dff['status'] == 'Churn Risk'].head(50)[cols]
        elif selected_list == 'new':
            cols = ['customerid', 'city', 'segment', 'joindate']
            table_df = dff[dff['status'] == 'New'].head(50)[cols]
        else: 
            table_df = dff.head(50)[cols]
            
        # Format dates for PDF if they exist
        if 'joindate' in table_df.columns:
            table_df['joindate'] = table_df['joindate'].dt.strftime('%Y-%m-%d')
        if 'last_purchase_date' in table_df.columns:
             table_df['last_purchase_date'] = table_df['last_purchase_date'].dt.strftime('%Y-%m-%d')

        # --- 4. PREPARE DATA FOR NEW GENERIC FUNCTION ---
        figures_list = [status_dist_fig]
        main_table_df = table_df # Already defined
        report_title = "Customer Segmentation Report"
        table_title = f"Customer List: {list_title_str} (Top 50)"

        filter_context = {
            "Join Date Start": start_date,
            "Join Date End": end_date,
            "Regions": selected_regions,
            "Segments": selected_segments,
            "List": kpi_data["Selected List"]
        }
        
        # --- 5. CALL NEW FUNCTION AND SEND BYTES ---
        pdf_bytes_io = generate_pdf_report(
            kpi_data, filter_context, main_table_df, figures_list,
            report_title=report_title, table_title=table_title
        )
        
        filename = f"Customer_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
        return dcc.send_bytes(pdf_bytes_io.getvalue(), filename)
    # --- END MODIFIED CALLBACK ---


    # --- CSV Export Callback (Unchanged from prev step) ---
    @app.callback(
        Output("download-dataframe-csv", "data"),
        [Input("export-csv-button", "n_clicks")],
        [State("customer-list-selector", "value")],
        prevent_initial_call=True,
    )
    def export_data(customer_clicks, selected_list):
        if customer_clicks is None:
            raise PreventUpdate
            
        ctx = callback_context
        if not ctx.triggered: raise PreventUpdate

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        df_to_export = pd.DataFrame()
        filename = ""
        
        if button_id == "export-csv-button":
            customer_analysis_df = DATA.get('customer_analysis_df', pd.DataFrame())
            if customer_analysis_df.empty: raise PreventUpdate
            if selected_list == 'top_value':
                df_to_export = customer_analysis_df.sort_values('monetary', ascending=False)
            elif selected_list == 'churn_risk':
                df_to_export = customer_analysis_df[customer_analysis_df['status'] == 'Churn Risk']
            elif selected_list == 'new':
                df_to_export = customer_analysis_df[customer_analysis_df['status'] == 'New']
            filename = f"{selected_list}_customers_{datetime.now().strftime('%Y-%m-%d')}.csv"

        if not df_to_export.empty:
            return dcc.send_data_frame(df_to_export.to_csv, filename, index=False)
            
        raise PreventUpdate
        
    # --- Marketing Dashboard Callback (Unchanged from prev step) ---
    @app.callback(
        [Output('kpi-total-ad-spend', 'children'), Output('kpi-avg-roas', 'children'),
         Output('kpi-avg-cpa', 'children'), Output('kpi-total-conversions', 'children'),
         Output('roas-by-campaign-chart', 'figure'), Output('cpa-by-campaign-chart', 'figure'),
         Output('conversions-by-channel-chart', 'figure')],
        [Input('marketing-apply-btn', 'n_clicks')],
        [State('marketing-date-picker', 'start_date'),
         State('marketing-date-picker', 'end_date'),
         State('marketing-channel-filter', 'value')]
    )
    def update_marketing_dashboard(n_clicks, start_date, end_date, selected_channel):
        if n_clicks is None:
            raise PreventUpdate
            
        campaign_performance_df = DATA.get('campaign_performance_df', pd.DataFrame()).copy()
        if campaign_performance_df.empty:
            placeholder = create_placeholder_figure("Marketing Data Not Available")
            empty_kpi = create_kpi_body("No Data", "-")
            return [empty_kpi]*4 + [placeholder]*3
        
        # --- Filtering Logic ---
        campaign_performance_df['startdate'] = pd.to_datetime(campaign_performance_df['startdate'])
        campaign_performance_df['enddate'] = pd.to_datetime(campaign_performance_df['enddate'])
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        date_mask = (campaign_performance_df['startdate'] <= end_dt) & (campaign_performance_df['enddate'] >= start_dt)
        channel_mask = (campaign_performance_df['channel'] == selected_channel) if selected_channel != 'All' else True
        dff = campaign_performance_df[date_mask & channel_mask]

        if dff.empty:
            placeholder = create_placeholder_figure("No data for selected filters")
            empty_kpi = create_kpi_body("No Data", "-")
            return [empty_kpi]*4 + [placeholder]*3

        # --- KPI Calculation ---
        total_spend, total_revenue = dff['totalcost'].sum(), dff['netsale'].sum()
        avg_roas = total_revenue / total_spend if total_spend > 0 else 0
        total_conversions = dff['conversions'].sum()
        avg_cpa = total_spend / total_conversions if total_conversions > 0 else 0
        
        kpi_spend = create_kpi_body("Total Ad Spend", f"{total_spend:,.2f} SAR")
        kpi_roas = create_kpi_body("Overall ROAS", f"{avg_roas:.2f}x")
        kpi_cpa = create_kpi_body("Average CPA", f"{avg_cpa:,.2f} SAR")
        kpi_conv = create_kpi_body("Attributed Conversions", f"{total_conversions:,.0f}")
        
        # --- Figure Generation ---
        roas_fig = px.bar(dff, x='campaignname', y='roas', color='channel', title='ROAS by Campaign')
        cpa_fig = px.bar(dff, x='campaignname', y='cpa', color='channel', title='CPA by Campaign')
        conv_by_channel = dff.groupby('channel')['conversions'].sum().reset_index()
        conv_channel_fig = px.pie(conv_by_channel, names='channel', values='conversions', title='Conversions by Channel', hole=0.3)
        
        return kpi_spend, kpi_roas, kpi_cpa, kpi_conv, roas_fig, cpa_fig, conv_channel_fig

    # --- MODIFIED MARKETING EXPORT CALLBACK ---
    @app.callback(
        Output('download-dashboard-pdf', 'data', allow_duplicate=True),
        Input('marketing-export-btn', 'n_clicks'),
        [State('marketing-date-picker', 'start_date'),
         State('marketing-date-picker', 'end_date'),
         State('marketing-channel-filter', 'value')],
        prevent_initial_call=True
    )
    def export_marketing_pdf(n_clicks, start_date, end_date, selected_channel):
        if n_clicks is None:
            raise PreventUpdate
            
        campaign_performance_df = DATA.get('campaign_performance_df', pd.DataFrame()).copy()
        if campaign_performance_df.empty:
            raise PreventUpdate
        
        # --- 1. (DUPLICATED) Filtering Logic ---
        campaign_performance_df['startdate'] = pd.to_datetime(campaign_performance_df['startdate'])
        campaign_performance_df['enddate'] = pd.to_datetime(campaign_performance_df['enddate'])
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        date_mask = (campaign_performance_df['startdate'] <= end_dt) & (campaign_performance_df['enddate'] >= start_dt)
        channel_mask = (campaign_performance_df['channel'] == selected_channel) if selected_channel != 'All' else True
        dff = campaign_performance_df[date_mask & channel_mask]

        if dff.empty:
            raise PreventUpdate

        # --- 2. (DUPLICATED) KPI Calculation ---
        total_spend, total_revenue = dff['totalcost'].sum(), dff['netsale'].sum()
        avg_roas = total_revenue / total_spend if total_spend > 0 else 0
        total_conversions = dff['conversions'].sum()
        avg_cpa = total_spend / total_conversions if total_conversions > 0 else 0
        
        kpi_data = {
            "Total Ad Spend": f"{total_spend:,.2f} SAR",
            "Overall ROAS": f"{avg_roas:.2f}x",
            "Average CPA": f"{avg_cpa:,.2f} SAR",
            "Total Conversions": f"{total_conversions:,.0f}"
        }
        
        # --- 3. (DUPLICATED) Figure Generation ---
        roas_fig = px.bar(dff, x='campaignname', y='roas', color='channel', title='ROAS by Campaign')
        cpa_fig = px.bar(dff, x='campaignname', y='cpa', color='channel', title='CPA by Campaign')
        conv_by_channel = dff.groupby('channel')['conversions'].sum().reset_index()
        conv_channel_fig = px.pie(conv_by_channel, names='channel', values='conversions', title='Conversions by Channel', hole=0.3)
        
        # --- 4. PREPARE DATA FOR NEW GENERIC FUNCTION ---
        figures_list = [roas_fig, cpa_fig, conv_channel_fig]
        # Use the filtered DF as the main table, selecting key columns
        main_table_df = dff[['campaignname', 'channel', 'totalcost', 'netsale', 'conversions', 'roas', 'cpa']].copy()
        main_table_df['roas'] = main_table_df['roas'].round(2)
        main_table_df['cpa'] = main_table_df['cpa'].round(2)
        report_title = "Marketing Performance Report"
        table_title = "Campaign Performance Details"

        filter_context = {
            "Start Date": start_date,
            "End Date": end_date,
            "Channel": selected_channel
        }
        
        # --- 5. CALL NEW FUNCTION AND SEND BYTES ---
        pdf_bytes_io = generate_pdf_report(
            kpi_data, filter_context, main_table_df, figures_list,
            report_title=report_title, table_title=table_title
        )
        
        filename = f"Marketing_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
        return dcc.send_bytes(pdf_bytes_io.getvalue(), filename)
    # --- END MODIFIED CALLBACK ---


    # --- Profit Dashboard Callback (Unchanged from prev step) ---
    @app.callback(
        [Output('kpi-total-net-profit', 'children'), Output('kpi-avg-profit-margin', 'children'),
         Output('kpi-profit-lost-returns', 'children'), Output('profit-by-channel-chart', 'figure'),
         Output('profit-by-category-chart', 'figure'), Output('high-margin-products-chart', 'figure'),
         Output('low-margin-products-chart', 'figure'), Output('automated-recommendations-list', 'children')],
        [Input('profit-apply-btn', 'n_clicks')],
        [State('profit-date-picker', 'start_date'),
         State('profit-date-picker', 'end_date'),
         State('profit-region-filter', 'value'),
         State('profit-category-filter', 'value')]
    )
    def update_profit_dashboard(n_clicks, start_date, end_date, selected_regions, selected_categories):
        if n_clicks is None:
            raise PreventUpdate
            
        profit_df = DATA.get('profit_df', pd.DataFrame())
        if profit_df.empty or not start_date or not end_date:
            placeholder = create_placeholder_figure("Profit Data Not Available")
            empty_kpi = create_kpi_body("No Data", "-")
            return [empty_kpi]*3 + [placeholder]*4 + [html.P("Not enough data.")]
        
        # --- Filtering Logic ---
        start_dt = pd.to_datetime(start_date).date()
        end_dt = pd.to_datetime(end_date).date()
        date_mask = (profit_df['date'] >= start_dt) & (profit_df['date'] <= end_dt)
        
        if 'All' in selected_regions or not selected_regions:
            region_mask = True
        else:
            region_mask = profit_df['city'].isin(selected_regions)
        
        if 'All' in selected_categories or not selected_categories:
            category_mask = True
        else:
            category_mask = profit_df['category'].isin(selected_categories)

        dff = profit_df[date_mask & region_mask & category_mask]

        if dff.empty:
            placeholder = create_placeholder_figure("No data for selected filters")
            empty_kpi = create_kpi_body("No Data", "-")
            return [empty_kpi]*3 + [placeholder]*4 + [html.P("No data for recommendations.")]

        # --- KPI Calculation ---
        total_net_profit = dff['net_profit'].sum()
        avg_profit_margin = dff['profit_margin'].mean()
        returned_orders_df = dff[dff['orderstatus'] == 'Returned']
        profit_lost_to_returns = returned_orders_df['net_profit'].sum()
        
        kpi_profit = create_kpi_body("Total Net Profit", f"{total_net_profit:,.2f} SAR")
        kpi_margin = create_kpi_body("Average Profit Margin", f"{avg_profit_margin:.2f}%")
        kpi_returns = create_kpi_body("Profit Lost to Returns", f"{profit_lost_to_returns:,.2f} SAR")
        
        # --- Figure Generation ---
        profit_by_channel = dff.groupby('channel')['net_profit'].sum().reset_index()
        profit_by_channel_fig = px.bar(profit_by_channel, x='channel', y='net_profit', title='Profit Contribution by Channel', color='channel')
        
        profit_by_category = dff.groupby('category')['net_profit'].sum().reset_index()
        profit_by_cat_fig = px.bar(profit_by_category, x='category', y='net_profit', title='Net Profit by Product Category')
        
        product_profit = dff.groupby('productname')['profit_margin'].mean().reset_index()
        high_margin_prods = product_profit.nlargest(10, 'profit_margin')
        low_margin_prods = product_profit.nsmallest(10, 'profit_margin')
        high_margin_fig = px.bar(high_margin_prods, x='profit_margin', y='productname', orientation='h', title='Top 10 Most Profitable Products').update_layout(yaxis={'categoryorder':'total ascending'})
        low_margin_fig = px.bar(low_margin_prods, x='profit_margin', y='productname', orientation='h', title='Top 10 Least Profitable Products').update_layout(yaxis={'categoryorder':'total descending'})
        
        # --- Recommendations Logic ---
        recommendations = []
        if not pd.isna(total_net_profit) and total_net_profit > 0 and not pd.isna(profit_lost_to_returns) and profit_lost_to_returns > (total_net_profit * 0.1):
            recommendations.append(html.Li("High profit loss from returns detected. Investigate top returned products/categories."))
        unprofitable_channel = profit_by_channel[profit_by_channel['net_profit'] < 0]
        if not unprofitable_channel.empty:
            recommendations.append(html.Li(f"Channel '{unprofitable_channel.iloc[0]['channel']}' is unprofitable. Review marketing strategy."))
        if not high_margin_prods.empty:
            recommendations.append(html.Li(f"'{high_margin_prods.iloc[0]['productname']}' has a high margin. Consider promoting it."))
        
        recommendation_list = html.Ul(recommendations) if recommendations else html.P("No critical issues detected.")
        
        return kpi_profit, kpi_margin, kpi_returns, profit_by_channel_fig, profit_by_cat_fig, high_margin_fig, low_margin_fig, recommendation_list

    # --- MODIFIED PROFIT EXPORT CALLBACK ---
    @app.callback(
        Output('download-dashboard-pdf', 'data', allow_duplicate=True),
        Input('profit-export-btn', 'n_clicks'),
        [State('profit-date-picker', 'start_date'),
         State('profit-date-picker', 'end_date'),
         State('profit-region-filter', 'value'),
         State('profit-category-filter', 'value')],
        prevent_initial_call=True
    )
    def export_profit_pdf(n_clicks, start_date, end_date, selected_regions, selected_categories):
        if n_clicks is None:
            raise PreventUpdate
            
        profit_df = DATA.get('profit_df', pd.DataFrame())
        if profit_df.empty or not start_date or not end_date:
            raise PreventUpdate
        
        # --- 1. (DUPLICATED) Filtering Logic ---
        start_dt = pd.to_datetime(start_date).date()
        end_dt = pd.to_datetime(end_date).date()
        date_mask = (profit_df['date'] >= start_dt) & (profit_df['date'] <= end_dt)
        if 'All' in selected_regions or not selected_regions:
            region_mask = True
        else:
            region_mask = profit_df['city'].isin(selected_regions)
        if 'All' in selected_categories or not selected_categories:
            category_mask = True
        else:
            category_mask = profit_df['category'].isin(selected_categories)
        dff = profit_df[date_mask & region_mask & category_mask]

        if dff.empty:
            raise PreventUpdate
            
        # --- 2. (DUPLICATED) KPI Calculation ---
        total_net_profit = dff['net_profit'].sum()
        avg_profit_margin = dff['profit_margin'].mean()
        returned_orders_df = dff[dff['orderstatus'] == 'Returned']
        profit_lost_to_returns = returned_orders_df['net_profit'].sum()
        
        kpi_data = {
            "Total Net Profit": f"{total_net_profit:,.2f} SAR",
            "Average Profit Margin": f"{avg_profit_margin:.2f}%",
            "Profit Lost to Returns": f"{profit_lost_to_returns:,.2f} SAR"
        }
        
        # --- 3. (DUPLICATED) Figure Generation ---
        profit_by_channel = dff.groupby('channel')['net_profit'].sum().reset_index()
        profit_by_channel_fig = px.bar(profit_by_channel, x='channel', y='net_profit', title='Profit Contribution by Channel', color='channel')
        
        profit_by_category = dff.groupby('category')['net_profit'].sum().reset_index()
        profit_by_cat_fig = px.bar(profit_by_category, x='category', y='net_profit', title='Net Profit by Product Category')
        
        product_profit = dff.groupby('productname')['profit_margin'].mean().reset_index()
        high_margin_prods = product_profit.nlargest(10, 'profit_margin')
        low_margin_prods = product_profit.nsmallest(10, 'profit_margin')
        high_margin_fig = px.bar(high_margin_prods, x='profit_margin', y='productname', orientation='h', title='Top 10 Most Profitable Products').update_layout(yaxis={'categoryorder':'total ascending'})
        low_margin_fig = px.bar(low_margin_prods, x='profit_margin', y='productname', orientation='h', title='Top 10 Least Profitable Products').update_layout(yaxis={'categoryorder':'total descending'})
        
        # --- 4. PREPARE DATA FOR NEW GENERIC FUNCTION ---
        figures_list = [profit_by_channel_fig, profit_by_cat_fig, high_margin_fig, low_margin_fig]
        main_table_df = high_margin_prods.copy()
        main_table_df['profit_margin'] = main_table_df['profit_margin'].round(2)
        report_title = "Profitability Analysis Report"
        table_title = "Top 10 Most Profitable Products (by Margin %)"

        filter_context = {
            "Start Date": start_date,
            "End Date": end_date,
            "Regions": selected_regions,
            "Categories": selected_categories
        }

        # --- 5. CALL NEW FUNCTION AND SEND BYTES ---
        pdf_bytes_io = generate_pdf_report(
            kpi_data, filter_context, main_table_df, figures_list,
            report_title=report_title, table_title=table_title
        )
        
        filename = f"Profit_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
        return dcc.send_bytes(pdf_bytes_io.getvalue(), filename)
    # --- END MODIFIED CALLBACK ---


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- ALL PREDICTIVE AND CHURN MODEL CALLBACKS REMAIN UNCHANGED ---
    # (These sections are unchanged from your original file)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @app.callback(
        Output('pred-kpi-forecast-rev', 'children'),
        Output('pred-kpi-sim-lift', 'children'),
        Output('forecast-simulation-chart', 'figure'),
        Input('forecast-run-button', 'n_clicks'),
        State('forecast-slider-days', 'value'),
        State('forecast-slider-promo', 'value'),
        prevent_initial_call=True
    )
    def update_forecast_simulation(n_clicks, forecast_days, promo_pct):
        # (This function is unchanged)
        if n_clicks == 0 or n_clicks is None:
            raise PreventUpdate
        if not os.path.exists(FORECAST_MODEL_PATH):
            logger.error("Forecast model artifact not found. Please run the training schedule.")
            return create_kpi_body("Error", "-"), create_kpi_body("Error", "-"), create_placeholder_figure("Model Not Trained")
        forecaster: DemandForecaster = joblib.load(FORECAST_MODEL_PATH)
        if forecaster is None:
             return create_kpi_body("Error", "-"), create_kpi_body("Error", "-"), create_placeholder_figure("Model Failed to Load")
        forecast_df = forecaster.predict_simulation(forecast_days, promo_pct)
        fig = go.Figure()
        history_df = forecaster.model.history
        fig.add_trace(go.Scatter(x=history_df['ds'], y=history_df['y'], mode='lines', name='Actual Sales', line=dict(color='#111111', width=2)))
        baseline_fc = forecast_df[forecast_df['forecast_type'] == 'Baseline']
        sim_fc = forecast_df[forecast_df['forecast_type'] == 'Simulation']
        fig.add_trace(go.Scatter(x=baseline_fc['ds'], y=baseline_fc['yhat'], mode='lines', name='Baseline Forecast', line=dict(color='blue', width=2, dash='dot')))
        fig.add_trace(go.Scatter(x=baseline_fc['ds'], y=baseline_fc['yhat_upper'], mode='lines', line=dict(width=0), fill=None, showlegend=False))
        fig.add_trace(go.Scatter(x=baseline_fc['ds'], y=baseline_fc['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,100,255,0.1)', name='Confidence Interval'))
        if not sim_fc.empty and promo_pct > 0:
             fig.add_trace(go.Scatter(x=sim_fc['ds'], y=sim_fc['yhat'], mode='lines', name=f'Simulation (+{promo_pct}%)', line=dict(color='green', width=3)))
        analysis_start_date = pd.to_datetime(datetime.now().date())
        analysis_end_date = analysis_start_date + timedelta(days=forecast_days)
        future_baseline_val = baseline_fc[(baseline_fc['ds'] >= analysis_start_date) & (baseline_fc['ds'] <= analysis_end_date)]['yhat'].sum()
        future_sim_val = 0.0
        if not sim_fc.empty:
            future_sim_val = sim_fc[(sim_fc['ds'] >= analysis_start_date) & (sim_fc['ds'] <= analysis_end_date)]['yhat'].sum()
        if pd.isna(future_sim_val) or future_sim_val == 0:
            future_sim_val = future_baseline_val
        sim_lift = future_sim_val - future_baseline_val
        kpi_rev_text = create_kpi_body("Forecasted Revenue", f"{future_baseline_val:,.0f} SAR")
        kpi_lift_text = create_kpi_body("Simulated Lift", f"{sim_lift:,.0f} SAR")
        fig.update_layout(title=f"Baseline Forecast vs. Simulation (+{promo_pct}%)", hovermode="x unified")
        return kpi_rev_text, kpi_lift_text, fig


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- CHURN TAB FALLBACKS AND TRAINING TRIGGERS (Unchanged) ---
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _generate_synthetic_data():
        # (This helper function is unchanged)
        logger.warning("No data found in DATA store. Generating synthetic data for model training.")
        num_customers = 500
        cities = ['Riyadh', 'Jeddah', 'Dammam']
        segments = ['Retail', 'VIP', 'Corporate']
        cust_data = []
        start_date = datetime(2022, 1, 1)
        for i in range(num_customers):
            join_date = start_date + timedelta(days=random.randint(0, 700))
            cust_data.append({'customerid': f'CUST_{i:04d}', 'city': random.choice(cities), 'segment': random.choice(segments), 'joindate': join_date.strftime('%Y-%m-%d')})
        customer_df = pd.DataFrame(cust_data)
        customer_df['joindate'] = pd.to_datetime(customer_df['joindate'])
        num_sales = 5000
        categories = ['Medication', 'Wellness', 'Personal Care', 'Equipment']
        sales_data = []
        order_id_counter = 1
        for _ in range(num_sales):
            cust = customer_df.sample(1).iloc[0]
            sale_date = cust['joindate'] + timedelta(days=random.randint(1, (datetime.now() - cust['joindate']).days - 1))
            if random.random() < 0.2:
                 sale_date = datetime.now() - timedelta(days=random.randint(181, 500))
            netsale = round(random.uniform(50.0, 800.0), 2)
            sales_data.append({'orderid': f'ORD_{order_id_counter:05d}', 'customerid': cust['customerid'], 'date': sale_date.date(), 'timestamp': sale_date, 'netsale': netsale, 'category': random.choice(categories), 'city': cust['city'], 'segment': cust['segment']})
            if random.random() < 0.3:
                continue
            order_id_counter += 1
        sales_df = pd.DataFrame(sales_data)
        sales_df['date'] = pd.to_datetime(sales_df['date'])
        sales_df['timestamp'] = pd.to_datetime(sales_df['timestamp'])
        logger.info(f"Generated {len(customer_df)} customers and {len(sales_df)} sales records.")
        return sales_df, customer_df

    
    def _run_churn_training_job():
        # (This helper function is unchanged)
        try:
            logger.info("Starting internal churn model training job...")
            sales_df = DATA.get('sales', pd.DataFrame())
            customer_df = DATA.get('customers', pd.DataFrame())
            if sales_df.empty or customer_df.empty:
                sales_df, customer_df = _generate_synthetic_data()
                DATA['sales'] = sales_df
                DATA['customers'] = customer_df
            analysis_date = pd.to_datetime(datetime.now())
            logger.info("Building RFM features for training...")
            latest_features_df = build_rfm_features(sales_df, customer_df, analysis_date)
            logger.info("Initializing ChurnPredictor and starting model fit...")
            churn_predictor = ChurnPredictor()
            predictor_instance, metrics = churn_predictor.fit(latest_features_df)
            logger.info(f"Saving trained model to: {CHURN_MODEL_PATH}")
            joblib.dump(predictor_instance, CHURN_MODEL_PATH)
            logger.info(f"Saving model metrics to: {CHURN_METRICS_PATH}")
            joblib.dump(metrics, CHURN_METRICS_PATH)
            logger.info("Churn model training job completed successfully.")
            return True, "Training successful."
        except Exception as e:
            logger.error(f"In-app model training failed: {e}", exc_info=True)
            return False, str(e)


    # --- MASTER CALLBACK 1: Renders Churn Tab Content (Unchanged) ---
    @app.callback(
        Output('churn-tab-content-wrapper', 'children'),
        Input('tabs-controller', 'active_tab'),
        Input('model-training-signal-store', 'data') # Triggered by button
    )
    def render_churn_tab_content(active_tab, training_signal):
        # (This function is unchanged)
        if active_tab != 'predictive-tab':
            raise PreventUpdate
        models_exist = os.path.exists(CHURN_MODEL_PATH) and os.path.exists(CHURN_METRICS_PATH)
        if not models_exist:
            logger.warning("Churn model artifacts not found. Displaying training prompt.")
            return dbc.Alert([
                    html.H4("Model Not Trained", className="alert-heading"),
                    html.P("The churn prediction model has not been trained yet. Please load data via the 'Refresh Data' button (if you haven't already), then run the initial training."),
                    html.Hr(),
                    dbc.Button("Run Initial Model Training", id="run-manual-churn-train-btn", color="primary", n_clicks=0)
                ], color="warning")
        
        logger.info("Churn models found. Loading artifacts and running live prediction.")
        try:
            churn_predictor: ChurnPredictor = joblib.load(CHURN_MODEL_PATH)
            metrics: dict = joblib.load(CHURN_METRICS_PATH)
            sales_df = DATA.get('sales', pd.DataFrame())
            customer_df = DATA.get('customers', pd.DataFrame())
            if sales_df.empty or customer_df.empty:
                if 'sales' not in DATA or 'customers' not in DATA:
                     return dbc.Alert(html.P("Model artifacts are present, but no Sales or Customer data is loaded. Please use the 'Refresh Data' button."), color="danger")
                else:
                    sales_df = DATA.get('sales')
                    customer_df = DATA.get('customers')
            analysis_date = pd.to_datetime(datetime.now())
            latest_features_df = build_rfm_features(sales_df, customer_df, analysis_date)
            predictions_df = churn_predictor.predict_churn_probability(latest_features_df)
            likely_churn_mask = predictions_df['ChurnProbability'] > 0.5
            likely_churn_count = predictions_df[likely_churn_mask]['customerid'].nunique()
            total_customers = predictions_df['customerid'].nunique()
            churn_rate_pct = (likely_churn_count / total_customers) * 100 if total_customers > 0 else 0
            kpi_churn_rate = create_kpi_body("Predicted Churn Rate", f"{churn_rate_pct:.1f}%")
            kpi_auc = create_kpi_body("Model AUC Score", f"{metrics.get('auc', 0):.3f}")
            at_risk_revenue = predictions_df[likely_churn_mask]['Monetary'].sum()
            kpi_risk_rev = create_kpi_body("Total At-Risk Revenue", f"{at_risk_revenue:,.0f} SAR")
            active_ltv_mean = predictions_df[~likely_churn_mask]['Estimated_LTV'].mean()
            active_ltv = active_ltv_mean if not pd.isna(active_ltv_mean) else 0.0
            kpi_ltv = create_kpi_body("Avg. LTV (Active)", f"{active_ltv:,.0f} SAR")
            drivers_df = churn_predictor.get_key_drivers_df().head(10)
            fig_drivers = px.bar(drivers_df, y='Feature', x='FeatureImportance', orientation='h', title='Top 10 Key Drivers of Churn (SHAP)').update_layout(yaxis={'categoryorder':'total ascending'})
            cols_to_show = ['customerid', 'City', 'Segment', 'Recency', 'Frequency', 'Monetary', 'ChurnProbability', 'Estimated_LTV']
            at_risk_df = predictions_df[likely_churn_mask][cols_to_show].head(50)
            at_risk_df['ChurnProbability'] = at_risk_df['ChurnProbability'].map('{:.1%}'.format)
            at_risk_df['Estimated_LTV'] = at_risk_df['Estimated_LTV'].map('{:,.0f} SAR'.format)
            at_risk_df['Monetary'] = at_risk_df['Monetary'].map('{:,.0f}'.format)
            table_cols = [{"name": i.replace("_", " ").title(), "id": i} for i in at_risk_df.columns]
            table_data = at_risk_df.to_dict('records')
            full_layout = [
                dbc.Row([
                    dbc.Col(dbc.Card(kpi_churn_rate, id="pred-kpi-churn-rate", color="danger", inverse=True), lg=3, md=6, sm=12, class_name="mb-4"),
                    dbc.Col(dbc.Card(kpi_auc, id="pred-kpi-churn-auc", color="info", inverse=True), lg=3, md=6, sm=12, class_name="mb-4"),
                    dbc.Col(dbc.Card(kpi_risk_rev, id="pred-kpi-churn-revenue", color="warning", inverse=True), lg=3, md=6, sm=12, class_name="mb-4"),
                    dbc.Col(dbc.Card(kpi_ltv, id="pred-kpi-ltv", color="success", inverse=True), lg=3, md=6, sm=12, class_name="mb-4"),
                ]),
                dbc.Row([
                    dbc.Col(dbc.Card(dbc.CardBody([html.H5("Key Drivers of Churn (Feature Importance)", className="card-title"), dcc.Graph(id="churn-key-drivers-chart", figure=fig_drivers)])), lg=5, md=12, sm=12, class_name="mb-4"),
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H5("Top Customers At-Risk of Churn", className="card-title"),
                        dash_table.DataTable(
                            id="churn-at-risk-table",
                            columns=table_cols,
                            data=table_data,
                            style_cell={'textAlign': 'left'},
                            style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
                            page_size=10,
                            sort_action='native',
                            style_table={'overflowX': 'auto'}
                        )
                    ])), lg=7, md=12, sm=12, class_name="mb-4")
                ])
            ]
            return full_layout
        except Exception as e:
            logger.error(f"Failed to render churn dashboard after loading models: {e}", exc_info=True)
            os.remove(CHURN_MODEL_PATH)
            os.remove(CHURN_METRICS_PATH)
            return dbc.Alert(f"A critical error occurred while loading the model: {e}. The corrupt model files have been deleted. Please try running the training again.", color="danger")

    # --- MASTER CALLBACK 2: Triggers Training Job (Unchanged) ---
    @app.callback(
        Output('model-training-signal-store', 'data'),
        Input('run-manual-churn-train-btn', 'n_clicks'),
        State('model-training-signal-store', 'data'),
        prevent_initial_call=True
    )
    def trigger_churn_model_training(n_clicks, current_signal):
        # (This function is unchanged)
        if n_clicks == 0 or n_clicks is None:
            raise PreventUpdate
        success, message = _run_churn_training_job()
        if not success:
            logger.error(f"Training failed: {message}")
        return (current_signal or 0) + 1