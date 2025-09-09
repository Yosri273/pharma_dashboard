# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Interactive Callbacks Module - V24.0 (Refactored)
#
# REFACTOR: All duplicated analytics logic from the update_* and export_*
# callbacks has been moved into single, reusable helper functions
# (e.g., _generate_sales_analytics). This follows the DRY principle,
# making the code much cleaner and easier to maintain.
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
import random
import json
from typing import Dict, Any

# --- App-Specific Imports ---
from etl.transforms import DATA, initialize_data
from services.db import get_engine
from app.utils import create_kpi_body, create_placeholder_figure
from app.reporting import generate_pdf_report
from services.storage import load_model_artifact
from models.predictors import DemandForecaster, ChurnPredictor
from models.features import build_rfm_features

# --- Constants and Setup ---
logger = logging.getLogger(__name__)
ALERT_STATE_FILE = "cache-directory/active_alerts.json"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODEL_STORE_PATH = os.path.join(PROJECT_ROOT, "model_store")
if not os.path.exists(MODEL_STORE_PATH):
    os.makedirs(MODEL_STORE_PATH, exist_ok=True)

CHURN_MODEL_PATH = os.path.join(MODEL_STORE_PATH, 'churn_predictor_main.joblib')
CHURN_METRICS_PATH = os.path.join(MODEL_STORE_PATH, 'churn_metrics.joblib')
FORECAST_MODEL_PATH = os.path.join(MODEL_STORE_PATH, 'demand_forecaster_main.joblib')


# =============================================================================
# SECTION 1: ANALYTICS HELPER FUNCTIONS (REFACTORED LOGIC)
# =============================================================================

def _generate_sales_analytics(selected_channel, start_date, end_date, time_agg, selected_regions, selected_categories) -> Dict[str, Any]:
    """
    Performs all filtering, KPI calculations, and figure generation for the Sales tab.
    This single function now contains all the business logic, which is then
    used by both the dashboard update and the PDF export callbacks.
    """
    sales_df = DATA.get('sales', pd.DataFrame())
    funnel_df = DATA.get('sales_funnel', pd.DataFrame())
    if sales_df.empty or not start_date or not end_date:
        return {"is_empty": True}

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
        return {"is_empty": True}

    # --- KPI Calculation ---
    total_revenue = filtered_sales['netsale'].sum()
    total_cogs = filtered_sales['costofgoodssold'].sum()
    net_profit = total_revenue - total_cogs
    gross_margin = (net_profit / total_revenue * 100) if total_revenue > 0 else 0
    total_orders = filtered_sales['orderid'].nunique()
    aov = total_revenue / total_orders if total_orders > 0 else 0
    returned_orders = filtered_sales[filtered_sales['orderstatus'] == 'Returned']['orderid'].nunique()
    return_rate = (returned_orders / total_orders * 100) if total_orders > 0 else 0

    kpis = {
        "kpi_revenue": create_kpi_body("Total Revenue", f"{total_revenue:,.2f} SAR"),
        "kpi_margin": create_kpi_body("Gross Margin", f"{gross_margin:.2f}%"),
        "kpi_profit": create_kpi_body("Net Profit", f"{net_profit:,.2f} SAR"),
        "kpi_orders": create_kpi_body("Total Orders", f"{total_orders:,}"),
        "kpi_aov": create_kpi_body("Avg Order Value", f"{aov:,.2f} SAR"),
        "kpi_return": create_kpi_body("Return Rate", f"{return_rate:.2f}%")
    }

    # --- Figure Generation ---
    unfiltered_view = not (selected_channel != 'All' or region_mask_active or category_mask_active)
    if not funnel_df.empty and unfiltered_view:
        completed = filtered_sales[filtered_sales['orderstatus'] == 'Completed']['orderid'].nunique()
        funnel_fig = go.Figure(go.Funnel(y=["Visits", "Carts", "Total Orders", "Fulfilled"], x=[funnel_df['visits'].sum(), funnel_df['carts'].sum(), total_orders, completed], textinfo="value+percent initial")).update_layout(title_text="Sales Funnel")
    else:
        funnel_fig = create_placeholder_figure("Funnel view disabled with active filters")

    agg_col = 'date' if time_agg not in ['week', 'month'] else time_agg
    time_grouped = filtered_sales.groupby(agg_col)['netsale'].sum().reset_index()

    figures = {
        "funnel_fig": funnel_fig,
        "sales_over_time_fig": px.line(time_grouped, x=agg_col, y='netsale', title=f'Net Sales Trend ({time_agg.capitalize()})'),
        "sales_by_cat_fig": px.pie(filtered_sales.groupby('category')['netsale'].sum().reset_index(), names='category', values='netsale', title='Sales by Category', hole=0.3),
        "top_prod_fig": px.bar(filtered_sales.groupby('productname')['netsale'].sum().nlargest(10).reset_index(), x='netsale', y='productname', orientation='h', title='Top 10 Products').update_layout(yaxis={'categoryorder':'total ascending'}),
        "sales_by_channel_fig": px.pie(filtered_sales.groupby('channel')['netsale'].sum().reset_index(), names='channel', values='netsale', title='Sales by Channel', hole=0.3),
        "sales_by_city_fig": px.bar(filtered_sales.groupby('city')['netsale'].sum().nlargest(10).reset_index(), x='netsale', y='city', orientation='h', title='Top 10 Cities by Sales').update_layout(yaxis={'categoryorder':'total ascending'}),
        "sales_by_branch_fig": px.bar(filtered_sales.groupby('locationid')['netsale'].sum().nlargest(10).reset_index(), x='netsale', y='locationid', orientation='h', title='Top 10 Pharmacy Branches by Sales').update_layout(yaxis={'categoryorder':'total ascending'})
    }
    
    # --- Data for Tables/Exports ---
    tables = {
        "top_products": filtered_sales.groupby('productname')['netsale'].sum().nlargest(10).reset_index()
    }

    return {"is_empty": False, "kpis": kpis, "figures": figures, "tables": tables}


def _generate_delivery_analytics(selected_partner, start_date, end_date, selected_regions) -> Dict[str, Any]:
    """Performs all filtering, KPI, and figure generation for the Delivery tab."""
    delivery_df = DATA.get('deliveries', pd.DataFrame())
    if delivery_df.empty or not start_date or not end_date:
        return {"is_empty": True}

    # --- Filtering Logic ---
    start_date_obj, end_date_obj = pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()
    date_mask = (delivery_df['date'] >= start_date_obj) & (delivery_df['date'] <= end_date_obj)
    partner_mask = (delivery_df['deliverypartner'] == selected_partner) if selected_partner != 'All' else True
    region_mask = delivery_df['city'].isin(selected_regions) if 'All' not in selected_regions and selected_regions else True
    filtered_df = delivery_df.loc[date_mask & partner_mask & region_mask].copy()

    if filtered_df.empty:
        return {"is_empty": True}

    # --- KPI Calculation ---
    total_deliveries = len(filtered_df)
    on_time_rate = (filtered_df['on_time'].sum() / total_deliveries * 100) if total_deliveries > 0 else 0
    failed_rate = ((filtered_df['status'] == 'Failed').sum() / total_deliveries * 100) if total_deliveries > 0 else 0
    avg_delivery_time = filtered_df['delivery_time_days'].mean()
    avg_delivery_cost = filtered_df['deliverycost'].mean()

    kpis = {
        "kpi_on_time": create_kpi_body("On-Time Rate", f"{on_time_rate:.2f}%"),
        "kpi_failed": create_kpi_body("Failed Delivery Rate", f"{failed_rate:.2f}%"),
        "kpi_avg_time": create_kpi_body("Avg. Delivery Time", f"{avg_delivery_time:.2f} Days"),
        "kpi_avg_cost": create_kpi_body("Avg. Cost per Delivery", f"{avg_delivery_cost:,.2f} SAR")
    }

    # --- Figure Generation ---
    status_order = ['Pending', 'Shipped', 'Delivered', 'Failed']
    pipeline_counts = filtered_df['status'].value_counts().reindex(status_order).fillna(0)
    partner_perf = filtered_df.groupby('deliverypartner')['on_time'].mean().reset_index()
    partner_perf['on_time'] *= 100
    
    figures = {
        "pipeline_fig": px.bar(pipeline_counts, x=pipeline_counts.index, y=pipeline_counts.values, title='Live Delivery Pipeline', labels={'x': 'Status', 'y': 'Number of Orders'}),
        "time_by_city_fig": px.bar(filtered_df.groupby('city')['delivery_time_days'].mean().reset_index(), x='city', y='delivery_time_days', title='Average Delivery Time by City', labels={'delivery_time_days': 'Average Days'}),
        "partner_perf_fig": px.bar(partner_perf.sort_values('on_time'), x='on_time', y='deliverypartner', orientation='h', title='On-Time Rate by Partner')
    }
    
    tables = {
        "partner_performance": partner_perf.sort_values('on_time', ascending=False)
    }

    return {"is_empty": False, "kpis": kpis, "figures": figures, "tables": tables}

# ... (Similar helper functions for Customer, Marketing, and Profit would follow) ...
def _generate_customer_analytics(selected_list, start_date, end_date, selected_regions, selected_segments) -> Dict[str, Any]:
    """Performs all filtering, KPI, and figure generation for the Customer tab."""
    customer_analysis_df = DATA.get('customer_analysis_df', pd.DataFrame())
    if customer_analysis_df.empty or not start_date or not end_date:
        return {"is_empty": True}

    # --- Filtering Logic ---
    start_dt, end_dt = pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()
    date_mask = (customer_analysis_df['joindate'].dt.date >= start_dt) & (customer_analysis_df['joindate'].dt.date <= end_dt)
    region_mask = customer_analysis_df['city'].isin(selected_regions) if 'All' not in selected_regions and selected_regions else True
    segment_mask = customer_analysis_df['segment'].isin(selected_segments) if 'All' not in selected_segments and selected_segments else True
    dff = customer_analysis_df[date_mask & region_mask & segment_mask]

    if dff.empty:
        return {"is_empty": True}

    # --- KPI Calculation ---
    status_counts = dff['status'].value_counts()
    total_cust, active_cust = len(dff), status_counts.get('Active', 0)
    dormant_cust, churn_risk_cust = status_counts.get('Dormant (At-Risk)', 0), status_counts.get('Churn Risk', 0)
    
    kpis = {
        "kpi_total": create_kpi_body("Total Customers", f"{total_cust:,}"),
        "kpi_active": create_kpi_body("Active Customers", f"{active_cust:,}"),
        "kpi_dormant": create_kpi_body("Dormant Customers", f"{dormant_cust:,}"),
        "kpi_churn": create_kpi_body("High Churn Risk", f"{churn_risk_cust:,}")
    }
    
    # --- Figure Generation ---
    figures = {
        "status_dist_fig": px.pie(status_counts, names=status_counts.index, values=status_counts.values, title='Customer Status Distribution (Filtered)', hole=0.3)
    }
    
    # --- Table Logic ---
    if selected_list == 'top_value':
        table_df = dff.sort_values('monetary', ascending=False).head(50)[['customerid', 'city', 'segment', 'monetary', 'frequency', 'recency']]
    elif selected_list == 'churn_risk':
        table_df = dff[dff['status'] == 'Churn Risk'].head(50)[['customerid', 'city', 'segment', 'recency', 'last_purchase_date']]
    elif selected_list == 'new':
        table_df = dff[dff['status'] == 'New'].head(50)[['customerid', 'city', 'segment', 'joindate']]
    else: 
        table_df = dff.head(50)[['customerid', 'city', 'segment', 'monetary', 'frequency', 'recency']]
    
    tables = {"customer_list": table_df}
    
    return {"is_empty": False, "kpis": kpis, "figures": figures, "tables": tables, "selected_list_title": selected_list.replace("_", " ").title()}

def _generate_marketing_analytics(start_date, end_date, selected_channel) -> Dict[str, Any]:
    """Performs all filtering, KPI, and figure generation for the Marketing tab."""
    campaign_df = DATA.get('campaign_performance_df', pd.DataFrame())
    if campaign_df.empty:
        return {"is_empty": True}

    dff = campaign_df.copy()
    dff['startdate'] = pd.to_datetime(dff['startdate'])
    dff['enddate'] = pd.to_datetime(dff['enddate'])
    start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
    date_mask = (dff['startdate'] <= end_dt) & (dff['enddate'] >= start_dt)
    channel_mask = (dff['channel'] == selected_channel) if selected_channel != 'All' else True
    filtered_df = dff[date_mask & channel_mask]

    if filtered_df.empty:
        return {"is_empty": True}

    total_spend, total_revenue = filtered_df['totalcost'].sum(), filtered_df['netsale'].sum()
    total_conversions = filtered_df['conversions'].sum()
    avg_roas = total_revenue / total_spend if total_spend > 0 else 0
    avg_cpa = total_spend / total_conversions if total_conversions > 0 else 0

    kpis = {
        "kpi_spend": create_kpi_body("Total Ad Spend", f"{total_spend:,.2f} SAR"),
        "kpi_roas": create_kpi_body("Overall ROAS", f"{avg_roas:.2f}x"),
        "kpi_cpa": create_kpi_body("Average CPA", f"{avg_cpa:,.2f} SAR"),
        "kpi_conv": create_kpi_body("Attributed Conversions", f"{total_conversions:,.0f}")
    }

    figures = {
        "roas_fig": px.bar(filtered_df, x='campaignname', y='roas', color='channel', title='ROAS by Campaign'),
        "cpa_fig": px.bar(filtered_df, x='campaignname', y='cpa', color='channel', title='CPA by Campaign'),
        "conv_channel_fig": px.pie(filtered_df.groupby('channel')['conversions'].sum().reset_index(), names='channel', values='conversions', title='Conversions by Channel', hole=0.3)
    }
    
    table_df = filtered_df[['campaignname', 'channel', 'totalcost', 'netsale', 'conversions', 'roas', 'cpa']].copy()
    table_df['roas'] = table_df['roas'].round(2)
    table_df['cpa'] = table_df['cpa'].round(2)
    tables = {"campaign_performance": table_df}

    return {"is_empty": False, "kpis": kpis, "figures": figures, "tables": tables}

def _generate_profit_analytics(start_date, end_date, selected_regions, selected_categories) -> Dict[str, Any]:
    """Performs all filtering, KPI, and figure generation for the Profit tab."""
    profit_df = DATA.get('profit_df', pd.DataFrame())
    if profit_df.empty or not start_date or not end_date:
        return {"is_empty": True}

    start_dt, end_dt = pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()
    date_mask = (profit_df['date'] >= start_dt) & (profit_df['date'] <= end_dt)
    region_mask = profit_df['city'].isin(selected_regions) if 'All' not in selected_regions and selected_regions else True
    category_mask = profit_df['category'].isin(selected_categories) if 'All' not in selected_categories and selected_categories else True
    dff = profit_df[date_mask & region_mask & category_mask]

    if dff.empty:
        return {"is_empty": True}

    total_net_profit = dff['net_profit'].sum()
    avg_profit_margin = dff['profit_margin'].mean()
    profit_lost_to_returns = dff[dff['orderstatus'] == 'Returned']['net_profit'].sum()

    kpis = {
        "kpi_profit": create_kpi_body("Total Net Profit", f"{total_net_profit:,.2f} SAR"),
        "kpi_margin": create_kpi_body("Average Profit Margin", f"{avg_profit_margin:.2f}%"),
        "kpi_returns": create_kpi_body("Profit Lost to Returns", f"{profit_lost_to_returns:,.2f} SAR")
    }

    profit_by_channel = dff.groupby('channel')['net_profit'].sum().reset_index()
    high_margin_prods = dff.groupby('productname')['profit_margin'].mean().nlargest(10).reset_index()

    figures = {
        "profit_by_channel_fig": px.bar(profit_by_channel, x='channel', y='net_profit', title='Profit Contribution by Channel', color='channel'),
        "profit_by_cat_fig": px.bar(dff.groupby('category')['net_profit'].sum().reset_index(), x='category', y='net_profit', title='Net Profit by Product Category'),
        "high_margin_fig": px.bar(high_margin_prods, x='profit_margin', y='productname', orientation='h', title='Top 10 Most Profitable Products').update_layout(yaxis={'categoryorder':'total ascending'}),
        "low_margin_fig": px.bar(dff.groupby('productname')['profit_margin'].mean().nsmallest(10).reset_index(), x='profit_margin', y='productname', orientation='h', title='Top 10 Least Profitable Products').update_layout(yaxis={'categoryorder':'total descending'})
    }

    recommendations = []
    if not pd.isna(total_net_profit) and total_net_profit > 0 and not pd.isna(profit_lost_to_returns) and profit_lost_to_returns > (total_net_profit * 0.1):
        recommendations.append(html.Li("High profit loss from returns detected. Investigate top returned products/categories."))
    if not profit_by_channel[profit_by_channel['net_profit'] < 0].empty:
        recommendations.append(html.Li(f"Channel '{profit_by_channel[profit_by_channel['net_profit'] < 0].iloc[0]['channel']}' is unprofitable. Review marketing strategy."))
    if not high_margin_prods.empty:
        recommendations.append(html.Li(f"'{high_margin_prods.iloc[0]['productname']}' has a high margin. Consider promoting it."))
    
    tables = {"high_margin_products": high_margin_prods}

    return {"is_empty": False, "kpis": kpis, "figures": figures, "tables": tables, "recommendations": html.Ul(recommendations) if recommendations else html.P("No critical issues detected.")}


# =============================================================================
# SECTION 2: DASHBOARD CALLBACKS (NOW REFACTORED)
# =============================================================================

def register_callbacks(app):
    """Registers all application callbacks."""

    # --- MAIN/ALERT CALLBACKS (Unchanged) ---
    @app.callback(Output('active-alert-banner-container', 'children'), Input('alert-poll-interval', 'n_intervals'))
    def update_active_alert_display(n):
        if not os.path.exists(ALERT_STATE_FILE): return []
        try:
            with open(ALERT_STATE_FILE, 'r') as f:
                content = f.read()
                if not content: return []
                alert_state = json.loads(content)
        except Exception as e:
            print(f"Error reading alert state file for UI: {e}")
            return []
        banners = []
        for name, data in alert_state.items():
            if data.get("is_active"):
                ts = data.get("last_triggered", "")
                header = f"{name} (Since: {datetime.fromisoformat(ts).strftime('%Y-%m-%d %I:%M %p')})" if ts else name
                banners.append(dbc.Alert([html.H5(header, className="alert-heading"), html.P(data.get("message"))], color="danger", dismissable=True, duration=45000, className="mb-3"))
        return banners

    @callback(Output("navbar-collapse", "is_open"), [Input("navbar-toggler", "n_clicks")], [State("navbar-collapse", "is_open")])
    def toggle_navbar_collapse(n, is_open):
        if n: return not is_open
        return is_open

    @app.callback(Output('data-store-trigger', 'data'), Input('refresh-data-button', 'n_clicks'), prevent_initial_call=True)
    def handle_refresh(n_clicks):
        logger.info("Refresh data button clicked. Re-initializing all data.")
        engine = get_engine()
        initialize_data(engine)
        return "refreshed"

    @app.callback(Output('tab-content', 'children'), Input('tabs-controller', 'active_tab'))
    def render_tab_content(active_tab):
        from app.layout import create_sales_layout, create_delivery_layout, create_customer_layout, create_marketing_layout, create_profit_layout, create_predictive_layout
        layouts = {"sales-tab": create_sales_layout, "delivery-tab": create_delivery_layout, "customer-tab": create_customer_layout, "marketing-tab": create_marketing_layout, "profit-tab": create_profit_layout, "predictive-tab": create_predictive_layout}
        return layouts.get(active_tab, lambda: html.H4("Tab not found."))()

    # --- SALES TAB (Refactored) ---
    @app.callback(
        [Output('kpi-total-revenue', 'children'), Output('kpi-gross-margin', 'children'),
         Output('kpi-net-profit', 'children'), Output('kpi-total-orders', 'children'),
         Output('kpi-aov', 'children'), Output('kpi-return-rate', 'children'),
         Output('sales-funnel-chart', 'figure'), Output('sales-over-time-chart', 'figure'),
         Output('sales-by-category-chart', 'figure'), Output('top-products-chart', 'figure'),
         Output('sales-by-channel-chart', 'figure'), Output('sales-by-city-chart', 'figure'),
         Output('sales-by-branch-chart', 'figure')],
        Input('sales-apply-btn', 'n_clicks'),
        [State('channel-filter-dropdown', 'value'), State('sales-date-picker', 'start_date'),
         State('sales-date-picker', 'end_date'), State('time-agg-selector', 'value'),
         State('sales-region-filter', 'value'), State('sales-category-filter', 'value')]
    )
    def update_sales_dashboard(n_clicks, selected_channel, start_date, end_date, time_agg, selected_regions, selected_categories):
        analytics = _generate_sales_analytics(selected_channel, start_date, end_date, time_agg, selected_regions, selected_categories)
        if analytics["is_empty"]:
            empty_kpi = create_kpi_body("No Data", "-")
            placeholder_fig = create_placeholder_figure("No data for selected filters")
            return [empty_kpi]*6 + [placeholder_fig]*7
        kpis = analytics["kpis"]
        figs = analytics["figures"]
        return list(kpis.values()) + list(figs.values())

    @app.callback(
        Output('download-dashboard-pdf', 'data', allow_duplicate=True),
        Input('sales-export-btn', 'n_clicks'),
        [State('channel-filter-dropdown', 'value'), State('sales-date-picker', 'start_date'),
         State('sales-date-picker', 'end_date'), State('time-agg-selector', 'value'),
         State('sales-region-filter', 'value'), State('sales-category-filter', 'value')],
        prevent_initial_call=True
    )
    def export_sales_pdf(n_clicks, selected_channel, start_date, end_date, time_agg, selected_regions, selected_categories):
        if n_clicks is None: raise PreventUpdate
        analytics = _generate_sales_analytics(selected_channel, start_date, end_date, time_agg, selected_regions, selected_categories)
        if analytics["is_empty"]: raise PreventUpdate
        
        kpi_data = {k.replace("kpi_", "").replace("_", " ").title(): v.children[1].children for k, v in analytics["kpis"].items()}
        filter_context = {"Start Date": start_date, "End Date": end_date, "Regions": selected_regions, "Categories": selected_categories, "Channel": selected_channel}
        
        pdf_bytes = generate_pdf_report(
            kpi_data=kpi_data,
            filters_dict=filter_context,
            main_dataframe=analytics["tables"]["top_products"],
            figures_list=list(analytics["figures"].values()),
            report_title="Sales Dashboard Report",
            table_title="Top 10 Products by Revenue"
        )
        return dcc.send_bytes(pdf_bytes.getvalue(), f"Sales_Report_{datetime.now().strftime('%Y%m%d')}.pdf")

    # --- DELIVERY TAB (Refactored) ---
    @app.callback(
        [Output('kpi-on-time-delivery', 'children'), Output('kpi-failed-delivery', 'children'),
         Output('kpi-avg-delivery-time', 'children'), Output('kpi-avg-delivery-cost', 'children'),
         Output('delivery-pipeline-chart', 'figure'), Output('avg-time-by-city-chart', 'figure'),
         Output('partner-performance-chart', 'figure')],
        Input('delivery-apply-btn', 'n_clicks'),
        [State('delivery-partner-filter', 'value'), State('delivery-date-picker', 'start_date'),
         State('delivery-date-picker', 'end_date'), State('delivery-region-filter', 'value')]
    )
    def update_delivery_dashboard(n_clicks, selected_partner, start_date, end_date, selected_regions):
        analytics = _generate_delivery_analytics(selected_partner, start_date, end_date, selected_regions)
        if analytics["is_empty"]:
            empty_kpi = create_kpi_body("No Data", "-")
            placeholder_fig = create_placeholder_figure("No data for selected filters")
            return [empty_kpi]*4 + [placeholder_fig]*3
        return list(analytics["kpis"].values()) + list(analytics["figures"].values())

    @app.callback(
        Output('download-dashboard-pdf', 'data', allow_duplicate=True),
        Input('delivery-export-btn', 'n_clicks'),
        [State('delivery-partner-filter', 'value'), State('delivery-date-picker', 'start_date'),
         State('delivery-date-picker', 'end_date'), State('delivery-region-filter', 'value')],
        prevent_initial_call=True
    )
    def export_delivery_pdf(n_clicks, selected_partner, start_date, end_date, selected_regions):
        if n_clicks is None: raise PreventUpdate
        analytics = _generate_delivery_analytics(selected_partner, start_date, end_date, selected_regions)
        if analytics["is_empty"]: raise PreventUpdate

        kpi_data = {k.replace("kpi_", "").replace("_", " ").title(): v.children[1].children for k, v in analytics["kpis"].items()}
        filter_context = {"Start Date": start_date, "End Date": end_date, "Regions": selected_regions, "Partner": selected_partner}

        pdf_bytes = generate_pdf_report(
            kpi_data=kpi_data,
            filters_dict=filter_context,
            main_dataframe=analytics["tables"]["partner_performance"],
            figures_list=list(analytics["figures"].values()),
            report_title="Logistics & Delivery Report",
            table_title="Partner On-Time Performance"
        )
        return dcc.send_bytes(pdf_bytes.getvalue(), f"Logistics_Report_{datetime.now().strftime('%Y%m%d')}.pdf")
    
    # --- CUSTOMER TAB (Refactored) ---
    @app.callback(
        [Output('kpi-total-customers', 'children'), Output('kpi-active-customers', 'children'),
         Output('kpi-dormant-customers', 'children'), Output('kpi-churn-risk', 'children'),
         Output('customer-status-dist-chart', 'figure'), Output('customer-data-table', 'data'),
         Output('customer-data-table', 'columns')],
        Input('customer-apply-btn', 'n_clicks'),
        [State('customer-list-selector', 'value'), State('customer-date-picker', 'start_date'),
         State('customer-date-picker', 'end_date'), State('customer-region-filter', 'value'),
         State('customer-segment-filter', 'value')]
    )
    def update_customer_dashboard(n_clicks, selected_list, start_date, end_date, selected_regions, selected_segments):
        analytics = _generate_customer_analytics(selected_list, start_date, end_date, selected_regions, selected_segments)
        if analytics["is_empty"]:
            placeholder = create_placeholder_figure("No data for selected filters")
            empty_kpi = create_kpi_body("No Data", "-")
            return [empty_kpi]*4 + [placeholder, [], []]
        
        table_df = analytics["tables"]["customer_list"]
        columns = [{"name": i, "id": i} for i in table_df.columns]
        data = table_df.to_dict('records')
        
        return list(analytics["kpis"].values()) + [analytics["figures"]["status_dist_fig"], data, columns]

    @app.callback(
        Output('download-dashboard-pdf', 'data', allow_duplicate=True),
        Input('customer-export-btn', 'n_clicks'),
        [State('customer-list-selector', 'value'), State('customer-date-picker', 'start_date'),
         State('customer-date-picker', 'end_date'), State('customer-region-filter', 'value'),
         State('customer-segment-filter', 'value')],
        prevent_initial_call=True
    )
    def export_customer_pdf(n_clicks, selected_list, start_date, end_date, selected_regions, selected_segments):
        if n_clicks is None: raise PreventUpdate
        analytics = _generate_customer_analytics(selected_list, start_date, end_date, selected_regions, selected_segments)
        if analytics["is_empty"]: raise PreventUpdate
        
        kpi_data = {k.replace("kpi_", "").replace("_", " ").title(): v.children[1].children for k, v in analytics["kpis"].items()}
        filter_context = {"Join Date Start": start_date, "Join Date End": end_date, "Regions": selected_regions, "Segments": selected_segments, "List": analytics["selected_list_title"]}
        
        table_df = analytics["tables"]["customer_list"]
        if 'joindate' in table_df.columns: table_df['joindate'] = table_df['joindate'].dt.strftime('%Y-%m-%d')
        if 'last_purchase_date' in table_df.columns: table_df['last_purchase_date'] = table_df['last_purchase_date'].dt.strftime('%Y-%m-%d')

        pdf_bytes = generate_pdf_report(
            kpi_data=kpi_data,
            filters_dict=filter_context,
            main_dataframe=table_df,
            figures_list=[analytics["figures"]["status_dist_fig"]],
            report_title="Customer Segmentation Report",
            table_title=f"Customer List: {analytics['selected_list_title']} (Top 50)"
        )
        return dcc.send_bytes(pdf_bytes.getvalue(), f"Customer_Report_{datetime.now().strftime('%Y%m%d')}.pdf")

    # --- MARKETING TAB (Refactored) ---
    @app.callback(
        [Output('kpi-total-ad-spend', 'children'), Output('kpi-avg-roas', 'children'),
         Output('kpi-avg-cpa', 'children'), Output('kpi-total-conversions', 'children'),
         Output('roas-by-campaign-chart', 'figure'), Output('cpa-by-campaign-chart', 'figure'),
         Output('conversions-by-channel-chart', 'figure')],
        Input('marketing-apply-btn', 'n_clicks'),
        [State('marketing-date-picker', 'start_date'), State('marketing-date-picker', 'end_date'),
         State('marketing-channel-filter', 'value')]
    )
    def update_marketing_dashboard(n_clicks, start_date, end_date, selected_channel):
        analytics = _generate_marketing_analytics(start_date, end_date, selected_channel)
        if analytics["is_empty"]:
            placeholder = create_placeholder_figure("No data for selected filters")
            empty_kpi = create_kpi_body("No Data", "-")
            return [empty_kpi]*4 + [placeholder]*3
        return list(analytics["kpis"].values()) + list(analytics["figures"].values())

    @app.callback(
        Output('download-dashboard-pdf', 'data', allow_duplicate=True),
        Input('marketing-export-btn', 'n_clicks'),
        [State('marketing-date-picker', 'start_date'), State('marketing-date-picker', 'end_date'),
         State('marketing-channel-filter', 'value')],
        prevent_initial_call=True
    )
    def export_marketing_pdf(n_clicks, start_date, end_date, selected_channel):
        if n_clicks is None: raise PreventUpdate
        analytics = _generate_marketing_analytics(start_date, end_date, selected_channel)
        if analytics["is_empty"]: raise PreventUpdate

        kpi_data = {k.replace("kpi_", "").replace("_", " ").title(): v.children[1].children for k, v in analytics["kpis"].items()}
        filter_context = {"Start Date": start_date, "End Date": end_date, "Channel": selected_channel}

        pdf_bytes = generate_pdf_report(
            kpi_data=kpi_data,
            filters_dict=filter_context,
            main_dataframe=analytics["tables"]["campaign_performance"],
            figures_list=list(analytics["figures"].values()),
            report_title="Marketing Performance Report",
            table_title="Campaign Performance Details"
        )
        return dcc.send_bytes(pdf_bytes.getvalue(), f"Marketing_Report_{datetime.now().strftime('%Y%m%d')}.pdf")

    # --- PROFIT TAB (Refactored) ---
    @app.callback(
        [Output('kpi-total-net-profit', 'children'), Output('kpi-avg-profit-margin', 'children'),
         Output('kpi-profit-lost-returns', 'children'), Output('profit-by-channel-chart', 'figure'),
         Output('profit-by-category-chart', 'figure'), Output('high-margin-products-chart', 'figure'),
         Output('low-margin-products-chart', 'figure'), Output('automated-recommendations-list', 'children')],
        Input('profit-apply-btn', 'n_clicks'),
        [State('profit-date-picker', 'start_date'), State('profit-date-picker', 'end_date'),
         State('profit-region-filter', 'value'), State('profit-category-filter', 'value')]
    )
    def update_profit_dashboard(n_clicks, start_date, end_date, selected_regions, selected_categories):
        analytics = _generate_profit_analytics(start_date, end_date, selected_regions, selected_categories)
        if analytics["is_empty"]:
            placeholder = create_placeholder_figure("No data for selected filters")
            empty_kpi = create_kpi_body("No Data", "-")
            return [empty_kpi]*3 + [placeholder]*4 + [html.P("No data for recommendations.")]
        return list(analytics["kpis"].values()) + list(analytics["figures"].values()) + [analytics["recommendations"]]

    @app.callback(
        Output('download-dashboard-pdf', 'data', allow_duplicate=True),
        Input('profit-export-btn', 'n_clicks'),
        [State('profit-date-picker', 'start_date'), State('profit-date-picker', 'end_date'),
         State('profit-region-filter', 'value'), State('profit-category-filter', 'value')],
        prevent_initial_call=True
    )
    def export_profit_pdf(n_clicks, start_date, end_date, selected_regions, selected_categories):
        if n_clicks is None: raise PreventUpdate
        analytics = _generate_profit_analytics(start_date, end_date, selected_regions, selected_categories)
        if analytics["is_empty"]: raise PreventUpdate

        kpi_data = {k.replace("kpi_", "").replace("_", " ").title(): v.children[1].children for k, v in analytics["kpis"].items()}
        filter_context = {"Start Date": start_date, "End Date": end_date, "Regions": selected_regions, "Categories": selected_categories}
        
        table_df = analytics["tables"]["high_margin_products"]
        table_df['profit_margin'] = table_df['profit_margin'].round(2)

        pdf_bytes = generate_pdf_report(
            kpi_data=kpi_data,
            filters_dict=filter_context,
            main_dataframe=table_df,
            figures_list=list(analytics["figures"].values()),
            report_title="Profitability Analysis Report",
            table_title="Top 10 Most Profitable Products (by Margin %)"
        )
        return dcc.send_bytes(pdf_bytes.getvalue(), f"Profit_Report_{datetime.now().strftime('%Y%m%d')}.pdf")

    # --- CSV Export Callback (Original) ---
    @app.callback(Output("download-dataframe-csv", "data"), Input("export-csv-button", "n_clicks"), State("customer-list-selector", "value"), prevent_initial_call=True)
    def export_data(n_clicks, selected_list):
        if n_clicks is None: raise PreventUpdate
        df_to_export = pd.DataFrame()
        customer_analysis_df = DATA.get('customer_analysis_df', pd.DataFrame())
        if customer_analysis_df.empty: raise PreventUpdate
        if selected_list == 'top_value':
            df_to_export = customer_analysis_df.sort_values('monetary', ascending=False)
        elif selected_list == 'churn_risk':
            df_to_export = customer_analysis_df[customer_analysis_df['status'] == 'Churn Risk']
        elif selected_list == 'new':
            df_to_export = customer_analysis_df[customer_analysis_df['status'] == 'New']
        if not df_to_export.empty:
            return dcc.send_data_frame(df_to_export.to_csv, f"{selected_list}_customers_{datetime.now().strftime('%Y-%m-%d')}.csv", index=False)
        raise PreventUpdate

    # =============================================================================
    # SECTION 3: PREDICTIVE ANALYTICS CALLBACKS (Unchanged Logic)
    # =============================================================================
    
    @app.callback(
        [Output('pred-kpi-forecast-rev', 'children'), Output('pred-kpi-sim-lift', 'children'),
         Output('forecast-simulation-chart', 'figure')],
        Input('forecast-run-button', 'n_clicks'),
        [State('forecast-slider-days', 'value'), State('forecast-slider-promo', 'value')],
        prevent_initial_call=True
    )
    def update_forecast_simulation(n_clicks, forecast_days, promo_pct):
        # (This function's logic is self-contained and not duplicated, so it remains unchanged)
        if n_clicks == 0 or n_clicks is None: raise PreventUpdate
        if not os.path.exists(FORECAST_MODEL_PATH):
            return create_kpi_body("Error", "-"), create_kpi_body("Error", "-"), create_placeholder_figure("Model Not Trained")
        forecaster: DemandForecaster = joblib.load(FORECAST_MODEL_PATH)
        if forecaster is None:
             return create_kpi_body("Error", "-"), create_kpi_body("Error", "-"), create_placeholder_figure("Model Failed to Load")

        forecast_df = forecaster.predict_simulation(forecast_days, promo_pct)
        # ... (rest of the original forecast logic is preserved)
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


    # --- CHURN TAB FALLBACKS AND TRAINING TRIGGERS (Unchanged) ---
    def _run_churn_training_job():
        try:
            logger.info("Starting internal churn model training job...")
            sales_df = DATA.get('sales', pd.DataFrame())
            customer_df = DATA.get('customers', pd.DataFrame())
            if sales_df.empty or customer_df.empty:
                # Synthetic data generation logic preserved
                num_customers, cust_data = 500, []
                start_date = datetime(2022, 1, 1)
                for i in range(num_customers):
                    join_date = start_date + timedelta(days=random.randint(0, 700))
                    cust_data.append({'customerid': f'CUST_{i:04d}', 'city': random.choice(['Riyadh', 'Jeddah', 'Dammam']), 'segment': random.choice(['Retail', 'VIP', 'Corporate']), 'joindate': join_date.strftime('%Y-%m-%d')})
                customer_df = pd.DataFrame(cust_data)
                customer_df['joindate'] = pd.to_datetime(customer_df['joindate'])
                num_sales, sales_data, order_id_counter = 5000, [], 1
                for _ in range(num_sales):
                    cust = customer_df.sample(1).iloc[0]
                    sale_date = cust['joindate'] + timedelta(days=random.randint(1, (datetime.now() - cust['joindate']).days - 1))
                    if random.random() < 0.2: sale_date = datetime.now() - timedelta(days=random.randint(181, 500))
                    netsale = round(random.uniform(50.0, 800.0), 2)
                    sales_data.append({'orderid': f'ORD_{order_id_counter:05d}', 'customerid': cust['customerid'], 'date': sale_date.date(), 'timestamp': sale_date, 'netsale': netsale, 'category': random.choice(['Medication', 'Wellness', 'Personal Care']), 'city': cust['city'], 'segment': cust['segment']})
                    if random.random() < 0.3: continue
                    order_id_counter += 1
                sales_df = pd.DataFrame(sales_data)
                sales_df['date'] = pd.to_datetime(sales_df['date'])
                sales_df['timestamp'] = pd.to_datetime(sales_df['timestamp'])
                DATA['sales'], DATA['customers'] = sales_df, customer_df

            analysis_date = pd.to_datetime(datetime.now())
            latest_features_df = build_rfm_features(sales_df, customer_df, analysis_date)
            churn_predictor = ChurnPredictor()
            predictor_instance, metrics = churn_predictor.fit(latest_features_df)
            joblib.dump(predictor_instance, CHURN_MODEL_PATH)
            joblib.dump(metrics, CHURN_METRICS_PATH)
            logger.info("Churn model training job completed successfully.")
            return True
        except Exception as e:
            logger.error(f"In-app model training failed: {e}", exc_info=True)
            return False

    @app.callback(Output('churn-tab-content-wrapper', 'children'), [Input('tabs-controller', 'active_tab'), Input('model-training-signal-store', 'data')])
    def render_churn_tab_content(active_tab, training_signal):
        if active_tab != 'predictive-tab': raise PreventUpdate
        models_exist = os.path.exists(CHURN_MODEL_PATH) and os.path.exists(CHURN_METRICS_PATH)
        if not models_exist:
            return dbc.Alert([html.H4("Model Not Trained"), html.P("The churn prediction model has not been trained. Please load data and run the initial training."), html.Hr(), dbc.Button("Run Initial Model Training", id="run-manual-churn-train-btn", color="primary")], color="warning")
        
        try:
            # All original logic for rendering the churn tab is preserved here...
            churn_predictor: ChurnPredictor = joblib.load(CHURN_MODEL_PATH)
            metrics: dict = joblib.load(CHURN_METRICS_PATH)
            sales_df, customer_df = DATA.get('sales', pd.DataFrame()), DATA.get('customers', pd.DataFrame())
            if sales_df.empty or customer_df.empty:
                return dbc.Alert(html.P("Model artifacts exist, but no Sales/Customer data is loaded. Please use 'Refresh Data'."), color="danger")
            
            predictions_df = churn_predictor.predict_churn_probability(build_rfm_features(sales_df, customer_df, pd.to_datetime(datetime.now())))
            likely_churn_mask = predictions_df['ChurnProbability'] > 0.5
            churn_rate_pct = (predictions_df[likely_churn_mask]['customerid'].nunique() / predictions_df['customerid'].nunique()) * 100 if predictions_df['customerid'].nunique() > 0 else 0
            at_risk_revenue = predictions_df[likely_churn_mask]['Monetary'].sum()
            active_ltv = predictions_df[~likely_churn_mask]['Estimated_LTV'].mean()
            
            kpi_churn_rate = create_kpi_body("Predicted Churn Rate", f"{churn_rate_pct:.1f}%")
            kpi_auc = create_kpi_body("Model AUC Score", f"{metrics.get('auc', 0):.3f}")
            kpi_risk_rev = create_kpi_body("Total At-Risk Revenue", f"{at_risk_revenue:,.0f} SAR")
            kpi_ltv = create_kpi_body("Avg. LTV (Active)", f"{active_ltv:,.0f} SAR" if not pd.isna(active_ltv) else "N/A")

            fig_drivers = px.bar(churn_predictor.get_key_drivers_df().head(10), y='Feature', x='FeatureImportance', orientation='h', title='Top 10 Key Drivers of Churn (SHAP)').update_layout(yaxis={'categoryorder':'total ascending'})
            
            at_risk_df = predictions_df[likely_churn_mask][['customerid', 'City', 'Segment', 'Recency', 'Monetary', 'ChurnProbability', 'Estimated_LTV']].head(50)
            at_risk_df['ChurnProbability'] = at_risk_df['ChurnProbability'].map('{:.1%}'.format)
            at_risk_df['Estimated_LTV'] = at_risk_df['Estimated_LTV'].map('{:,.0f} SAR'.format)
            at_risk_df['Monetary'] = at_risk_df['Monetary'].map('{:,.0f}'.format)
            table_cols = [{"name": i.replace("_", " ").title(), "id": i} for i in at_risk_df.columns]
            table_data = at_risk_df.to_dict('records')
            
            return [
                dbc.Row([
                    dbc.Col(dbc.Card(kpi_churn_rate, color="danger", inverse=True), lg=3, md=6, className="mb-4"),
                    dbc.Col(dbc.Card(kpi_auc, color="info", inverse=True), lg=3, md=6, className="mb-4"),
                    dbc.Col(dbc.Card(kpi_risk_rev, color="warning", inverse=True), lg=3, md=6, className="mb-4"),
                    dbc.Col(dbc.Card(kpi_ltv, color="success", inverse=True), lg=3, md=6, className="mb-4"),
                ]),
                dbc.Row([
                    dbc.Col(dbc.Card(dbc.CardBody([html.H5("Key Drivers of Churn"), dcc.Graph(figure=fig_drivers)])), lg=5, className="mb-4"),
                    dbc.Col(dbc.Card(dbc.CardBody([html.H5("Top Customers At-Risk of Churn"), dash_table.DataTable(columns=table_cols, data=table_data, page_size=10, sort_action='native', style_table={'overflowX': 'auto'})])), lg=7, className="mb-4")
                ])
            ]
        except Exception as e:
            logger.error(f"Failed to render churn dashboard: {e}", exc_info=True)
            if os.path.exists(CHURN_MODEL_PATH): os.remove(CHURN_MODEL_PATH)
            if os.path.exists(CHURN_METRICS_PATH): os.remove(CHURN_METRICS_PATH)
            return dbc.Alert(f"A critical error occurred loading the model: {e}. Corrupt files deleted. Please try training again.", color="danger")

    @app.callback(Output('model-training-signal-store', 'data'), Input('run-manual-churn-train-btn', 'n_clicks'), State('model-training-signal-store', 'data'), prevent_initial_call=True)
    def trigger_churn_model_training(n_clicks, current_signal):
        if n_clicks is None: raise PreventUpdate
        _run_churn_training_job()
        return (current_signal or 0) + 1
