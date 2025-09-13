# -*- coding: utf-8 -*-
"""
This module contains helper functions that perform the core data analysis
for each dashboard tab. Callbacks will import and use these functions
to get the data, figures, and KPIs they need to update the UI.
"""
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import html, dash_table
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import joblib
import os
import sys
import random
from typing import Dict, Any

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from etl.transforms import DATA
from app.utils.kpi import create_kpi_body, create_placeholder_figure
from models.predictors import ChurnPredictor
from models.features import build_rfm_features

logger = logging.getLogger(__name__)

# --- Model and Cache Paths ---
MODEL_STORE_PATH = os.path.join(PROJECT_ROOT, "model_store")
if not os.path.exists(MODEL_STORE_PATH):
    os.makedirs(MODEL_STORE_PATH, exist_ok=True)

CHURN_MODEL_PATH = os.path.join(MODEL_STORE_PATH, 'churn_predictor_main.joblib')
CHURN_METRICS_PATH = os.path.join(MODEL_STORE_PATH, 'churn_metrics.joblib')
FORECAST_MODEL_PATH = os.path.join(MODEL_STORE_PATH, 'demand_forecaster_main.joblib')


def set_dark_theme(fig):
    fig.update_layout(
        paper_bgcolor="#232a3d",
        plot_bgcolor="#232a3d",
        font_color="#e0e6f1"
    )
    return fig

# --- Analytics Functions ---

def generate_sales_analytics(selected_channel, start_date, end_date, time_agg, selected_regions, selected_categories, selected_products=None, selected_branches=None) -> Dict[str, Any]:
    sales_df, funnel_df = DATA.get('sales', pd.DataFrame()), DATA.get('sales_funnel', pd.DataFrame())
    if sales_df.empty or not start_date or not end_date: return {"is_empty": True}
    start_date_obj, end_date_obj = pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()
    date_mask = (sales_df['date'] >= start_date_obj) & (sales_df['date'] <= end_date_obj)
    channel_mask = (sales_df['channel'] == selected_channel) if selected_channel != 'All' else True
    region_mask_active = 'All' not in selected_regions and bool(selected_regions)
    region_mask = sales_df['city'].isin(selected_regions) if region_mask_active else True
    category_mask_active = 'All' not in selected_categories and bool(selected_categories)
    category_mask = sales_df['category'].isin(selected_categories) if category_mask_active else True
    # Product and Branch filters: compute after initial filter and ensure boolean Series aligned to filtered_sales
    filtered_sales = sales_df.loc[date_mask & channel_mask & region_mask & category_mask].copy()
    # If no product/branch filters provided or set to 'All', keep all rows
    if filtered_sales.empty:
        return {"is_empty": True}

    if selected_products and isinstance(selected_products, (list, tuple)) and 'All' not in selected_products and 'productname' in filtered_sales.columns:
        product_mask_series = filtered_sales['productname'].isin(selected_products)
    else:
        product_mask_series = pd.Series(True, index=filtered_sales.index)

    if selected_branches and isinstance(selected_branches, (list, tuple)) and 'All' not in selected_branches:
        if 'LocationID' in filtered_sales.columns:
            branch_mask_series = filtered_sales['LocationID'].isin(selected_branches)
        elif 'locationid' in filtered_sales.columns:
            branch_mask_series = filtered_sales['locationid'].isin(selected_branches)
        else:
            branch_mask_series = pd.Series(True, index=filtered_sales.index)
    else:
        branch_mask_series = pd.Series(True, index=filtered_sales.index)

    # Apply product & branch masks (both are Series aligned to filtered_sales)
    filtered_sales = filtered_sales.loc[product_mask_series & branch_mask_series]
    if filtered_sales.empty:
        return {"is_empty": True}
    if filtered_sales.empty: return {"is_empty": True}
    total_revenue, total_cogs = filtered_sales['netsale'].sum(), filtered_sales['costofgoodssold'].sum()
    net_profit, total_orders = total_revenue - total_cogs, filtered_sales['orderid'].nunique()
    gross_margin, aov, return_rate = ((net_profit / total_revenue * 100) if total_revenue > 0 else 0), (total_revenue / total_orders if total_orders > 0 else 0), ((filtered_sales[filtered_sales['orderstatus'] == 'Returned']['orderid'].nunique() / total_orders * 100) if total_orders > 0 else 0)
    kpis = {"kpi_revenue": create_kpi_body("Total Revenue", f"{total_revenue:,.2f} SAR"), "kpi_margin": create_kpi_body("Gross Margin", f"{gross_margin:.2f}%"), "kpi_profit": create_kpi_body("Net Profit", f"{net_profit:,.2f} SAR"), "kpi_orders": create_kpi_body("Total Orders", f"{total_orders:,}"), "kpi_aov": create_kpi_body("Avg Order Value", f"{aov:,.2f} SAR"), "kpi_return": create_kpi_body("Return Rate", f"{return_rate:.2f}%")}
    # --- Improved Sales Over Time Chart ---
    # Aggregate by selected time aggregation (time_agg)
    if (
        'date' in filtered_sales and 'netsale' in filtered_sales and
        not filtered_sales.empty
    ):
        if time_agg == 'Month':
            filtered_sales['period'] = pd.to_datetime(filtered_sales['date']).to_period('M').dt.to_timestamp()
        elif time_agg == 'Week':
            filtered_sales['period'] = pd.to_datetime(filtered_sales['date']).to_period('W').dt.start_time
        else:
            filtered_sales['period'] = pd.to_datetime(filtered_sales['date'])
        sales_over_time = filtered_sales.groupby('period')['netsale'].sum().reset_index()
        if not sales_over_time.empty and 'period' in sales_over_time and 'netsale' in sales_over_time:
            sales_over_time['MA'] = sales_over_time['netsale'].rolling(window=3, min_periods=1).mean()
            fig_sales_over_time = px.line(sales_over_time, x='period', y='netsale', title='Sales Over Time', markers=True)
            fig_sales_over_time.add_scatter(x=sales_over_time['period'], y=sales_over_time['MA'], mode='lines', name='3-period MA', line=dict(dash='dash', color='#e0e6f1'))
            fig_sales_over_time = set_dark_theme(fig_sales_over_time)
        else:
            fig_sales_over_time = create_placeholder_figure("No sales over time data")
    else:
        fig_sales_over_time = create_placeholder_figure("No sales over time data")

    # Ecommerce funnel stages: Visitors, Add to Cart, Checkout, Purchase
    funnel_stages = ["Visitors", "Add to Cart", "Checkout", "Purchase"]
    visitors = filtered_sales['visitor_id'].nunique() if 'visitor_id' in filtered_sales else 0
    add_to_cart = filtered_sales['added_to_cart'].sum() if 'added_to_cart' in filtered_sales else 0
    checkout = filtered_sales['checkout_started'].sum() if 'checkout_started' in filtered_sales else 0
    purchase = filtered_sales['orderid'].nunique() if 'orderid' in filtered_sales else len(filtered_sales)
    funnel_counts = [visitors, add_to_cart, checkout, purchase]
    figures = {
        "funnel_fig": set_dark_theme(px.funnel(x=funnel_counts, y=funnel_stages, title="Ecommerce Sales Funnel")),
        "sales_over_time_fig": fig_sales_over_time,
        "period_growth_fig": set_dark_theme(px.bar(filtered_sales, x="period", y="growth", title="Period-over-Period Growth")) if 'period' in filtered_sales and 'growth' in filtered_sales else create_placeholder_figure("No period data"),
        "price_volume_fig": set_dark_theme(px.scatter(filtered_sales, x="price", y="volume", title="Price vs. Volume")) if 'price' in filtered_sales and 'volume' in filtered_sales else create_placeholder_figure("No price/volume data"),
        "sales_by_cat_fig": set_dark_theme(px.pie(filtered_sales, names="category", title="Sales by Category")) if 'category' in filtered_sales else create_placeholder_figure("No category data"),
        "top_prod_fig": set_dark_theme(px.bar(filtered_sales.groupby('productname')['netsale'].sum().nlargest(10).reset_index(), x="productname", y="netsale", title="Top Products")) if 'productname' in filtered_sales and 'netsale' in filtered_sales else create_placeholder_figure("No product data"),
        "sales_by_channel_fig": set_dark_theme(px.bar(filtered_sales.groupby('channel')['netsale'].sum().reset_index(), x="channel", y="netsale", title="Sales by Channel")) if 'channel' in filtered_sales and 'netsale' in filtered_sales else create_placeholder_figure("No channel data"),
        "sales_by_city_fig": set_dark_theme(px.bar(filtered_sales.groupby('city')['netsale'].sum().reset_index(), x="city", y="netsale", title="Sales by City")) if 'city' in filtered_sales and 'netsale' in filtered_sales else create_placeholder_figure("No city data"),
        "sales_by_branch_fig": set_dark_theme(
            px.bar(
                filtered_sales.groupby('LocationID')['netsale'].sum().nlargest(10).reset_index(),
                x="netsale", y="LocationID", orientation="h", title="Top 10 Branches by Sales"
            )
        ) if 'LocationID' in filtered_sales and 'netsale' in filtered_sales else create_placeholder_figure("No branch data")
    }

    # --- Robust Sales by Branch Chart ---
    print('DEBUG: filtered_sales columns:', filtered_sales.columns.tolist())
    if 'locationid' in filtered_sales.columns and 'netsale' in filtered_sales:
        branch_sales = filtered_sales.groupby('locationid')['netsale'].sum().reset_index()
        if not branch_sales.empty:
            top_branches = branch_sales.nlargest(10, 'netsale') if len(branch_sales) > 10 else branch_sales.sort_values('netsale', ascending=False)
            figures["sales_by_branch_fig"] = set_dark_theme(
                px.bar(top_branches, x="netsale", y="locationid", orientation="h", title="Top Branches by Sales")
            )
        else:
            figures["sales_by_branch_fig"] = create_placeholder_figure("No branch sales data")
    else:
        figures["sales_by_branch_fig"] = create_placeholder_figure("No branch data")

    tables = {"top_products": filtered_sales.groupby('productname')['netsale'].sum().nlargest(10).reset_index() if 'productname' in filtered_sales and 'netsale' in filtered_sales else pd.DataFrame()}
    return {"is_empty": False, "kpis": kpis, "figures": figures, "tables": tables}

def generate_delivery_analytics(selected_driver, selected_vehicle, start_date, end_date, selected_regions):
    delivery_df = DATA.get('deliveries', pd.DataFrame())
    if delivery_df.empty or not start_date or not end_date: return {"is_empty": True}
    if 'driverid' not in delivery_df.columns or 'vehicletype' not in delivery_df.columns:
        logger.error("Delivery data is missing 'driverid' or 'vehicletype' columns. Cannot render internal logistics tab.")
        return {"is_empty": True, "error": "Data schema mismatch."}

    start_date_obj, end_date_obj = pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()
    date_mask = (delivery_df['date'] >= start_date_obj) & (delivery_df['date'] <= end_date_obj)
    driver_mask = (delivery_df['driverid'] == selected_driver) if selected_driver != 'All' else True
    vehicle_mask = (delivery_df['vehicletype'] == selected_vehicle) if selected_vehicle != 'All' else True
    region_mask = delivery_df['city'].isin(selected_regions) if 'All' not in selected_regions and selected_regions else True
    filtered_df = delivery_df.loc[date_mask & driver_mask & vehicle_mask & region_mask].copy()
    
    if filtered_df.empty: return {"is_empty": True}
    
    total_deliveries = len(filtered_df)
    on_time_rate, failed_rate = ((filtered_df['on_time'].sum() / total_deliveries * 100) if total_deliveries > 0 else 0), (((filtered_df['status'] == 'Failed').sum() / total_deliveries * 100) if total_deliveries > 0 else 0)
    avg_delivery_time, avg_delivery_cost = filtered_df['delivery_time_days'].mean(), filtered_df['deliverycost'].mean()
    kpis = {"kpi_on_time": create_kpi_body("On-Time Rate", f"{on_time_rate:.2f}%"), "kpi_failed": create_kpi_body("Failed Delivery Rate", f"{failed_rate:.2f}%"), "kpi_avg_time": create_kpi_body("Avg. Delivery Time", f"{avg_delivery_time:.2f} Days"), "kpi_avg_cost": create_kpi_body("Avg. Cost per Delivery", f"{avg_delivery_cost:,.2f} SAR")}
    
    status_order = ['Pending', 'Shipped', 'Delivered', 'Failed']
    pipeline_counts = filtered_df['status'].value_counts().reindex(status_order).fillna(0)
    
    figures = {
        "pipeline_fig": set_dark_theme(px.bar(pipeline_counts, x=pipeline_counts.index, y=pipeline_counts.values, title='Live Delivery Pipeline')),
        "time_by_city_fig": set_dark_theme(px.bar(filtered_df.groupby('city')['delivery_time_days'].mean().reset_index(), x='city', y='delivery_time_days', title='Average Delivery Time by City')),
    }
    driver_perf = filtered_df.groupby('driverid').agg(total_deliveries=('orderid', 'nunique'), on_time_rate=('on_time', lambda x: x.mean() * 100)).reset_index().nlargest(10, 'total_deliveries')
    figures['driver_leaderboard_fig'] = set_dark_theme(px.bar(driver_perf, x='total_deliveries', y='driverid', color='on_time_rate', orientation='h', title='Top 10 Drivers by Volume'))
    vehicle_perf = filtered_df.groupby('vehicletype').agg(avg_cost=('deliverycost', 'mean'), avg_time=('delivery_time_days', 'mean')).reset_index()
    figures['vehicle_efficiency_fig'] = set_dark_theme(px.bar(vehicle_perf, x='vehicletype', y='avg_cost', color='avg_time', title='Avg. Cost & Time by Vehicle Type'))
    
    tables = {"driver_performance": driver_perf}
    return {"is_empty": False, "kpis": kpis, "figures": figures, "tables": tables}

def generate_customer_analytics(selected_list, start_date, end_date, selected_regions, selected_segments):
    customer_analysis_df = DATA.get('customer_analysis_df', pd.DataFrame())
    if customer_analysis_df.empty or not start_date or not end_date: return {"is_empty": True}
    start_dt, end_dt = pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()
    date_mask = (customer_analysis_df['joindate'].dt.date >= start_dt) & (customer_analysis_df['joindate'].dt.date <= end_dt)
    region_mask = customer_analysis_df['city'].isin(selected_regions) if 'All' not in selected_regions and selected_regions else True
    segment_mask = customer_analysis_df['segment'].isin(selected_segments) if 'All' not in selected_segments and selected_segments else True
    dff = customer_analysis_df[date_mask & region_mask & segment_mask]
    if dff.empty: return {"is_empty": True}
    status_counts = dff['status'].value_counts()
    total_cust, active_cust, dormant_cust, churn_risk_cust = len(dff), status_counts.get('Active', 0), status_counts.get('Dormant (At-Risk)', 0), status_counts.get('Churn Risk', 0)
    kpis = {"kpi_total": create_kpi_body("Total Customers", f"{total_cust:,}"), "kpi_active": create_kpi_body("Active Customers", f"{active_cust:,}"), "kpi_dormant": create_kpi_body("Dormant Customers", f"{dormant_cust:,}"), "kpi_churn": create_kpi_body("High Churn Risk", f"{churn_risk_cust:,}")}
    figures = {
        "status_dist_fig": set_dark_theme(px.pie(status_counts, names=status_counts.index, values=status_counts.values, title='Customer Status Distribution (Filtered)', hole=0.3)),
    }
    rfm_segment_analysis = dff.groupby('status').agg(recency=('recency', 'mean'), frequency=('frequency', 'mean'), monetary=('monetary', 'sum'), size=('customerid', 'nunique')).reset_index()
    figures['rfm_bubble_fig'] = set_dark_theme(px.scatter(rfm_segment_analysis, x='recency', y='frequency', size='monetary', color='status', hover_name='status', size_max=60, text='status'))
    if selected_list == 'top_value': table_df = dff.sort_values('monetary', ascending=False).head(50)[['customerid', 'city', 'segment', 'monetary', 'frequency', 'recency']]
    elif selected_list == 'churn_risk': table_df = dff[dff['status'] == 'Churn Risk'].head(50)[['customerid', 'city', 'segment', 'recency', 'last_purchase_date']]
    elif selected_list == 'new': table_df = dff[dff['status'] == 'New'].head(50)[['customerid', 'city', 'segment', 'joindate']]
    else: table_df = dff.head(50)[['customerid', 'city', 'segment', 'monetary', 'frequency', 'recency']]
    tables = {"customer_list": table_df}
    return {"is_empty": False, "kpis": kpis, "figures": figures, "tables": tables, "selected_list_title": selected_list.replace("_", " ").title()}

def generate_marketing_analytics(start_date, end_date, selected_channel, selected_products=None):
    campaign_df = DATA.get('campaign_performance_df', pd.DataFrame())
    if campaign_df.empty: return {"is_empty": True}
    dff = campaign_df.copy()
    dff['startdate'], dff['enddate'] = pd.to_datetime(dff['startdate']), pd.to_datetime(dff['enddate'])
    start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
    date_mask = (dff['startdate'] <= end_dt) & (dff['enddate'] >= start_dt)
    channel_mask = (dff['channel'] == selected_channel) if selected_channel != 'All' else pd.Series(True, index=dff.index)
    filtered_df = dff.loc[date_mask & channel_mask].copy()
    # product filter (if provided)
    if selected_products and isinstance(selected_products, (list, tuple)) and 'All' not in selected_products and 'productname' in filtered_df.columns:
        prod_mask = filtered_df['productname'].isin(selected_products)
        filtered_df = filtered_df.loc[prod_mask]
    if filtered_df.empty: return {"is_empty": True}
    total_spend, total_revenue, total_conversions = filtered_df['totalcost'].sum(), filtered_df['netsale'].sum(), filtered_df['conversions'].sum()
    avg_roas, avg_cpa = (total_revenue / total_spend if total_spend > 0 else 0), (total_spend / total_conversions if total_conversions > 0 else 0)
    kpis = {"kpi_spend": create_kpi_body("Total Ad Spend", f"{total_spend:,.2f} SAR"), "kpi_roas": create_kpi_body("Overall ROAS", f"{avg_roas:.2f}x"), "kpi_cpa": create_kpi_body("Average CPA (CAC)", f"{avg_cpa:,.2f} SAR"), "kpi_conv": create_kpi_body("Attributed Conversions", f"{total_conversions:,.0f}")}
    figures = {
        "roas_fig": set_dark_theme(px.bar(filtered_df, x='campaignname', y='roas', color='channel', title='ROAS by Campaign')),
        "cpa_fig": set_dark_theme(px.bar(filtered_df, x='campaignname', y='cpa', color='channel', title='CPA by Campaign')),
        "conv_channel_fig": set_dark_theme(px.pie(filtered_df.groupby('channel')['conversions'].sum().reset_index(), names='channel', values='conversions', title='Conversions by Channel', hole=0.3)),
    }
    clv_by_channel_df = DATA.get('advanced_analytics', {}).get('clv_by_channel', pd.DataFrame())
    if not clv_by_channel_df.empty:
        figures['clv_by_channel_fig'] = set_dark_theme(px.bar(clv_by_channel_df, x='channel', y='Estimated_LTV', color='channel'))
    else:
        figures['clv_by_channel_fig'] = create_placeholder_figure("Not enough data for CLV by Channel")
    table_df = filtered_df[['campaignname', 'channel', 'totalcost', 'netsale', 'conversions', 'roas', 'cpa']].copy()
    table_df[['roas', 'cpa']] = table_df[['roas', 'cpa']].round(2)
    tables = {"campaign_performance": table_df}
    return {"is_empty": False, "kpis": kpis, "figures": figures, "tables": tables}

def generate_profit_analytics(start_date, end_date, selected_regions, selected_categories, selected_products=None, selected_branches=None):
    profit_df = DATA.get('profit_df', pd.DataFrame())
    if profit_df.empty or not start_date or not end_date: return {"is_empty": True}
    start_dt, end_dt = pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()
    date_mask = (profit_df['date'] >= start_dt) & (profit_df['date'] <= end_dt)
    region_mask = profit_df['city'].isin(selected_regions) if 'All' not in selected_regions and selected_regions else True
    category_mask = profit_df['category'].isin(selected_categories) if 'All' not in selected_categories and selected_categories else True
    dff = profit_df.loc[date_mask & region_mask & category_mask].copy()
    if dff.empty: return {"is_empty": True}
    # product filter
    if selected_products and isinstance(selected_products, (list, tuple)) and 'All' not in selected_products and 'productname' in dff.columns:
        dff = dff.loc[dff['productname'].isin(selected_products)]
    # branch filter
    if selected_branches and isinstance(selected_branches, (list, tuple)) and 'All' not in selected_branches:
        if 'LocationID' in dff.columns:
            dff = dff.loc[dff['LocationID'].isin(selected_branches)]
        elif 'locationid' in dff.columns:
            dff = dff.loc[dff['locationid'].isin(selected_branches)]
    if dff.empty: return {"is_empty": True}
    total_net_profit, avg_profit_margin, profit_lost_to_returns = dff['net_profit'].sum(), dff['profit_margin'].mean(), dff[dff['orderstatus'] == 'Returned']['net_profit'].sum()
    kpis = {"kpi_profit": create_kpi_body("Total Net Profit", f"{total_net_profit:,.2f} SAR"), "kpi_margin": create_kpi_body("Average Profit Margin", f"{avg_profit_margin:.2f}%"), "kpi_returns": create_kpi_body("Profit Lost to Returns", f"{profit_lost_to_returns:,.2f} SAR")}
    profit_by_channel = dff.groupby('channel')['net_profit'].sum().reset_index()
    high_margin_prods = dff.groupby('productname')['profit_margin'].mean().nlargest(10).reset_index()
    figures = {
        "profit_by_channel_fig": set_dark_theme(px.bar(profit_by_channel, x='channel', y='net_profit', title='Profit Contribution by Channel')),
        "profit_by_cat_fig": set_dark_theme(px.bar(dff.groupby('category')['net_profit'].sum().reset_index(), x='category', y='net_profit', title='Net Profit by Product Category')),
        "high_margin_fig": set_dark_theme(px.bar(high_margin_prods, x='profit_margin', y='productname', orientation='h', title='Top 10 Most Profitable Products')),
        "low_margin_fig": set_dark_theme(px.bar(dff.groupby('productname')['profit_margin'].mean().nsmallest(10).reset_index(), x='profit_margin', y='productname', orientation='h', title='Top 10 Least Profitable Products')),
    }
    waterfall_df = DATA.get('advanced_analytics', {}).get('waterfall_data', pd.DataFrame())
    if not waterfall_df.empty:
        net_profit_calc = waterfall_df[waterfall_df['measure'] != 'Net Profit']['amount'].sum()
        waterfall_df.loc[waterfall_df['measure'] == 'Net Profit', 'amount'] = net_profit_calc
        figures['profit_waterfall_fig'] = set_dark_theme(go.Figure(go.Waterfall(name="Profit Breakdown", orientation="v", measure=["absolute", "relative", "relative", "relative", "relative", "total"], x=waterfall_df['measure'], y=waterfall_df['amount'])))
    else:
        figures['profit_waterfall_fig'] = create_placeholder_figure("Not enough data for Waterfall chart")
    recommendations = []
    if not pd.isna(total_net_profit) and total_net_profit > 0 and not pd.isna(profit_lost_to_returns) and profit_lost_to_returns > (total_net_profit * 0.1): recommendations.append(html.Li("High profit loss from returns detected."))
    if not profit_by_channel[profit_by_channel['net_profit'] < 0].empty: recommendations.append(html.Li(f"Channel '{profit_by_channel[profit_by_channel['net_profit'] < 0].iloc[0]['channel']}' is unprofitable."))
    if not high_margin_prods.empty: recommendations.append(html.Li(f"'{high_margin_prods.iloc[0]['productname']}' has a high margin. Consider promoting it."))
    tables = {"high_margin_products": high_margin_prods}
    return {"is_empty": False, "kpis": kpis, "figures": figures, "tables": tables, "recommendations": html.Ul(recommendations) if recommendations else html.P("No critical issues detected.")}


def run_churn_training_job(job_id: str = None):
    try:
        logger.info("Starting internal churn model training job...")
        # Optional progress reporting helper
        try:
            from services.training import update_status
        except Exception:
            update_status = None

        if job_id and update_status:
            try:
                update_status(job_id, 'started', {'phase': 'initializing'})
            except Exception:
                pass
        sales_df, customer_df = DATA.get('sales', pd.DataFrame()), DATA.get('customers', pd.DataFrame())
        
        # If data is empty, generate synthetic data for demonstration
        if sales_df.empty or customer_df.empty:
            logger.warning("Sales or customer data not found. Generating synthetic data for churn model training.")
            if job_id and update_status:
                try:
                    update_status(job_id, 'progress', {'phase': 'generating_synthetic_data', 'percent': 10})
                except Exception:
                    pass
            num_customers, cust_data, start_date = 500, [], datetime(2022, 1, 1)
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
            sales_df['date'], sales_df['timestamp'] = pd.to_datetime(sales_df['date']), pd.to_datetime(sales_df['timestamp'])
            DATA['sales'], DATA['customers'] = sales_df, customer_df

        analysis_date = pd.to_datetime(datetime.now())
        if job_id and update_status:
            try:
                update_status(job_id, 'progress', {'phase': 'building_features', 'percent': 20})
            except Exception:
                pass
        latest_features_df = build_rfm_features(sales_df, customer_df, analysis_date)
        # --- Pre-flight validation: normalize column names to canonical casing expected by ChurnPredictor ---
        required_numeric = ['Recency', 'Frequency', 'Monetary', 'Tenure']
        required_categorical = ['City', 'Segment']
        required_target = 'Churned'

        # Build mapping lower -> actual column name
        cols_map = {c.lower(): c for c in latest_features_df.columns}

        # Attempt to rename any case-insensitive matches to the canonical casing
        rename_map = {}
        for col in required_numeric + required_categorical + [required_target]:
            low = col.lower()
            if low in cols_map:
                actual = cols_map[low]
                if actual != col:
                    rename_map[actual] = col

        if rename_map:
            try:
                latest_features_df = latest_features_df.rename(columns=rename_map)
                # refresh cols_map after rename
                cols_map = {c.lower(): c for c in latest_features_df.columns}
            except Exception:
                logger.exception('Failed to rename feature columns during preflight normalization')

        # Ensure categorical placeholders exist (OneHotEncoder can handle NA/unknown)
        for rc in required_categorical:
            if rc.lower() not in cols_map:
                latest_features_df[rc] = pd.NA

        # Check required numeric and target columns are present
        missing = []
        for rn in required_numeric:
            if rn not in latest_features_df.columns:
                missing.append(rn)
        if required_target not in latest_features_df.columns:
            missing.append(required_target)

        if missing:
            msg = f"Training aborted: required columns missing or misnamed: {missing}. Expected exact column names: {required_numeric + required_categorical + [required_target]}."
            logger.error(msg)
            if job_id and update_status:
                try:
                    update_status(job_id, 'failed', {'error': msg, 'phase': 'preflight_validation'})
                except Exception:
                    pass
            # raise so the worker records the failure details rather than being overwritten
            raise RuntimeError(msg)

        if job_id and update_status:
            try:
                update_status(job_id, 'progress', {'phase': 'features_built', 'percent': 40})
            except Exception:
                pass
        churn_predictor = ChurnPredictor()
        if job_id and update_status:
            try:
                update_status(job_id, 'progress', {'phase': 'training_model', 'percent': 45})
            except Exception:
                pass
        predictor_instance, metrics = churn_predictor.fit(latest_features_df)
        if job_id and update_status:
            try:
                update_status(job_id, 'progress', {'phase': 'model_trained', 'percent': 80})
            except Exception:
                pass
        
        joblib.dump(predictor_instance, CHURN_MODEL_PATH)
        joblib.dump(metrics, CHURN_METRICS_PATH)
        if job_id and update_status:
            try:
                update_status(job_id, 'progress', {'phase': 'artifacts_saved', 'percent': 90})
            except Exception:
                pass

        # Persist simple metadata for auditability and UI display
        try:
            import json, time, os
            meta = {
                'trained_at': time.time(),
                'metrics': metrics,
                'training_rows': int(len(latest_features_df)),
                'model_path': CHURN_MODEL_PATH
            }
            meta_path = os.path.join(MODEL_STORE_PATH, 'churn_predictor_main.metadata.json')
            with open(meta_path, 'w') as fh:
                json.dump(meta, fh)
            logger.info(f"Wrote churn model metadata to {meta_path}")
        except Exception:
            logger.exception("Failed to write model metadata")

        # Also register in model registry
        try:
            from services.model_registry import register_model
            register_model('churn_predictor_main', CHURN_MODEL_PATH, metrics, meta)
            logger.info('Registered churn model in model_registry table')
        except Exception:
            logger.exception('Failed to register model in registry')
        
        logger.info("Churn model training job completed successfully.")
        if job_id and update_status:
            try:
                update_status(job_id, 'success', {'phase': 'completed', 'percent': 100})
            except Exception:
                pass
        return True
    except Exception as e:
        logger.error(f"In-app model training failed: {e}", exc_info=True)
        return False


def get_shap_summary(model_path: str, top_n: int = 20):
    """Load a model artifact and return a small SHAP summary DataFrame.

    Returns a list of dicts: [{'Feature': name, 'MeanAbsSHAP': value}, ...]
    This function is defensive: if SHAP explainer is not available, returns [].
    """
    try:
        if not os.path.exists(model_path):
            return []
        mdl = joblib.load(model_path)
        if not hasattr(mdl, 'shap_explainer') or mdl.shap_explainer is None:
            return []
        feature_names = getattr(mdl, 'feature_names', None)
        shap_exp = mdl.shap_explainer
        try:
            shap_values = shap_exp.values_for_class(1)
        except Exception:
            try:
                shap_values = shap_exp.values
            except Exception:
                return []
        import numpy as _np
        mean_abs = _np.abs(shap_values).mean(axis=0)
        cols = feature_names if feature_names is not None else [f'f{i}' for i in range(len(mean_abs))]
        rows = [{'Feature': c, 'MeanAbsSHAP': float(v)} for c, v in zip(cols, mean_abs)]
        rows = sorted(rows, key=lambda r: r['MeanAbsSHAP'], reverse=True)[:top_n]
        return rows
    except Exception:
        logger.exception('Failed to compute SHAP summary')
        return []
