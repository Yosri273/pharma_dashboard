# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Interactive Callbacks Module - V20.0
#
# This module contains all the application's interactive logic. All @app.callback
# functions are defined here to keep the application's "brains" separate
# from its "looks" (layouts.py) and "engine" (data.py).
# -----------------------------------------------------------------------------

from dash import Input, Output, State, html, dcc
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import dash_bootstrap_components as dbc

from data import DATA, prepare_and_enrich_data
from database import get_engine, refresh_all_data
from utils import create_placeholder_figure

def register_callbacks(app):
    """Registers all callbacks for the application."""

    # --- MAIN CALLBACKS (Refresh and Tab Switching) ---
    @app.callback(
        Output('data-store-trigger', 'data'),
        Input('refresh-data-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def handle_refresh(n_clicks):
        global DATA
        engine = get_engine()
        DATA = prepare_and_enrich_data(refresh_all_data(engine))
        return "refreshed"

    @app.callback(
        Output('tab-content', 'children'),
        [Input('tabs-controller', 'active_tab'),
         Input('data-store-trigger', 'data')]
    )
    def render_tab_content(active_tab, refresh_trigger):
        from layouts import (create_sales_layout, create_delivery_layout,
                             create_customer_layout, create_competitor_layout,
                             create_marketing_layout, create_profit_layout,
                             create_predictive_layout)
        
        layouts = {
            "sales-tab": create_sales_layout, "delivery-tab": create_delivery_layout,
            "customer-tab": create_customer_layout, "competitor-tab": create_competitor_layout,
            "marketing-tab": create_marketing_layout, "profit-tab": create_profit_layout,
            "predictive-tab": create_predictive_layout
        }
        return layouts.get(active_tab, lambda: html.P("Tab not found"))()

    # --- SALES DASHBOARD CALLBACK ---
    @app.callback(
        [Output('kpi-total-revenue', 'children'), Output('kpi-gross-margin', 'children'), Output('kpi-net-profit', 'children'),
         Output('kpi-total-orders', 'children'), Output('kpi-aov', 'children'), Output('kpi-return-rate', 'children'),
         Output('sales-funnel-chart', 'figure'), Output('sales-over-time-chart', 'figure'),
         Output('sales-by-category-chart', 'figure'), Output('top-products-chart', 'figure'),
         Output('sales-by-channel-chart', 'figure'), Output('sales-by-city-chart', 'figure'),
         Output('sales-by-branch-chart', 'figure')],
        [Input('data-store-trigger', 'data'), Input('channel-filter-dropdown', 'value'), 
         Input('sales-date-picker', 'start_date'), Input('sales-date-picker', 'end_date'), 
         Input('time-agg-selector', 'value')]
    )
    def update_sales_dashboard(refresh_trigger, selected_channel, start_date, end_date, time_agg):
        sales_df = DATA.get('sales', pd.DataFrame())
        funnel_df = DATA.get('sales_funnel', pd.DataFrame())
        if sales_df.empty: raise PreventUpdate
        
        start_date_obj, end_date_obj = pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()
        date_mask = (sales_df['date'] >= start_date_obj) & (sales_df['date'] <= end_date_obj)
        channel_mask = (sales_df['channel'] == selected_channel) if selected_channel != 'All' else True
        filtered_sales = sales_df.loc[date_mask & channel_mask]

        if filtered_sales.empty:
            empty_card = dbc.CardBody([html.H4("No Data"), html.P("-", className="fs-3")])
            placeholder_fig = create_placeholder_figure("No data for this period")
            return [empty_card]*6 + [placeholder_fig]*7

        total_revenue = filtered_sales.get('netsale', 0).sum()
        total_cogs = filtered_sales.get('costofgoodssold', 0).sum()
        net_profit = total_revenue - total_cogs
        gross_margin = (net_profit / total_revenue * 100) if total_revenue > 0 else 0
        total_orders = filtered_sales['orderid'].nunique()
        aov = total_revenue / total_orders if total_orders > 0 else 0
        returned_orders = filtered_sales[filtered_sales['orderstatus'] == 'Returned']['orderid'].nunique()
        return_rate = (returned_orders / total_orders * 100) if total_orders > 0 else 0

        kpi_revenue_card = dbc.CardBody([html.H4("Total Revenue"), html.P(f"{total_revenue:,.2f} SAR", className="fs-3")])
        kpi_margin_card, kpi_profit_card = dbc.CardBody([html.H4("Gross Margin"), html.P(f"{gross_margin:.2f}%", className="fs-3")]), dbc.CardBody([html.H4("Net Profit"), html.P(f"{net_profit:,.2f} SAR", className="fs-3")])
        kpi_orders_card, kpi_aov_card = dbc.CardBody([html.H4("Total Orders"), html.P(f"{total_orders:,}", className="fs-3")]), dbc.CardBody([html.H4("Avg Order Value"), html.P(f"{aov:,.2f} SAR", className="fs-3")])
        kpi_return_card = dbc.CardBody([html.H4("Return Rate"), html.P(f"{return_rate:.2f}%", className="fs-3")])
        
        funnel_fig = create_placeholder_figure("Funnel Data Not Available")
        if not funnel_df.empty:
            completed = filtered_sales[filtered_sales['orderstatus'] == 'Completed']['orderid'].nunique()
            funnel_fig = go.Figure(go.Funnel(y=["Visits", "Carts", "Orders", "Fulfilled"], x=[funnel_df['visits'].sum(), funnel_df['carts'].sum(), funnel_df['orders'].sum(), completed], textinfo="value+percent initial")).update_layout(title_text="Sales Funnel")
        
        time_grouped = filtered_sales.groupby(time_agg)['netsale'].sum().reset_index()
        sales_over_time_fig = px.line(time_grouped, x=time_agg, y='netsale', title=f'Net Sales Trend ({time_agg.capitalize()})')
        category_sales = filtered_sales.groupby('category')['netsale'].sum().reset_index()
        sales_by_cat_fig = px.pie(category_sales, names='category', values='netsale', title='Sales by Category', hole=0.3)
        product_sales = filtered_sales.groupby('productname')['netsale'].sum().nlargest(10).reset_index()
        top_prod_fig = px.bar(product_sales, x='netsale', y='productname', orientation='h', title='Top 10 Products').update_layout(yaxis={'categoryorder':'total ascending'})
        channel_sales = filtered_sales.groupby('channel')['netsale'].sum().reset_index()
        sales_by_channel_fig = px.pie(channel_sales, names='channel', values='netsale', title='Sales by Channel', hole=0.3)
        city_sales = filtered_sales.groupby('city')['netsale'].sum().nlargest(10).reset_index()
        sales_by_city_fig = px.bar(city_sales, x='netsale', y='city', orientation='h', title='Top 10 Cities').update_layout(yaxis={'categoryorder':'total ascending'})
        branch_sales = filtered_sales.groupby('locationid')['netsale'].sum().nlargest(10).reset_index()
        sales_by_branch_fig = px.bar(branch_sales, x='netsale', y='locationid', orientation='h', title='Top 10 Pharmacy Branches').update_layout(yaxis={'categoryorder':'total ascending'})
        
        return kpi_revenue_card, kpi_margin_card, kpi_profit_card, kpi_orders_card, kpi_aov_card, kpi_return_card, funnel_fig, sales_over_time_fig, sales_by_cat_fig, top_prod_fig, sales_by_channel_fig, sales_by_city_fig, sales_by_branch_fig

    # --- OTHER CALLBACKS (Delivery, Customer, etc.) ---
    # (These would be fully implemented here in the same pattern)
    # ...

    print("All callbacks registered.")

