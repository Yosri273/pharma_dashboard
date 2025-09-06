# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# UI Layouts Module - V20.0
#
# This module defines the visual structure (the UI components) for the entire
# application, including the main layout and each individual dashboard tab.
# It reads data from the global DATA store in data.py.
# -----------------------------------------------------------------------------

from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from data import DATA
from utils import create_placeholder_figure

# --- MAIN LAYOUT ---
def create_main_layout():
    """Creates the main layout of the application with the title and tabs."""
    return dbc.Container([
        dcc.Store(id='data-store-trigger'),
        dcc.Download(id="download-dataframe-csv"),
        html.H1("Pharma Analytics Hub - v20.0 (Modular)", className='text-center text-primary mb-4'),
        dbc.Button("Refresh Data", id="refresh-data-button", color="info", className="mb-2 float-end"),
        dbc.Tabs(id="tabs-controller", active_tab="sales-tab", children=[
            dbc.Tab(label="Sales", tab_id="sales-tab"),
            dbc.Tab(label="Logistics", tab_id="delivery-tab"),
            dbc.Tab(label="Customers", tab_id="customer-tab"),
            dbc.Tab(label="Market Intel", tab_id="competitor-tab"),
            dbc.Tab(label="Marketing", tab_id="marketing-tab"),
            dbc.Tab(label="Profit Optimization", tab_id="profit-tab"),
            dbc.Tab(label="Predictive Insights", tab_id="predictive-tab"),
        ]),
        html.Div(id='tab-content', className="mt-4")
    ], fluid=True)


# --- INDIVIDUAL DASHBOARD LAYOUTS ---

def create_sales_layout():
    sales_df = DATA.get('sales')
    if sales_df is None or sales_df.empty:
        return html.Div("Sales Data Not Available", className="text-center mt-5")
    return dbc.Container([
        dbc.Row([
            dbc.Col(dcc.Dropdown(id='channel-filter-dropdown', options=[{'label': 'All Channels', 'value': 'All'}] + [{'label': ch, 'value': ch} for ch in sorted(sales_df['channel'].unique())], value='All', clearable=False), width=5, className="mb-4"),
            dbc.Col(dcc.DatePickerRange(id='sales-date-picker', min_date_allowed=sales_df['date'].min(), max_date_allowed=sales_df['date'].max(), start_date=sales_df['date'].min(), end_date=sales_df['date'].max()), width=5, className="mb-4"),
            dbc.Col(dcc.RadioItems(id='time-agg-selector', options=[{'label': 'Daily', 'value': 'date'}, {'label': 'Weekly', 'value': 'week'}, {'label': 'Monthly', 'value': 'month'}], value='date', inline=True, labelStyle={'margin-right': '10px'}), width=2, className="mb-4"),
        ]),
        dbc.Row([
            dbc.Col(dbc.Card(id='kpi-total-revenue', color="primary", inverse=True), width=4),
            dbc.Col(dbc.Card(id='kpi-gross-margin', color="success", inverse=True), width=4),
            dbc.Col(dbc.Card(id='kpi-net-profit', color="dark", inverse=True), width=4),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(id='kpi-total-orders', color="info", inverse=True), width=4),
            dbc.Col(dbc.Card(id='kpi-aov', color="secondary", inverse=True), width=4),
            dbc.Col(dbc.Card(id='kpi-return-rate', color="danger", inverse=True), width=4),
        ], className="mb-4"),
        dbc.Row([dbc.Col(dcc.Graph(id='sales-funnel-chart'), width=12)]),
        dbc.Row([dbc.Col(dcc.Graph(id='sales-over-time-chart'), width=12)]),
        dbc.Row([dbc.Col(dcc.Graph(id='sales-by-category-chart'), width=6), dbc.Col(dcc.Graph(id='top-products-chart'), width=6)]),
        dbc.Row([dbc.Col(dcc.Graph(id='sales-by-channel-chart'), width=6), dbc.Col(dcc.Graph(id='sales-by-city-chart'), width=6)]),
        dbc.Row([dbc.Col(dcc.Graph(id='sales-by-branch-chart'), width=12)]),
    ], fluid=True)

def create_delivery_layout():
    delivery_df = DATA.get('deliveries')
    if delivery_df is None or delivery_df.empty: return html.Div("Delivery Data Not Available", className="text-center mt-5")
    return dbc.Container([
        dbc.Row([
            dbc.Col(dcc.Dropdown(id='delivery-partner-filter', options=[{'label': 'All Partners', 'value': 'All'}] + [{'label': p, 'value': p} for p in sorted(delivery_df['deliverypartner'].unique())], value='All', clearable=False), width=6),
            dbc.Col(dcc.DatePickerRange(id='delivery-date-picker', min_date_allowed=delivery_df['date'].min(), max_date_allowed=delivery_df['date'].max(), start_date=delivery_df['date'].min(), end_date=delivery_df['date'].max()), width=6),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(id='kpi-on-time-delivery', color="success", inverse=True), width=3),
            dbc.Col(dbc.Card(id='kpi-failed-delivery', color="danger", inverse=True), width=3),
            dbc.Col(dbc.Card(id='kpi-avg-delivery-time', color="warning", inverse=True), width=3),
            dbc.Col(dbc.Card(id='kpi-avg-delivery-cost', color="info", inverse=True), width=3),
        ], className="mb-4"),
        dbc.Row([dbc.Col(dcc.Graph(id='delivery-pipeline-chart'), width=12)]),
        dbc.Row([dbc.Col(dcc.Graph(id='avg-time-by-city-chart'), width=6), dbc.Col(dcc.Graph(id='partner-performance-chart'), width=6)]),
    ], fluid=True)

def create_customer_layout():
    customer_analysis_df = DATA.get('customer_analysis')
    if customer_analysis_df is None or customer_analysis_df.empty: return html.Div("Customer or Sales Data Not Available", className="text-center mt-5")
    return dbc.Container([
        dbc.Row([
            dbc.Col(dbc.Card(id='kpi-total-customers', color="primary", inverse=True), width=3),
            dbc.Col(dbc.Card(id='kpi-active-customers', color="success", inverse=True), width=3),
            dbc.Col(dbc.Card(id='kpi-dormant-customers', color="warning", inverse=True), width=3),
            dbc.Col(dbc.Card(id='kpi-churn-risk', color="danger", inverse=True), width=3),
        ], className="mb-4"),
        dbc.Row([dbc.Col(dcc.Graph(id='customer-status-dist-chart'), width=12)]),
        html.Hr(),
        dbc.Row([
            dbc.Col(html.H4("Actionable Customer Lists"), width=6),
            dbc.Col(dcc.RadioItems(id='customer-list-selector', options=[{'label': 'Top-Value Customers', 'value': 'top_value'}, {'label': 'High Churn Risk', 'value': 'churn_risk'}, {'label': 'New Customers', 'value': 'new'}], value='top_value', inline=True, labelStyle={'margin-right': '20px'}), width=6),
        ]),
        dbc.Row([
            dbc.Col(dash_table.DataTable(id='customer-data-table', style_cell={'textAlign': 'left'}, style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}, page_size=10), width=10, className="mt-3"),
            dbc.Col(dbc.Button("Export Current List", id="export-csv-button", color="primary", className="mt-3"), width=2),
        ]),
    ], fluid=True)

def create_competitor_layout():
    competitor_df = DATA.get('competitors')
    price_comparison_df = DATA.get('price_comparison')
    if competitor_df is None or competitor_df.empty or price_comparison_df is None or price_comparison_df.empty:
        return html.Div("Competitor or Sales Data Not Available", className="text-center mt-5")
    return dbc.Container([
        dbc.Row([
            dbc.Col(dbc.Card(id='kpi-price-advantage', color="success", inverse=True), width=4),
            dbc.Col(dbc.Card(id='kpi-price-disadvantage', color="danger", inverse=True), width=4),
            dbc.Col(dbc.Card(id='kpi-promo-frequency', color="info", inverse=True), width=4),
        ], className="mb-4"),
        dbc.Row([dbc.Col(dcc.Graph(id='price-comparison-scatter-chart'), width=12)]),
        dbc.Row([dbc.Col(dcc.Graph(id='promo-analysis-chart'), width=6), dbc.Col(dcc.Graph(id='assortment-overlap-chart'), width=6)]),
    ], fluid=True)

def create_marketing_layout():
    campaign_performance_df = DATA.get('campaign_performance')
    if campaign_performance_df is None or campaign_performance_df.empty:
        return html.Div("Marketing Campaign Data Not Available", className="text-center mt-5")
    return dbc.Container([
        dbc.Row([
            dbc.Col(dbc.Card(id='kpi-total-ad-spend', color="info", inverse=True), width=3),
            dbc.Col(dbc.Card(id='kpi-avg-roas', color="success", inverse=True), width=3),
            dbc.Col(dbc.Card(id='kpi-avg-cpa', color="warning", inverse=True), width=3),
            dbc.Col(dbc.Card(id='kpi-total-conversions', color="primary", inverse=True), width=3),
        ], className="mb-4"),
        dbc.Row([dbc.Col(dcc.Graph(id='roas-by-campaign-chart'), width=12)]),
        dbc.Row([dbc.Col(dcc.Graph(id='cpa-by-campaign-chart'), width=6), dbc.Col(dcc.Graph(id='conversions-by-channel-chart'), width=6)]),
    ], fluid=True)

def create_profit_layout():
    profit_df = DATA.get('profit')
    if profit_df is None or profit_df.empty:
        return html.Div("Profit calculation requires Sales, Delivery, and Marketing data.", className="text-center mt-5")
    return dbc.Container([
        dbc.Row([
            dbc.Col(dbc.Card(id='kpi-total-net-profit', color="success", inverse=True), width=4),
            dbc.Col(dbc.Card(id='kpi-avg-profit-margin', color="primary", inverse=True), width=4),
            dbc.Col(dbc.Card(id='kpi-profit-lost-returns', color="danger", inverse=True), width=4),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col([html.H4("Key Profit Drivers"), dcc.Graph(id='profit-by-channel-chart'), dcc.Graph(id='profit-by-category-chart')], width=6),
            dbc.Col([html.H4("Actionable Recommendations"), dbc.Card(dbc.CardBody(id='automated-recommendations-list'), className="mb-4", style={"height": "95%"})], width=6),
        ]),
        html.Hr(),
        dbc.Row([dbc.Col(dcc.Graph(id='high-margin-products-chart'), width=6), dbc.Col(dcc.Graph(id='low-margin-products-chart'), width=6)]),
    ], fluid=True)

def create_predictive_layout():
    predictions_df = DATA.get('predictions')
    if predictions_df is None or predictions_df.empty:
        return html.Div([html.H4("Churn Prediction Model Not Available"), html.P("Please run 'model_trainer.py' to enable this feature.")], className="text-center mt-5")
    return dbc.Container([
        dbc.Row([
            dbc.Col(dbc.Card(id='kpi-high-risk-customers', color="danger", inverse=True), width=4),
            dbc.Col(dbc.Card(id='kpi-med-risk-customers', color="warning", inverse=True), width=4),
            dbc.Col(dbc.Card(id='kpi-low-risk-customers', color="success", inverse=True), width=4),
        ], className="mb-4"),
        dbc.Row([dbc.Col(dcc.Graph(id='churn-risk-distribution-chart'), width=12)]),
        html.Hr(),
        dbc.Row([
            dbc.Col(html.H4("High-Risk Customer List for Retention Campaign"), width=10),
            dbc.Col(dbc.Button("Export High-Risk List", id="export-churn-button"), width=2),
        ]),
        dbc.Row([
            dbc.Col(dash_table.DataTable(
                id='churn-data-table',
                data=predictions_df[predictions_df['churn_probability'] > 0.7].to_dict('records'),
                columns=[
                    {'name': 'Customer ID', 'id': 'customerid'}, {'name': 'City', 'id': 'city'},
                    {'name': 'Segment', 'id': 'segment'},
                    {'name': 'Churn Probability', 'id': 'churn_probability', 'format': {'specifier': '.2%'}},
                    {'name': 'Days Since Last Purchase', 'id': 'Recency'},
                ],
                style_cell={'textAlign': 'left'}, style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
                page_size=10, sort_action='native'
            ), className="mt-3")
        ]),
    ], fluid=True)

