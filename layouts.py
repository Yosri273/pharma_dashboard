# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# UI Layouts Module - V22.0 (Modern UI Redesign)
#
# This module defines the visual structure of the application, focusing on a
# professional, modern, and clean aesthetic using a new theme and better
# component organization. This is the full, unabbreviated version.
# -----------------------------------------------------------------------------

import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
import pandas as pd

# Import the global data store to build dynamic components like dropdowns
from data import DATA

# --- 1. REUSABLE UI COMPONENTS ---

def create_kpi_card(title: str, kpi_id: str, color: str, width: int = 4, md_width: int = 6) -> dbc.Col:
    """Creates a Bootstrap Column containing a KPI Card."""
    return dbc.Col(dbc.Card(id=kpi_id, color=color, inverse=True), lg=width, md=md_width, sm=12, class_name="mb-4")

def create_graph_card(graph_id: str, width: int = 6) -> dbc.Col:
    """Creates a Bootstrap Column containing a Graph component in a Card."""
    return dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id=graph_id))), md=width, class_name="mb-4")

# --- 2. DASHBOARD LAYOUT FUNCTIONS ---

def create_sales_layout() -> dbc.Container:
    """Creates the layout for the Sales Command Center."""
    sales_df = DATA.get('sales', pd.DataFrame())
    if sales_df.empty:
        return dbc.Container(html.H4("Sales Data Not Available", className="text-center mt-5"))
    
    return dbc.Container([
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col(dcc.Dropdown(id='channel-filter-dropdown', options=[{'label': 'All Channels', 'value': 'All'}] + [{'label': ch, 'value': ch} for ch in sorted(sales_df['channel'].unique())], value='All', clearable=False), md=5),
                dbc.Col(dcc.DatePickerRange(id='sales-date-picker', min_date_allowed=sales_df['date'].min(), max_date_allowed=sales_df['date'].max(), start_date=sales_df['date'].min(), end_date=sales_df['date'].max()), md=5),
                dbc.Col(dcc.RadioItems(id='time-agg-selector', options=[{'label': 'Daily', 'value': 'date'}, {'label': 'Weekly', 'value': 'week'}, {'label': 'Monthly', 'value': 'month'}], value='date', inline=True), md=2),
            ], align="center"),
        ]), className="mb-4"),
        dbc.Row([
            create_kpi_card("Total Revenue", "kpi-total-revenue", "primary"),
            create_kpi_card("Gross Margin", "kpi-gross-margin", "success"),
            create_kpi_card("Net Profit", "kpi-net-profit", "dark"),
        ]),
        dbc.Row([
            create_kpi_card("Total Orders", "kpi-total-orders", "info"),
            create_kpi_card("Avg Order Value", "kpi-aov", "secondary"),
            create_kpi_card("Return Rate", "kpi-return-rate", "danger"),
        ]),
        dbc.Row([create_graph_card('sales-funnel-chart', width=12)]),
        dbc.Row([create_graph_card('sales-over-time-chart', width=12)]),
        dbc.Row([create_graph_card('sales-by-category-chart'), create_graph_card('top-products-chart')]),
        dbc.Row([create_graph_card('sales-by-channel-chart'), create_graph_card('sales-by-city-chart')]),
        dbc.Row([create_graph_card('sales-by-branch-chart', width=12)]),
    ], fluid=True)

def create_delivery_layout() -> dbc.Container:
    """Creates the layout for the Logistics Command Center."""
    delivery_df = DATA.get('deliveries', pd.DataFrame())
    if delivery_df.empty:
        return dbc.Container(html.H4("Delivery Data Not Available", className="text-center mt-5"))
    return dbc.Container([
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col(dcc.Dropdown(id='delivery-partner-filter', options=[{'label': 'All Partners', 'value': 'All'}] + [{'label': p, 'value': p} for p in sorted(delivery_df['deliverypartner'].unique())], value='All', clearable=False), md=6),
                dbc.Col(dcc.DatePickerRange(id='delivery-date-picker', min_date_allowed=delivery_df['date'].min(), max_date_allowed=delivery_df['date'].max(), start_date=delivery_df['date'].min(), end_date=delivery_df['date'].max()), md=6),
            ]),
        ]), className="mb-4"),
        dbc.Row([
            create_kpi_card("On-Time Rate", "kpi-on-time-delivery", "success", width=3, md_width=6),
            create_kpi_card("Failed Delivery Rate", "kpi-failed-delivery", "danger", width=3, md_width=6),
            create_kpi_card("Avg. Delivery Time", "kpi-avg-delivery-time", "warning", width=3, md_width=6),
            create_kpi_card("Avg. Cost per Delivery", "kpi-avg-delivery-cost", "info", width=3, md_width=6),
        ]),
        dbc.Row([create_graph_card('delivery-pipeline-chart', width=12)]),
        dbc.Row([create_graph_card('avg-time-by-city-chart'), create_graph_card('partner-performance-chart')]),
    ], fluid=True)

def create_customer_layout() -> dbc.Container:
    """Creates the layout for the Customer Action Center."""
    customer_analysis_df = DATA.get('customer_analysis_df', pd.DataFrame())
    if customer_analysis_df.empty:
        return dbc.Container(html.H4("Customer or Sales Data Not Available", className="text-center mt-5"))
    return dbc.Container([
        dbc.Row([
            create_kpi_card("Total Customers", "kpi-total-customers", "primary", width=3, md_width=6),
            create_kpi_card("Active Customers", "kpi-active-customers", "success", width=3, md_width=6),
            create_kpi_card("Dormant Customers", "kpi-dormant-customers", "warning", width=3, md_width=6),
            create_kpi_card("High Churn Risk", "kpi-churn-risk", "danger", width=3, md_width=6),
        ]),
        dbc.Row([create_graph_card('customer-status-dist-chart', width=12)]),
        html.Hr(className="my-4"),
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col(html.H4("Actionable Customer Lists"), md=6),
                dbc.Col(dcc.RadioItems(id='customer-list-selector', options=[{'label': 'Top-Value Customers', 'value': 'top_value'}, {'label': 'High Churn Risk', 'value': 'churn_risk'}, {'label': 'New Customers', 'value': 'new'}], value='top_value', inline=True, labelClassName="me-3"), md=6),
            ], align="center"),
            dbc.Row([
                dbc.Col(dash_table.DataTable(id='customer-data-table', style_cell={'textAlign': 'left'}, style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'}, page_size=10), width=10, className="mt-3"),
                dbc.Col(dbc.Button(["Export ", html.I(className="bi bi-download")], id="export-csv-button", color="primary", className="mt-3 w-100"), width=2),
            ]),
        ])),
    ], fluid=True)

def create_competitor_layout() -> dbc.Container:
    """Creates the layout for the Market Intelligence dashboard."""
    competitor_df = DATA.get('competitors', pd.DataFrame())
    price_comparison_df = DATA.get('price_comparison_df', pd.DataFrame())
    if competitor_df.empty or price_comparison_df.empty:
        return dbc.Container(html.H4("Competitor or Sales Data Not Available", className="text-center mt-5"))
    return dbc.Container([
        dbc.Row([
            create_kpi_card("Products We Undercut", "kpi-price-advantage", "success"),
            create_kpi_card("Products More Expensive", "kpi-price-disadvantage", "danger"),
            create_kpi_card("Avg. Competitor Promo Rate", "kpi-promo-frequency", "info"),
        ]),
        dbc.Row([create_graph_card('price-comparison-scatter-chart', width=12)]),
        dbc.Row([create_graph_card('promo-analysis-chart'), create_graph_card('assortment-overlap-chart')]),
    ], fluid=True)

def create_marketing_layout() -> dbc.Container:
    """Creates the layout for the Marketing Effectiveness dashboard."""
    campaign_performance_df = DATA.get('campaign_performance_df', pd.DataFrame())
    if campaign_performance_df.empty:
        return dbc.Container(html.H4("Marketing Campaign Data Not Available", className="text-center mt-5"))
    return dbc.Container([
        dbc.Row([
            create_kpi_card("Total Ad Spend", "kpi-total-ad-spend", "info", width=3, md_width=6),
            create_kpi_card("Overall ROAS", "kpi-avg-roas", "success", width=3, md_width=6),
            create_kpi_card("Average CPA", "kpi-avg-cpa", "warning", width=3, md_width=6),
            create_kpi_card("Attributed Conversions", "kpi-total-conversions", "primary", width=3, md_width=6),
        ]),
        dbc.Row([create_graph_card('roas-by-campaign-chart', width=12)]),
        dbc.Row([create_graph_card('cpa-by-campaign-chart'), create_graph_card('conversions-by-channel-chart')]),
    ], fluid=True)

def create_profit_layout() -> dbc.Container:
    """Creates the layout for the Profit Optimization dashboard."""
    profit_df = DATA.get('profit_df', pd.DataFrame())
    if profit_df.empty:
        return dbc.Container(html.H4("Profit calculation requires Sales, Delivery, and Marketing data.", className="text-center mt-5"))
    return dbc.Container([
        dbc.Row([
            create_kpi_card("Total Net Profit", "kpi-total-net-profit", "success"),
            create_kpi_card("Average Profit Margin", "kpi-avg-profit-margin", "primary"),
            create_kpi_card("Profit Lost to Returns", "kpi-profit-lost-returns", "danger"),
        ]),
        dbc.Row([
            dbc.Col([html.H4("Key Profit Drivers", className="mb-3"), create_graph_card('profit-by-channel-chart', width=12), create_graph_card('profit-by-category-chart', width=12)], md=6),
            dbc.Col([html.H4("Actionable Recommendations", className="mb-3"), dbc.Card(dbc.CardBody(id='automated-recommendations-list'), style={"height": "95%"})], md=6),
        ]),
        html.Hr(className="my-4"),
        dbc.Row([create_graph_card('high-margin-products-chart'), create_graph_card('low-margin-products-chart')]),
    ], fluid=True)

def create_predictive_layout() -> dbc.Container:
    """Creates the layout for the Predictive Insights dashboard."""
    predictions_df = DATA.get('predictions_df', pd.DataFrame())
    if predictions_df.empty:
        return dbc.Container([
            html.H4("Churn Prediction Model Not Available", className="text-center mt-5"),
            html.P("Please run the 'model_trainer.py' script to enable this feature.", className="text-center")
        ])
    return dbc.Container([
        dbc.Row([
            create_kpi_card("High Risk (>70%)", "kpi-high-risk-customers", "danger"),
            create_kpi_card("Medium Risk (40-70%)", "kpi-med-risk-customers", "warning"),
            create_kpi_card("Low Risk (<40%)", "kpi-low-risk-customers", "success"),
        ]),
        dbc.Row([create_graph_card('churn-risk-distribution-chart', width=12)]),
        html.Hr(className="my-4"),
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col(html.H4("High-Risk Customer List for Retention Campaign"), width=10),
                dbc.Col(dbc.Button(["Export List ", html.I(className="bi bi-download")], id="export-churn-button", color="secondary"), width=2),
            ]),
            dbc.Row([
                dbc.Col(dash_table.DataTable(id='churn-data-table', style_cell={'textAlign': 'left'}, style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'}, page_size=10, sort_action='native'), className="mt-3")
            ]),
        ])),
    ], fluid=True)

# --- 3. MAIN APPLICATION LAYOUT ---
def create_main_layout() -> html.Div:
    """Creates the main application layout, including the new Navbar and tabs."""
    navbar = dbc.NavbarSimple(
        children=[
            dbc.Button(
                ["Refresh Data ", html.I(className="bi bi-arrow-clockwise")],
                id="refresh-data-button", color="secondary", className="ms-auto"
            ),
        ],
        brand="Pharma Analytics Hub", brand_href="#", color="primary", dark=True, className="mb-4"
    )
    return html.Div([
        dcc.Store(id='data-store-trigger'),
        dcc.Download(id="download-dataframe-csv"),
        navbar,
        dbc.Container([
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
    ])

