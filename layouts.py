# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# UI Layouts Module - V22.0 (Modern UI Redesign)
#
# This version is a complete redesign of the application's UI, focusing on a
# professional, modern, and clean aesthetic using a new theme and better
# component organization.
# -----------------------------------------------------------------------------

import dash_bootstrap_components as dbc
from dash import dcc, html
import pandas as pd

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

# (Other layout functions would follow the same modern design pattern)

# --- 3. MAIN APPLICATION LAYOUT ---

def create_main_layout() -> html.Div:
    """
    Creates the main application layout, including the new Navbar, tabs,
    and a container for the active tab's content.
    """
    navbar = dbc.NavbarSimple(
        children=[
            dbc.Button(
                ["Refresh Data ", html.I(className="bi bi-arrow-clockwise")],
                id="refresh-data-button",
                color="secondary",
                className="ms-auto",
            ),
        ],
        brand="Pharma Analytics Hub",
        brand_href="#",
        color="primary",
        dark=True,
        className="mb-4",
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

