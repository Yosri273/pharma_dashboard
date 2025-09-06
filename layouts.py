# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# UI Layouts Module - V22.1 (Mobile Responsive Refactor)
#
# This module is refactored for a mobile-first responsive design.
# - Replaced generic 'width' and 'md' props with explicit lg/md/sm props.
# - All components now stack vertically on small screens (sm=12).
# - DataTables are configured with horizontal scroll to prevent page overflow.
# -----------------------------------------------------------------------------

import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
import pandas as pd

# Import the global data store
from data import DATA

# --- 1. REUSABLE UI COMPONENTS (Mobile Optimized) ---

def create_kpi_card(title: str, kpi_id: str, color: str, lg_width: int = 4, md_width: int = 6) -> dbc.Col:
    """
    Creates a responsive KPI Card Column.
    Stacks on mobile (sm=12), fits 2 per row on tablet (md=6), and 3 (or specified) on desktop (lg=4).
    """
    return dbc.Col(dbc.Card(id=kpi_id, color=color, inverse=True), lg=lg_width, md=md_width, sm=12, class_name="mb-4")

def create_graph_card(graph_id: str, lg_width: int = 6) -> dbc.Col:
    """
    Creates a responsive Graph Card Column.
    Stacks on mobile (sm=12), stacks on tablet (md=12) for readability, and allows side-by-side on desktop (lg=6).
    """
    # Note: Defaulting md=12 ensures graphs are full-width and readable on tablets.
    return dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id=graph_id))), lg=lg_width, md=12, sm=12, class_name="mb-4")

# --- 2. DASHBOARD LAYOUT FUNCTIONS ---

def create_sales_layout() -> dbc.Container:
    """Creates the layout for the Sales Command Center."""
    sales_df = DATA.get('sales', pd.DataFrame())
    if sales_df.empty:
        return dbc.Container(html.H4("Sales Data Not Available", className="text-center mt-5"))
    
    return dbc.Container([
        dbc.Card(dbc.CardBody([
            dbc.Row([
                # FIX: Cols now use 'lg' for desktop and 'sm=12' to stack on mobile.
                # Added 'mb-2 mb-lg-0' to add margin on mobile only.
                dbc.Col(dcc.Dropdown(id='channel-filter-dropdown', options=[{'label': 'All Channels', 'value': 'All'}] + [{'label': ch, 'value': ch} for ch in sorted(sales_df['channel'].unique())], value='All', clearable=False), lg=5, sm=12, class_name="mb-2 mb-lg-0"),
                dbc.Col(dcc.DatePickerRange(id='sales-date-picker', min_date_allowed=sales_df['date'].min(), max_date_allowed=sales_df['date'].max(), start_date=sales_df['date'].min(), end_date=sales_df['date'].max()), lg=5, sm=12, class_name="mb-2 mb-lg-0"),
                dbc.Col(dcc.RadioItems(id='time-agg-selector', options=[{'label': 'Daily', 'value': 'date'}, {'label': 'Weekly', 'value': 'week'}, {'label': 'Monthly', 'value': 'month'}], value='date', inline=True), lg=2, sm=12),
            ], align="center"),
        ]), className="mb-4"),
        dbc.Row([
            create_kpi_card("Total Revenue", "kpi-total-revenue", "primary", lg_width=4),
            create_kpi_card("Gross Margin", "kpi-gross-margin", "success", lg_width=4),
            create_kpi_card("Net Profit", "kpi-net-profit", "dark", lg_width=4),
        ]),
        dbc.Row([
            create_kpi_card("Total Orders", "kpi-total-orders", "info", lg_width=4),
            create_kpi_card("Avg Order Value", "kpi-aov", "secondary", lg_width=4),
            create_kpi_card("Return Rate", "kpi-return-rate", "danger", lg_width=4),
        ]),
        dbc.Row([create_graph_card('sales-funnel-chart', lg_width=12)]),
        dbc.Row([create_graph_card('sales-over-time-chart', lg_width=12)]),
        dbc.Row([
            create_graph_card('sales-by-category-chart', lg_width=6), 
            create_graph_card('top-products-chart', lg_width=6)
        ]),
        dbc.Row([
            create_graph_card('sales-by-channel-chart', lg_width=6), 
            create_graph_card('sales-by-city-chart', lg_width=6)
        ]),
        dbc.Row([create_graph_card('sales-by-branch-chart', lg_width=12)]),
    ], fluid=True)

def create_delivery_layout() -> dbc.Container:
    """Creates the layout for the Logistics Command Center."""
    delivery_df = DATA.get('deliveries', pd.DataFrame())
    if delivery_df.empty:
        return dbc.Container(html.H4("Delivery Data Not Available", className="text-center mt-5"))
    return dbc.Container([
        dbc.Card(dbc.CardBody([
            dbc.Row([
                # FIX: Cols now use 'lg' and 'sm=12' to stack filters on mobile.
                dbc.Col(dcc.Dropdown(id='delivery-partner-filter', options=[{'label': 'All Partners', 'value': 'All'}] + [{'label': p, 'value': p} for p in sorted(delivery_df['deliverypartner'].unique())], value='All', clearable=False), lg=6, sm=12, class_name="mb-2 mb-lg-0"),
                dbc.Col(dcc.DatePickerRange(id='delivery-date-picker', min_date_allowed=delivery_df['date'].min(), max_date_allowed=delivery_df['date'].max(), start_date=delivery_df['date'].min(), end_date=delivery_df['date'].max()), lg=6, sm=12),
            ]),
        ]), className="mb-4"),
        dbc.Row([
            create_kpi_card("On-Time Rate", "kpi-on-time-delivery", "success", lg_width=3, md_width=6),
            create_kpi_card("Failed Delivery Rate", "kpi-failed-delivery", "danger", lg_width=3, md_width=6),
            create_kpi_card("Avg. Delivery Time", "kpi-avg-delivery-time", "warning", lg_width=3, md_width=6),
            create_kpi_card("Avg. Cost per Delivery", "kpi-avg-delivery-cost", "info", lg_width=3, md_width=6),
        ]),
        dbc.Row([create_graph_card('delivery-pipeline-chart', lg_width=12)]),
        dbc.Row([
            create_graph_card('avg-time-by-city-chart', lg_width=6), 
            create_graph_card('partner-performance-chart', lg_width=6)
        ]),
    ], fluid=True)

def create_customer_layout() -> dbc.Container:
    """Creates the layout for the Customer Action Center."""
    customer_analysis_df = DATA.get('customer_analysis_df', pd.DataFrame())
    if customer_analysis_df.empty:
        return dbc.Container(html.H4("Customer or Sales Data Not Available", className="text-center mt-5"))
    return dbc.Container([
        dbc.Row([
            create_kpi_card("Total Customers", "kpi-total-customers", "primary", lg_width=3, md_width=6),
            create_kpi_card("Active Customers", "kpi-active-customers", "success", lg_width=3, md_width=6),
            create_kpi_card("Dormant Customers", "kpi-dormant-customers", "warning", lg_width=3, md_width=6),
            create_kpi_card("High Churn Risk", "kpi-churn-risk", "danger", lg_width=3, md_width=6),
        ]),
        dbc.Row([create_graph_card('customer-status-dist-chart', lg_width=12)]),
        html.Hr(className="my-4"),
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col(html.H4("Actionable Customer Lists"), lg=6, sm=12),
                dbc.Col(dcc.RadioItems(id='customer-list-selector', options=[{'label': 'Top-Value Customers', 'value': 'top_value'}, {'label': 'High Churn Risk', 'value': 'churn_risk'}, {'label': 'New Customers', 'value': 'new'}], value='top_value', inline=True, labelClassName="me-3"), lg=6, sm=12),
            ], align="center", class_name="mb-3"),
            dbc.Row([
                # FIX: Table Col is now responsive (lg=10, sm=12)
                dbc.Col(dash_table.DataTable(
                    id='customer-data-table', 
                    style_cell={'textAlign': 'left', 'fontSize': '0.9rem'}, 
                    style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'}, 
                    page_size=10,
                    # CRITICAL FIX: Allows table to scroll horizontally on mobile without breaking the page.
                    style_table={'overflowX': 'auto'}
                ), lg=10, sm=12, className="mt-3"),
                # FIX: Button Col stacks under table on mobile (lg=2, sm=12)
                dbc.Col(dbc.Button(["Export ", html.I(className="bi bi-download")], id="export-csv-button", color="primary", className="mt-3 w-100"), lg=2, sm=12),
            ]),
        ])),
    ], fluid=True)

def create_competitor_layout() -> dbc.Container:
    """Creates the layout for the Market Intelligence dashboard."""
    # (This layout has no tables or complex filters, so only graph/kpi updates are needed)
    competitor_df = DATA.get('competitors', pd.DataFrame())
    price_comparison_df = DATA.get('price_comparison_df', pd.DataFrame())
    if competitor_df.empty or price_comparison_df.empty:
        return dbc.Container(html.H4("Competitor or Sales Data Not Available", className="text-center mt-5"))
    return dbc.Container([
        dbc.Row([
            create_kpi_card("Products We Undercut", "kpi-price-advantage", "success", lg_width=4),
            create_kpi_card("Products More Expensive", "kpi-price-disadvantage", "danger", lg_width=4),
            create_kpi_card("Avg. Competitor Promo Rate", "kpi-promo-frequency", "info", lg_width=4),
        ]),
        dbc.Row([create_graph_card('price-comparison-scatter-chart', lg_width=12)]),
        dbc.Row([
            create_graph_card('promo-analysis-chart', lg_width=6), 
            create_graph_card('assortment-overlap-chart', lg_width=6)
        ]),
    ], fluid=True)

def create_marketing_layout() -> dbc.Container:
    """Creates the layout for the Marketing Effectiveness dashboard."""
    campaign_performance_df = DATA.get('campaign_performance_df', pd.DataFrame())
    if campaign_performance_df.empty:
        return dbc.Container(html.H4("Marketing Campaign Data Not Available", className="text-center mt-5"))
    return dbc.Container([
        dbc.Row([
            create_kpi_card("Total Ad Spend", "kpi-total-ad-spend", "info", lg_width=3, md_width=6),
            create_kpi_card("Overall ROAS", "kpi-avg-roas", "success", lg_width=3, md_width=6),
            create_kpi_card("Average CPA", "kpi-avg-cpa", "warning", lg_width=3, md_width=6),
            create_kpi_card("Attributed Conversions", "kpi-total-conversions", "primary", lg_width=3, md_width=6),
        ]),
        dbc.Row([create_graph_card('roas-by-campaign-chart', lg_width=12)]),
        dbc.Row([
            create_graph_card('cpa-by-campaign-chart', lg_width=6), 
            create_graph_card('conversions-by-channel-chart', lg_width=6)
        ]),
    ], fluid=True)

def create_profit_layout() -> dbc.Container:
    """Creates the layout for the Profit Optimization dashboard."""
    profit_df = DATA.get('profit_df', pd.DataFrame())
    if profit_df.empty:
        return dbc.Container(html.H4("Profit calculation requires Sales, Delivery, and Marketing data.", className="text-center mt-5"))
    return dbc.Container([
        dbc.Row([
            create_kpi_card("Total Net Profit", "kpi-total-net-profit", "success", lg_width=4),
            create_kpi_card("Average Profit Margin", "kpi-avg-profit-margin", "primary", lg_width=4),
            create_kpi_card("Profit Lost to Returns", "kpi-profit-lost-returns", "danger", lg_width=4),
        ]),
        dbc.Row([
            # FIX: Stack these two main columns on mobile/tablet (lg=6, md=12, sm=12)
            dbc.Col([html.H4("Key Profit Drivers", className="mb-3"), 
                     create_graph_card('profit-by-channel-chart', lg_width=12), 
                     create_graph_card('profit-by-category-chart', lg_width=12)], lg=6, md=12, sm=12),
            dbc.Col([html.H4("Actionable Recommendations", className="mb-3"), 
                     dbc.Card(dbc.CardBody(id='automated-recommendations-list'), style={"height": "95%"})], lg=6, md=12, sm=12),
        ]),
        html.Hr(className="my-4"),
        dbc.Row([
            create_graph_card('high-margin-products-chart', lg_width=6), 
            create_graph_card('low-margin-products-chart', lg_width=6)
        ]),
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
            create_kpi_card("High Risk (>70%)", "kpi-high-risk-customers", "danger", lg_width=4), # Updated from 3 to 4 for better spacing
            create_kpi_card("Medium Risk (40-70%)", "kpi-med-risk-customers", "warning", lg_width=4),
            create_kpi_card("Low Risk (<40%)", "kpi-low-risk-customers", "success", lg_width=4),
        ]),
        dbc.Row([create_graph_card('churn-risk-distribution-chart', lg_width=12)]),
        html.Hr(className="my-4"),
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col(html.H4("High-Risk Customer List for Retention Campaign"), lg=10, sm=12, class_name="mb-2 mb-lg-0"),
                dbc.Col(dbc.Button(["Export List ", html.I(className="bi bi-download")], id="export-churn-button", color="secondary", className="w-100"), lg=2, sm=12),
            ]),
            dbc.Row([
                dbc.Col(dash_table.DataTable(
                    id='churn-data-table', 
                    style_cell={'textAlign': 'left', 'fontSize': '0.9rem'}, 
                    style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'}, 
                    page_size=10, 
                    sort_action='native',
                    # CRITICAL FIX: Allows table to scroll horizontally on mobile.
                    style_table={'overflowX': 'auto'}
                ), className="mt-3")
            ]),
        ])),
    ], fluid=True)

# --- 3. MAIN APPLICATION LAYOUT (Mobile Optimized) ---
def create_main_layout() -> html.Div:
    """Creates the main application layout, using Card components for cleaner tab containerization."""
    navbar = dbc.NavbarSimple(
        children=[
            dbc.Button(
                ["Refresh Data ", html.I(className="bi bi-arrow-clockwise")],
                id="refresh-data-button", color="secondary", className="ms-auto"
            ),
        ],
        brand="Pharma Analytics Hub", brand_href="#", color="primary", dark=True
    )
    
    tabs_control = dbc.Tabs(
        id="tabs-controller", 
        active_tab="sales-tab", 
        card=True, # FIX: Use 'card=True' styling for a cleaner, encapsulated tab look
        children=[
            dbc.Tab(label="Sales", tab_id="sales-tab"),
            dbc.Tab(label="Logistics", tab_id="delivery-tab"),
            dbc.Tab(label="Customers", tab_id="customer-tab"),
            dbc.Tab(label="Marketing", tab_id="marketing-tab"),
            dbc.Tab(label="Profit Optimization", tab_id="profit-tab"),
            dbc.Tab(label="Predictive Insights", tab_id="predictive-tab"),
    ])

    return html.Div([
        dcc.Store(id='data-store-trigger'),
        dcc.Download(id="download-dataframe-csv"),
        navbar,
        # FIX: Wrap tabs and content in a Card for a better container, and use CardBody for tab content.
        dbc.Container(
            dbc.Card([
                dbc.CardHeader(tabs_control),
                dbc.CardBody(html.Div(id='tab-content')) # Content renders here, padding is handled by CardBody
            ]),
            fluid=True,
            className="mt-4" # Add margin to the whole content card
        )
    ])