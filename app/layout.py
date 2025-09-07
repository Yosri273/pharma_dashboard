# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# UI Layouts Module - V23.0 (Predictive Analytics Extension)
#
# Predictive layout replaced with multi-tab forecasting and churn analysis.
# Added new reusable components and client-side stores for models.
# -----------------------------------------------------------------------------

import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
import pandas as pd

# FIX: Import the global data store from its new location in etl.transforms
from etl.transforms import DATA

# --- 1. REUSABLE UI COMPONENTS ---

def create_kpi_card(title: str, kpi_id: str, color: str, width: int = 4, md_width: int = 6) -> dbc.Col:
    """Creates a Bootstrap Column containing a KPI Card."""
    # This is already mobile-first: sm=12 stacks it vertically on small screens.
    return dbc.Col(dbc.Card(id=kpi_id, color=color, inverse=True), lg=width, md=md_width, sm=12, class_name="mb-4")

def create_graph_card(graph_id: str, title: str = None, width: int = 6, lg_width: int = None) -> dbc.Col:
    """
    Creates a Bootstrap Column containing a Graph component in a Card.
    Added title parameter and responsive lg_width.
    """
    card_content = [dcc.Graph(id=graph_id)]
    if title:
        card_content.insert(0, html.H5(title, className="card-title"))
        
    # Use lg_width if provided, otherwise default to width
    lg_col_width = lg_width if lg_width is not None else width
        
    return dbc.Col(
        dbc.Card(dbc.CardBody(card_content)), 
        lg=lg_col_width, 
        md=width, 
        sm=12, # This sm=12 is key for mobile responsiveness
        class_name="mb-4"
    )

def create_datatable_card(table_id: str, title: str, width: int = 6, lg_width: int = None) -> dbc.Col:
    """
    NEW: Creates a Bootstrap Column containing a DataTable in a Card.
    MODIFIED: Added style_table={'overflowX': 'auto'} to prevent mobile page scroll.
    """
    lg_col_width = lg_width if lg_width is not None else width
    
    return dbc.Col(
        dbc.Card(dbc.CardBody([
            html.H5(title, className="card-title"),
            dash_table.DataTable(
                id=table_id,
                style_cell={'textAlign': 'left'},
                style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
                page_size=10,
                sort_action='native',
                # --- MOBILE OPTIMIZATION ADDED ---
                # This makes the table scroll horizontally *within the card* # instead of breaking the entire page layout on mobile.
                style_table={'overflowX': 'auto'}
            )
        ])),
        lg=lg_col_width,
        md=width,
        sm=12,
        class_name="mb-4"
    )


# --- 2. DASHBOARD LAYOUT FUNCTIONS ---
# (All existing layouts are preserved - they are already responsive via the reusable components)

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
                # Use our mobile-friendly datatable card here
                create_datatable_card(table_id='customer-data-table', title="", width=10),
                dbc.Col(dbc.Button(["Export ", html.I(className="bi bi-download")], id="export-csv-button", color="primary", className="mt-3 w-100"), lg=2, md=12, sm=12),
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


# --- NEW: PREDICTIVE ANALYTICS LAYOUT (COMPLETE REPLACEMENT) ---

def _create_forecast_tab() -> dbc.Tab:
    """NEW HELPER: Layout for Forecasting and Promo Simulation."""
    return dbc.Tab(label="Demand Forecasting & Promotion Simulation", children=[
        dbc.Row([
            # Controls - will stack on mobile (lg=3, md=12)
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H5("Simulation Controls"),
                    dbc.Label("Forecast Horizon (Days):"),
                    dcc.Slider(
                        id="forecast-slider-days", min=30, max=180, step=30, value=90, 
                        marks={30:'30', 90:'90', 180:'180'}, tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Hr(),
                    dbc.Label("Simulate Promotion Uplift (% Increase):"),
                    dcc.Slider(
                        id="forecast-slider-promo", min=0, max=50, step=5, value=0, 
                        marks={i: f"{i}%" for i in range(0, 51, 10)}, tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    dbc.Button("Run Simulation", id="forecast-run-button", color="primary", className="mt-4 w-100")
                ])
            ]), lg=3, md=12, className="mb-4"), # md=12 ensures it's full-width on tablet and mobile
            
            # KPIs and Main Graph
            dbc.Col([
                dbc.Row([
                    create_kpi_card("Forecasted Revenue (Baseline)", "pred-kpi-forecast-rev", "primary", width=6, md_width=6),
                    create_kpi_card("Simulated Revenue Lift (Promo)", "pred-kpi-sim-lift", "success", width=6, md_width=6),
                ]),
                dbc.Row([
                    create_graph_card(
                        graph_id="forecast-simulation-chart", 
                        title="Demand Forecast & Promotion Simulation", 
                        width=12
                    )
                ])
            ], lg=9, md=12)
        ], className="mt-3")
    ])

def _create_churn_tab() -> dbc.Tab:
    """NEW HELPER: Layout for Customer Churn Prediction."""
    return dbc.Tab(label="Customer Churn & LTV", children=[
         dbc.Row([
            # KPIs stack correctly with sm=12 (via md_width=6 and the default component logic)
            create_kpi_card("Predicted Churn Rate", "pred-kpi-churn-rate", "danger", width=3, md_width=6),
            create_kpi_card("Model AUC Score", "pred-kpi-churn-auc", "info", width=3, md_width=6),
            create_kpi_card("Total At-Risk Revenue", "pred-kpi-churn-revenue", "warning", width=3, md_width=6),
            create_kpi_card("Avg. LTV (Active Customer)", "pred-kpi-ltv", "success", width=3, md_width=6),
         ], className="mt-3"),
         
         dbc.Row([
            # Key Drivers Chart (stacks first on mobile)
            create_graph_card(
                graph_id="churn-key-drivers-chart", 
                title="Key Drivers of Churn (Feature Importance)", 
                lg_width=5,
                width=5 # This becomes md=12, and sm=12 is the default, so it's full-width on mobile
            ),
            
            # At-Risk Customer Table (stacks second on mobile)
            create_datatable_card(
                table_id="churn-at-risk-table", 
                title="Top Customers At-Risk of Churn",
                lg_width=7,
                width=7 # Full-width on mobile
            )
         ])
    ])

def create_predictive_layout() -> dbc.Container:
    """
    NEW: Creates the layout for the Predictive Insights dashboard.
    This function replaces the original. It provides sub-tabs for the
    new predictive models (Forecasting and Churn).
    """
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H4("Predictive Analytics & Simulations"), width=12, className="mb-3")
        ]),
        
        # Sub-tabs for different models
        dbc.Tabs([
            _create_forecast_tab(),
            _create_churn_tab()
        ])
    ], fluid=True)


# --- 3. MAIN APPLICATION LAYOUT ---
def create_main_layout() -> html.Div:
    """
    Creates the main application layout.
    MODIFIED: Replaced NavbarSimple with a fully collapsible Navbar for mobile.
    """
    
    # --- NEW: Collapsible Navbar ---
    # We replace NavbarSimple with a full Navbar to get the mobile hamburger menu
    navbar = dbc.Navbar(
        dbc.Container([
            html.A(
                dbc.Row(
                    [
                        # You could add a logo/icon here with dbc.Col
                        dbc.Col(dbc.NavbarBrand("Pharma Analytics Hub", className="ms-2")),
                    ],
                    align="center",
                    className="g-0", # g-0 removes gutters
                ),
                href="#",
                style={"textDecoration": "none"},
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            dbc.Collapse(
                dbc.Nav(
                    [
                        dbc.Button(
                            ["Refresh Data ", html.I(className="bi bi-arrow-clockwise")],
                            id="refresh-data-button", color="secondary"
                        )
                    ], 
                    # 'ms-auto' pushes the button to the right on desktop
                    # 'p-2' adds padding on mobile when it's stacked vertically
                    className="ms-auto p-2", 
                    navbar=True
                ),
                id="navbar-collapse",
                is_open=False,
                navbar=True,
            ),
        ], fluid=True),
        color="primary",
        dark=True,
        className="mb-4"
    )
    
    return html.Div([
        dcc.Store(id='data-store-trigger'),
        dcc.Download(id="download-dataframe-csv"),
        
        # --- Client-side model stores (Good for performance!) ---
        dcc.Store(id='store-forecast-model'),
        dcc.Store(id='store-churn-model'),
        
        navbar, # Use the new navbar object
        dbc.Container([
            dbc.Tabs(id="tabs-controller", active_tab="sales-tab", children=[
                dbc.Tab(label="Sales", tab_id="sales-tab"),
                dbc.Tab(label="Logistics", tab_id="delivery-tab"),
                dbc.Tab(label="Customers", tab_id="customer-tab"),
                dbc.Tab(label="Marketing", tab_id="marketing-tab"),
                dbc.Tab(label="Profit Optimization", tab_id="profit-tab"),
                dbc.Tab(label="Predictive Insights", tab_id="predictive-tab"),
            ]),
            html.Div(id='tab-content', className="mt-4")
        ], fluid=True)
    ])