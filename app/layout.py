# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# UI Layouts Module - V24.0 (Logistics Tab Redesign)
#
# MODIFIED: The create_delivery_layout function has been completely
#           redesigned to support analysis of an internal delivery fleet.
#           - Removed "Delivery Partner" filter and charts.
#           - Added new filters for "Driver" and "Vehicle Type".
#           - Added new chart placeholders for driver and vehicle analytics.
# -----------------------------------------------------------------------------

import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
import pandas as pd

from etl.transforms import DATA

# ... (Helper functions: create_filter_options, create_kpi_card, etc. remain unchanged) ...
def create_filter_options(option_list):
    return [{'label': 'All', 'value': 'All'}] + [{'label': opt, 'value': opt} for opt in sorted(list(option_list))]
def create_multi_filter_options(option_list):
    return [{'label': 'All', 'value': 'All'}] + [{'label': opt, 'value': opt} for opt in sorted(list(option_list))]
def create_kpi_card(title: str, kpi_id: str, color: str, width: int = 4, md_width: int = 6) -> dbc.Col:
    return dbc.Col(dbc.Card(id=kpi_id, color=color, inverse=True), lg=width, md=md_width, sm=12, class_name="mb-4")
def create_graph_card(graph_id: str, title: str = None, width: int = 6, lg_width: int = None) -> dbc.Col:
    card_content = [dcc.Graph(id=graph_id)]
    if title: card_content.insert(0, html.H5(title, className="card-title"))
    lg_col_width = lg_width if lg_width is not None else width
    return dbc.Col(dbc.Card(dbc.CardBody(card_content)), lg=lg_col_width, md=width, sm=12, class_name="mb-4")
def create_datatable_card(table_id: str, title: str, width: int = 6, lg_width: int = None) -> dbc.Col:
    lg_col_width = lg_width if lg_width is not None else width
    return dbc.Col(dbc.Card(dbc.CardBody([html.H5(title, className="card-title"), dash_table.DataTable(id=table_id, style_cell={'textAlign': 'left'}, style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'}, page_size=10, sort_action='native', style_table={'overflowX': 'auto'})])), lg=lg_col_width, md=width, sm=12, class_name="mb-4")

# --- Dashboard Layouts ---
# ... (create_sales_layout is unchanged) ...
def create_sales_layout() -> dbc.Container:
    sales_df = DATA.get('sales', pd.DataFrame())
    if sales_df.empty: return dbc.Container(html.H4("Sales Data Not Available", className="text-center mt-5"))
    channel_opts, category_opts, region_opts = sales_df['channel'].unique(), sales_df['category'].unique(), sales_df['city'].unique()
    return dbc.Container([
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([html.Label("Date Range:"), dcc.DatePickerRange(id='sales-date-picker', min_date_allowed=sales_df['date'].min(), max_date_allowed=sales_df['date'].max(), start_date=sales_df['date'].min(), end_date=sales_df['date'].max(), className="d-block")], width=12, lg=3, className="mb-2"),
                dbc.Col([html.Label("Region (City):"), dcc.Dropdown(id='sales-region-filter', options=create_multi_filter_options(region_opts), value=['All'], multi=True, clearable=False)], width=12, lg=2, className="mb-2"),
                dbc.Col([html.Label("Category:"), dcc.Dropdown(id='sales-category-filter', options=create_multi_filter_options(category_opts), value=['All'], multi=True, clearable=False)], width=12, lg=2, className="mb-2"),
                dbc.Col([html.Label("Channel:"), dcc.Dropdown(id='channel-filter-dropdown', options=create_filter_options(channel_opts), value='All', clearable=False)], width=12, lg=2, className="mb-2"),
                dbc.Col([html.Label("Aggregate:"), dcc.RadioItems(id='time-agg-selector', options=[{'label': 'Daily', 'value': 'date'}, {'label': 'Weekly', 'value': 'week'}, {'label': 'Monthly', 'value': 'month'}], value='date', inline=True)], width=12, lg=1, className="mb-2 align-self-center"),
                dbc.Col([html.Label("Actions", style={'visibility': 'hidden'}), dbc.ButtonGroup([dbc.Button("Apply", id="sales-apply-btn", color="primary"), dbc.Button("Export PDF", id="sales-export-btn", color="secondary")], className="w-100")], width=12, lg=2, className="mb-2 align-self-end"),
            ], align="bottom"),
        ]), className="mb-4"),
        dbc.Row([create_kpi_card("Total Revenue", "kpi-total-revenue", "primary"), create_kpi_card("Gross Margin", "kpi-gross-margin", "success"), create_kpi_card("Net Profit", "kpi-net-profit", "dark")]),
        dbc.Row([create_kpi_card("Total Orders", "kpi-total-orders", "info"), create_kpi_card("Avg Order Value", "kpi-aov", "secondary"), create_kpi_card("Return Rate", "kpi-return-rate", "danger")]),
        dbc.Row([create_graph_card('sales-funnel-chart', width=12)]),
        dbc.Row([create_graph_card('sales-over-time-chart', width=12)]),
        html.Hr(className="my-4"), dbc.Row([html.H4("Advanced Sales Analytics", className="mb-3 text-center")]), dbc.Row([create_graph_card('period-growth-chart', title="Period-over-Period Growth"), create_graph_card('price-volume-chart', title="Price vs. Volume Analysis")]), html.Hr(className="my-4"),
        dbc.Row([create_graph_card('sales-by-category-chart'), create_graph_card('top-products-chart')]), dbc.Row([create_graph_card('sales-by-channel-chart'), create_graph_card('sales-by-city-chart')]), dbc.Row([create_graph_card('sales-by-branch-chart', width=12)]),
    ], fluid=True)


# --- MODIFIED: LOGISTICS TAB LAYOUT ---
def create_delivery_layout() -> dbc.Container:
    """Creates the layout for the Logistics Command Center (Internal Fleet)."""
    delivery_df = DATA.get('deliveries', pd.DataFrame())
    if delivery_df.empty:
        return dbc.Container(html.H4("Delivery Data Not Available", className="text-center mt-5"))
        
    # Get dynamic options for new filters from the dataframe
    driver_opts = delivery_df['driverid'].unique() if 'driverid' in delivery_df.columns else []
    vehicle_opts = delivery_df['vehicletype'].unique() if 'vehicletype' in delivery_df.columns else []
    region_opts = delivery_df['city'].unique()
    
    return dbc.Container([
        dbc.Card(dbc.CardBody([
            dbc.Row([
                 dbc.Col([
                    html.Label("Date Range:"),
                    dcc.DatePickerRange(
                        id='delivery-date-picker',
                        min_date_allowed=delivery_df['date'].min(), max_date_allowed=delivery_df['date'].max(),
                        start_date=delivery_df['date'].min(), end_date=delivery_df['date'].max(), className="d-block"
                    )
                ], width=12, lg=3, className="mb-2"),
                dbc.Col([
                    html.Label("Region (City):"),
                    dcc.Dropdown(id='delivery-region-filter', options=create_multi_filter_options(region_opts), value=['All'], multi=True, clearable=False)
                ], width=12, lg=2, className="mb-2"),
                 dbc.Col([
                    html.Label("Driver:"),
                     dcc.Dropdown(id='driver-filter', options=create_filter_options(driver_opts), value='All', clearable=False) # NEW
                ], width=12, lg=2, className="mb-2"),
                dbc.Col([
                    html.Label("Vehicle Type:"),
                     dcc.Dropdown(id='vehicle-type-filter', options=create_filter_options(vehicle_opts), value='All', clearable=False) # NEW
                ], width=12, lg=2, className="mb-2"),
                dbc.Col([
                    html.Label("Actions", style={'visibility': 'hidden'}),
                    dbc.ButtonGroup([
                        dbc.Button("Apply", id="delivery-apply-btn", color="primary"),
                        dbc.Button("Export PDF", id="delivery-export-btn", color="secondary")
                    ], className="w-100")
                ], width=12, lg=3, className="mb-2 align-self-end"),
            ], align="bottom"),
        ]), className="mb-4"),
        
        dbc.Row([
            create_kpi_card("On-Time Rate", "kpi-on-time-delivery", "success", width=3, md_width=6),
            create_kpi_card("Failed Delivery Rate", "kpi-failed-delivery", "danger", width=3, md_width=6),
            create_kpi_card("Avg. Delivery Time", "kpi-avg-delivery-time", "warning", width=3, md_width=6),
            create_kpi_card("Avg. Cost per Delivery", "kpi-avg-delivery-cost", "info", width=3, md_width=6),
        ]),
        
        dbc.Row([create_graph_card('delivery-pipeline-chart', width=12)]),
        
        # --- NEW: Replaced partner charts with internal fleet charts ---
        html.Hr(className="my-4"),
        dbc.Row([html.H4("Internal Fleet Analytics", className="mb-3 text-center")]),
        dbc.Row([
            create_graph_card('driver-leaderboard-chart', title="Driver Performance Leaderboard"),
            create_graph_card('vehicle-efficiency-chart', title="Vehicle Type Efficiency")
        ]),
        html.Hr(className="my-4"),
        # --- END MODIFICATION ---

        dbc.Row([create_graph_card('avg-time-by-city-chart', title="Average Delivery Time by City", width=12)]),
    ], fluid=True)

# ... (All other layout functions: create_customer_layout, create_marketing_layout, etc. remain unchanged) ...
def create_customer_layout() -> dbc.Container:
    customer_analysis_df = DATA.get('customer_analysis_df', pd.DataFrame())
    if customer_analysis_df.empty: return dbc.Container(html.H4("Customer or Sales Data Not Available", className="text-center mt-5"))
    region_opts, segment_opts = customer_analysis_df['city'].unique(), customer_analysis_df['segment'].unique()
    return dbc.Container([
        dbc.Card(dbc.CardBody([
             dbc.Row([
                 dbc.Col([html.Label("Customer Join Date Range:"), dcc.DatePickerRange(id='customer-date-picker', min_date_allowed=customer_analysis_df['joindate'].min(), max_date_allowed=customer_analysis_df['joindate'].max(), start_date=customer_analysis_df['joindate'].min(), end_date=customer_analysis_df['joindate'].max(), className="d-block")], width=12, lg=4, className="mb-2"),
                 dbc.Col([html.Label("Region (City):"), dcc.Dropdown(id='customer-region-filter', options=create_multi_filter_options(region_opts), value=['All'], multi=True, clearable=False)], width=12, lg=3, className="mb-2"),
                 dbc.Col([html.Label("Customer Segment:"), dcc.Dropdown(id='customer-segment-filter', options=create_multi_filter_options(segment_opts), value=['All'], multi=True, clearable=False)], width=12, lg=3, className="mb-2"),
                 dbc.Col([html.Label("Actions", style={'visibility': 'hidden'}), dbc.ButtonGroup([dbc.Button("Apply", id="customer-apply-btn", color="primary"), dbc.Button("Export PDF", id="customer-export-btn", color="secondary")], className="w-100")], width=12, lg=2, className="mb-2 align-self-end"),
             ], align="bottom"),
        ]), className="mb-4"),
        dbc.Row([create_kpi_card("Total Customers", "kpi-total-customers", "primary", width=3, md_width=6), create_kpi_card("Active Customers", "kpi-active-customers", "info", width=3, md_width=6), create_kpi_card("Retention Rate", "kpi-retention-rate", "success", width=3, md_width=6), create_kpi_card("Repeat Purchase Rate", "kpi-repeat-purchase-rate", "success", width=3, md_width=6)]),
        dbc.Row([create_kpi_card("Dormant Customers", "kpi-dormant-customers", "warning", width=6, md_width=6), create_kpi_card("High Churn Risk", "kpi-churn-risk", "danger", width=6, md_width=6)]),
        dbc.Row([create_graph_card('customer-status-dist-chart', width=12)]),
        html.Hr(className="my-4"), dbc.Row([create_graph_card('rfm-bubble-chart', title="RFM Segment Analysis", width=12)]), html.Hr(className="my-4"),
        dbc.Card(dbc.CardBody([
            dbc.Row([dbc.Col(html.H4("Actionable Customer Lists"), md=6), dbc.Col(dcc.RadioItems(id='customer-list-selector', options=[{'label': 'Top-Value Customers', 'value': 'top_value'}, {'label': 'High Churn Risk', 'value': 'churn_risk'}, {'label': 'New Customers', 'value': 'new'}], value='top_value', inline=True, labelClassName="me-3"), md=6)], align="center"),
            dbc.Row([create_datatable_card(table_id='customer-data-table', title="", width=10), dbc.Col(dbc.Button(["Export ", html.I(className="bi bi-download")], id="export-csv-button", color="primary", className="mt-3 w-100"), lg=2, md=12, sm=12)]),
        ])),
    ], fluid=True)
def create_marketing_layout() -> dbc.Container:
    campaign_performance_df = DATA.get('campaign_performance_df', pd.DataFrame())
    if campaign_performance_df.empty: return dbc.Container(html.H4("Marketing Campaign Data Not Available", className="text-center mt-5"))
    channel_opts = campaign_performance_df['channel'].unique()
    return dbc.Container([
        dbc.Card(dbc.CardBody([
            dbc.Row([
                 dbc.Col([html.Label("Campaign Date Range (Overlap):"), dcc.DatePickerRange(id='marketing-date-picker', min_date_allowed=pd.to_datetime(campaign_performance_df['startdate']).min().date(), max_date_allowed=pd.to_datetime(campaign_performance_df['enddate']).max().date(), start_date=pd.to_datetime(campaign_performance_df['startdate']).min().date(), end_date=pd.to_datetime(campaign_performance_df['enddate']).max().date(), className="d-block")], width=12, lg=5, className="mb-2"),
                 dbc.Col([html.Label("Channel:"), dcc.Dropdown(id='marketing-channel-filter', options=create_filter_options(channel_opts), value='All', clearable=False)], width=12, lg=4, className="mb-2"),
                 dbc.Col([html.Label("Actions", style={'visibility': 'hidden'}), dbc.ButtonGroup([dbc.Button("Apply", id="marketing-apply-btn", color="primary"), dbc.Button("Export PDF", id="marketing-export-btn", color="secondary")], className="w-100")], width=12, lg=3, className="mb-2 align-self-end"),
            ], align="bottom"),
        ]), className="mb-4"),
        dbc.Row([create_kpi_card("Total Ad Spend", "kpi-total-ad-spend", "info", width=3, md_width=6), create_kpi_card("Overall ROAS", "kpi-avg-roas", "primary", width=3, md_width=6), create_kpi_card("Average CPA (CAC)", "kpi-avg-cpa", "warning", width=3, md_width=6), create_kpi_card("CLV to CAC Ratio", "kpi-clv-cac-ratio", "success", width=3, md_width=6)]),
        dbc.Row([create_kpi_card("Attributed Conversions", "kpi-total-conversions", "dark", width=12)]),
        html.Hr(className="my-4"), dbc.Row([create_graph_card('clv-by-channel-chart', title="Average Customer Lifetime Value by Acquisition Channel", width=12)]), html.Hr(className="my-4"),
        dbc.Row([create_graph_card('roas-by-campaign-chart', width=12)]), dbc.Row([create_graph_card('cpa-by-campaign-chart'), create_graph_card('conversions-by-channel-chart')]),
    ], fluid=True)
def create_profit_layout() -> dbc.Container:
    profit_df = DATA.get('profit_df', pd.DataFrame())
    if profit_df.empty: return dbc.Container(html.H4("Profit calculation requires Sales, Delivery, and Marketing data.", className="text-center mt-5"))
    region_opts, category_opts = profit_df['city'].unique(), profit_df['category'].unique()
    return dbc.Container([
        dbc.Card(dbc.CardBody([
            dbc.Row([
                 dbc.Col([html.Label("Date Range:"), dcc.DatePickerRange(id='profit-date-picker', min_date_allowed=profit_df['date'].min(), max_date_allowed=profit_df['date'].max(), start_date=profit_df['date'].min(), end_date=profit_df['date'].max(), className="d-block")], width=12, lg=4, className="mb-2"),
                 dbc.Col([html.Label("Region (City):"), dcc.Dropdown(id='profit-region-filter', options=create_multi_filter_options(region_opts), value=['All'], multi=True, clearable=False)], width=12, lg=3, className="mb-2"),
                 dbc.Col([html.Label("Category:"), dcc.Dropdown(id='profit-category-filter', options=create_multi_filter_options(category_opts), value=['All'], multi=True, clearable=False)], width=12, lg=3, className="mb-2"),
                 dbc.Col([html.Label("Actions", style={'visibility': 'hidden'}), dbc.ButtonGroup([dbc.Button("Apply", id="profit-apply-btn", color="primary"), dbc.Button("Export PDF", id="profit-export-btn", color="secondary")], className="w-100")], width=12, lg=2, className="mb-2 align-self-end"),
            ], align="bottom"),
        ]), className="mb-4"),
        dbc.Row([create_kpi_card("Total Net Profit", "kpi-total-net-profit", "success"), create_kpi_card("Average Profit Margin", "kpi-avg-profit-margin", "primary"), create_kpi_card("Profit Lost to Returns", "kpi-profit-lost-returns", "danger")]),
        html.Hr(className="my-4"), dbc.Row([create_graph_card('profit-waterfall-chart', title="Profitability Waterfall Analysis", width=12)]), html.Hr(className="my-4"),
        dbc.Row([dbc.Col([html.H4("Key Profit Drivers", className="mb-3"), create_graph_card('profit-by-channel-chart', width=12), create_graph_card('profit-by-category-chart', width=12)], md=6), dbc.Col([html.H4("Actionable Recommendations", className="mb-3"), dbc.Card(dbc.CardBody(id='automated-recommendations-list'), style={"height": "95%"})], md=6)]),
        html.Hr(className="my-4"), dbc.Row([create_graph_card('high-margin-products-chart'), create_graph_card('low-margin-products-chart')]),
    ], fluid=True)
def _create_forecast_tab() -> dbc.Tab:
    return dbc.Tab(label="Demand Forecasting & Promotion Simulation", children=[dbc.Row([dbc.Col(dbc.Card([dbc.CardBody([html.H5("Simulation Controls"), dbc.Label("Forecast Horizon (Days):"), dcc.Slider(id="forecast-slider-days", min=30, max=180, step=30, value=90, marks={30:'30', 90:'90', 180:'180'}, tooltip={"placement": "bottom", "always_visible": True}), html.Hr(), dbc.Label("Simulate Promotion Uplift (% Increase):"), dcc.Slider(id="forecast-slider-promo", min=0, max=50, step=5, value=0, marks={i: f"{i}%" for i in range(0, 51, 10)}, tooltip={"placement": "bottom", "always_visible": True}), dbc.Button("Run Simulation", id="forecast-run-button", color="primary", className="mt-4 w-100")])]), lg=3, md=12, className="mb-4"), dbc.Col([dbc.Row([create_kpi_card("Forecasted Revenue (Baseline)", "pred-kpi-forecast-rev", "primary", width=6, md_width=6), create_kpi_card("Simulated Revenue Lift (Promo)", "pred-kpi-sim-lift", "success", width=6, md_width=6)]), dbc.Row([create_graph_card(graph_id="forecast-simulation-chart", title="Demand Forecast & Promotion Simulation", width=12)])], lg=9, md=12)], className="mt-3")])
def _create_churn_tab() -> dbc.Tab:
    return dbc.Tab(label="Customer Churn & LTV", children=[dcc.Loading(id="loading-churn-content", type="default", children=html.Div(id="churn-tab-content-wrapper", className="mt-3"))])
def create_predictive_layout() -> dbc.Container:
    return dbc.Container([dbc.Row([dbc.Col(html.H4("Predictive Analytics & Simulations"), width=12, className="mb-3")]), dbc.Tabs([_create_forecast_tab(), _create_churn_tab()])], fluid=True)
def create_main_layout() -> html.Div:
    navbar = dbc.Navbar(dbc.Container([html.A(dbc.Row([dbc.Col(dbc.NavbarBrand("Yosri Analytics Hub", className="ms-2"))], align="center", className="g-0"), href="#", style={"textDecoration": "none"}), dbc.NavbarToggler(id="navbar-toggler", n_clicks=0), dbc.Collapse(dbc.Nav([dbc.Button(["Refresh Data ", html.I(className="bi bi-arrow-clockwise")], id="refresh-data-button", color="secondary")], className="ms-auto p-2", navbar=True), id="navbar-collapse", is_open=False, navbar=True)]), color="primary", dark=True, className="mb-4")
    return html.Div([
        dcc.Interval(id='alert-poll-interval', interval=60 * 1000, n_intervals=0), html.Div(id='active-alert-banner-container', style={'padding': '10px'}),
        dcc.Store(id='data-store-trigger'), dcc.Download(id="download-dataframe-csv"), dcc.Download(id="download-dashboard-pdf"),
        dcc.Store(id='store-forecast-model'), dcc.Store(id='store-churn-model'), dcc.Store(id='model-training-signal-store', data=0),
        navbar,
        dbc.Container([dbc.Tabs(id="tabs-controller", active_tab="sales-tab", children=[dbc.Tab(label="Sales", tab_id="sales-tab"), dbc.Tab(label="Logistics", tab_id="delivery-tab"), dbc.Tab(label="Customers", tab_id="customer-tab"), dbc.Tab(label="Marketing", tab_id="marketing-tab"), dbc.Tab(label="Profit Optimization", tab_id="profit-tab"), dbc.Tab(label="Predictive Insights", tab_id="predictive-tab")]), html.Div(id='tab-content', className="mt-4")], fluid=True)
    ])

