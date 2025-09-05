# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# E-commerce Data Analyst App - V19.1 (Code Quality)
#
# This version resolves the FutureWarning from pandas by updating the syntax
# for fillna operations, ensuring future compatibility.
# -----------------------------------------------------------------------------

# --- 1. IMPORT LIBRARIES ---
import pandas
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, dash_table
from dash.exceptions import PreventUpdate
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import os
import re
import joblib

# --- 2. DATABASE CONNECTION ---
def get_database_url():
    """Gets the correct database URL from environment variables or local fallback."""
    DATABASE_URL = os.environ.get('DATABASE_URL')
    if DATABASE_URL:
        DATABASE_URL = re.sub(r"^(postgres|https)://", "postgresql://", DATABASE_URL)
        if "render.com" in DATABASE_URL and "?sslmode=require" not in DATABASE_URL:
            DATABASE_URL += "?sslmode=require"
    else:
        print("DATABASE_URL not found. Falling back to local connection.")
        DB_NAME = 'pharma_db'
        DB_USER = 'mohamedyousri' # Replace with your Mac username
        DB_HOST = 'localhost'
        DATABASE_URL = f"postgresql://{DB_USER}@{DB_HOST}:5432/{DB_NAME}"
    return DATABASE_URL

engine = create_engine(get_database_url())

# --- 3. RESILIENT DATA LOADING ---
def load_data_safely(table_name, engine):
    """Tries to load a table; returns an empty DataFrame on failure."""
    try:
        with engine.connect() as connection:
            query = text(f"SELECT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename  = '{table_name}');")
            if not connection.execute(query).scalar_one():
                print(f"  [WARNING] Table '{table_name}' not found. Skipping.")
                return pandas.DataFrame()
        df = pandas.read_sql(f'SELECT * FROM {table_name}', engine)
        df.columns = [col.lower() for col in df.columns]
        print(f"  [SUCCESS] Loaded {table_name} data.")
        return df
    except Exception as e:
        print(f"  [ERROR] Could not load table '{table_name}'. Error: {e}")
        return pandas.DataFrame()

print("Connecting to database and loading data...")
sales_df = load_data_safely('sales', engine)
delivery_df = load_data_safely('deliveries', engine)
customer_df = load_data_safely('customers', engine)
marketing_campaigns_df = load_data_safely('marketing_campaigns', engine)
marketing_attribution_df = load_data_safely('marketing_attribution', engine)
funnel_df = load_data_safely('sales_funnel', engine)
competitor_df = load_data_safely('competitors', engine)
print("Data loading process finished.")


# --- 4. MASTER DATA PREPARATION & MODEL PREDICTION ---

# Initialize all analysis dataframes to be safe
profit_df = pandas.DataFrame()
customer_analysis_df = pandas.DataFrame()
price_comparison_df = pandas.DataFrame()
campaign_performance_df = pandas.DataFrame()

# Base Data Type Conversion
if not sales_df.empty:
    sales_df['timestamp'] = pandas.to_datetime(sales_df['timestamp'])
    sales_df['date'] = sales_df['timestamp'].dt.date
    sales_df['week'] = sales_df['timestamp'].dt.to_period('W').astype(str)
    sales_df['month'] = sales_df['timestamp'].dt.to_period('M').astype(str)

if not delivery_df.empty:
    delivery_df['orderdate'] = pandas.to_datetime(delivery_df['orderdate'], errors='coerce')
    delivery_df['date'] = delivery_df['orderdate'].dt.date
    delivery_df['actualdeliverydate'] = pandas.to_datetime(delivery_df['actualdeliverydate'], errors='coerce')
    delivery_df['promiseddate'] = pandas.to_datetime(delivery_df['promiseddate'], errors='coerce')
    if 'actualdeliverydate' in delivery_df.columns and 'orderdate' in delivery_df.columns:
        delivery_df['delivery_time_days'] = (delivery_df['actualdeliverydate'] - delivery_df['orderdate']).dt.days
    if 'actualdeliverydate' in delivery_df.columns and 'promiseddate' in delivery_df.columns:
        delivery_df['on_time'] = delivery_df['actualdeliverydate'] <= delivery_df['promiseddate']


# Customer Segmentation
if not sales_df.empty and not customer_df.empty:
    customer_df['joindate'] = pandas.to_datetime(customer_df['joindate'])
    rfm_df = sales_df.groupby('customerid').agg(
        last_purchase_date=('timestamp', 'max'),
        frequency=('orderid', 'nunique'),
        monetary=('netsale', 'sum')
    ).reset_index()
    current_date = datetime.now()
    rfm_df['recency'] = (current_date - rfm_df['last_purchase_date']).dt.days
    customer_analysis_df = pandas.merge(customer_df, rfm_df, on='customerid', how='left')
    def get_status(row):
        join_recency = (current_date - row['joindate']).days
        if join_recency <= 90: return 'New'
        if pandas.isna(row['recency']): return 'Never Purchased'
        if row['recency'] <= 90: return 'Active'
        if 90 < row['recency'] <= 180: return 'Dormant (At-Risk)'
        return 'Churn Risk'
    customer_analysis_df['status'] = customer_analysis_df.apply(get_status, axis=1)

# Marketing Performance
if not sales_df.empty and not marketing_campaigns_df.empty and not marketing_attribution_df.empty:
    order_revenue = sales_df.groupby('orderid')['netsale'].sum().reset_index()
    attributed_revenue = pandas.merge(marketing_attribution_df, order_revenue, on='orderid')
    campaign_revenue = attributed_revenue.groupby('campaignid')['netsale'].sum().reset_index()
    campaign_conversions = attributed_revenue.groupby('campaignid')['orderid'].nunique().reset_index().rename(columns={'orderid': 'conversions'})
    campaign_performance_df = pandas.merge(marketing_campaigns_df, campaign_revenue, on='campaignid', how='left')
    campaign_performance_df = pandas.merge(campaign_performance_df, campaign_conversions, on='campaignid', how='left')
    campaign_performance_df['conversions'] = campaign_performance_df['conversions'].fillna(0)
    campaign_performance_df['netsale'] = campaign_performance_df['netsale'].fillna(0)
    campaign_performance_df['roas'] = campaign_performance_df.apply(lambda r: r['netsale']/r['totalcost'] if r['totalcost']>0 else 0, axis=1)
    campaign_performance_df['cpa'] = campaign_performance_df.apply(lambda r: r['totalcost']/r['conversions'] if r['conversions']>0 else 0, axis=1)
    campaign_performance_df['ctr'] = campaign_performance_df.apply(lambda r: (r['clicks']/r['impressions'])*100 if r['impressions']>0 else 0, axis=1)

# Profitability Analysis
if not all(df.empty for df in [sales_df, delivery_df, marketing_campaigns_df, marketing_attribution_df]):
    
    # PASTE THE DEBUG PRINT STATEMENT HERE
    print("Columns in delivery_df:", delivery_df.columns.tolist())
    
    profit_df = pandas.merge(sales_df, delivery_df[['orderid', 'deliverycost']], on='orderid', how='left')
    profit_df = pandas.merge(profit_df, marketing_attribution_df, on='orderid', how='left')
    profit_df = pandas.merge(sales_df, delivery_df[['orderid', 'deliverycost']], on='orderid', how='left')
    profit_df = pandas.merge(profit_df, marketing_attribution_df, on='orderid', how='left')
    if not marketing_attribution_df.empty:
        campaign_costs = marketing_campaigns_df.copy()
        order_counts = marketing_attribution_df['campaignid'].value_counts().reset_index()
        order_counts.columns = ['campaignid', 'orders_in_campaign']
        campaign_costs = pandas.merge(campaign_costs, order_counts, on='campaignid', how='left')
        campaign_costs['marketing_cost_per_order'] = campaign_costs.apply(lambda r: r['totalcost']/r['orders_in_campaign'] if pandas.notna(r['orders_in_campaign']) and r['orders_in_campaign']>0 else 0, axis=1)
        profit_df = pandas.merge(profit_df, campaign_costs[['campaignid', 'marketing_cost_per_order']], on='campaignid', how='left')
    else: profit_df['marketing_cost_per_order'] = 0
    # --- FIX: Replaced inplace=True to prevent FutureWarning ---
    profit_df['deliverycost'] = profit_df['deliverycost'].fillna(delivery_df['deliverycost'].mean())
    profit_df['marketing_cost_per_order'] = profit_df['marketing_cost_per_order'].fillna(0)
    profit_df['total_cost'] = profit_df['costofgoodssold'] + profit_df['deliverycost'] + profit_df['marketing_cost_per_order']
    profit_df['net_profit'] = profit_df['netsale'] - profit_df['total_cost']
    profit_df['profit_margin'] = profit_df.apply(lambda r: (r['net_profit']/r['netsale'])*100 if r['netsale']>0 else 0, axis=1)

# Competitor Price Comparison
if not competitor_df.empty and not sales_df.empty:
    competitor_df['date'] = pandas.to_datetime(competitor_df['date'])
    our_prices = sales_df.groupby('productname')['netsale'].mean().reset_index().rename(columns={'netsale': 'our_price'})
    competitor_avg_prices = competitor_df.groupby('productname')['price'].mean().reset_index().rename(columns={'price': 'avg_competitor_price'})
    price_comparison_df = pandas.merge(our_prices, competitor_avg_prices, on='productname', how='inner')
    price_comparison_df['price_difference'] = price_comparison_df['our_price'] - price_comparison_df['avg_competitor_price']


# Load the trained model and scaler
churn_model = None
scaler = None
predictions_df = pandas.DataFrame()
try:
    churn_model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("  [SUCCESS] Churn prediction model loaded.")
    
    if not sales_df.empty and not customer_df.empty:
        snapshot_date = sales_df['timestamp'].max() + timedelta(days=1)
        
        rfm_predict = sales_df.groupby('customerid').agg(
            timestamp=('timestamp', 'max'),
            orderid=('orderid', 'nunique'),
            netsale=('netsale', 'sum')
        ).rename(columns={'timestamp': 'Recency', 'orderid': 'Frequency', 'netsale': 'MonetaryValue'})
        rfm_predict['Recency'] = (snapshot_date - rfm_predict['Recency']).dt.days

        customer_features_predict = sales_df.groupby('customerid').agg(avg_basket_value=('netsale', 'mean'), total_quantity=('quantity', 'sum')).reset_index()
        features_df_predict = rfm_predict.merge(customer_features_predict, on='customerid')
        features_df_predict = features_df_predict.merge(customer_df[['customerid', 'joindate', 'segment', 'city']], on='customerid')
        features_df_predict['tenure_days'] = (snapshot_date - features_df_predict['joindate']).dt.days
        
        model_features = pandas.get_dummies(features_df_predict.drop(['customerid', 'joindate'], axis=1), columns=['segment'], drop_first=True)
        model_columns = ['Recency', 'Frequency', 'MonetaryValue', 'avg_basket_value', 'total_quantity', 'tenure_days', 'segment_Silver', 'segment_Gold']
        for col in model_columns:
            if col not in model_features.columns:
                model_features[col] = 0
        model_features = model_features[model_columns]
        
        features_scaled = scaler.transform(model_features)
        churn_probabilities = churn_model.predict_proba(features_scaled)[:, 1]
        
        predictions_df = features_df_predict[['customerid', 'city', 'segment', 'Recency', 'Frequency', 'MonetaryValue']].copy()
        predictions_df['churn_probability'] = churn_probabilities
        predictions_df = predictions_df.sort_values('churn_probability', ascending=False)

except FileNotFoundError:
    print("  [WARNING] 'churn_model.pkl' or 'scaler.pkl' not found. Predictive tab will be disabled.")
except Exception as e:
    print(f"  [ERROR] Failed to load model or make predictions. Error: {e}")

# --- 5. INITIALIZE THE APP ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE], suppress_callback_exceptions=True)
server = app.server

# --- 6. DEFINE LAYOUTS ---
def create_placeholder_figure(message):
    return {"layout": {"xaxis": {"visible": False}, "yaxis": {"visible": False}, "annotations": [{"text": message, "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16}}]}}

def create_sales_layout():
    if sales_df.empty: return html.Div("Sales Data Not Available", className="text-center mt-5")
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
    if delivery_df.empty: return html.Div("Delivery Data Not Available", className="text-center mt-5")
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
    if customer_analysis_df.empty: return html.Div("Customer or Sales Data Not Available", className="text-center mt-5")
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
    if competitor_df.empty or price_comparison_df.empty: return html.Div("Competitor or Sales Data Not Available", className="text-center mt-5")
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
    if campaign_performance_df.empty: return html.Div("Marketing Campaign Data Not Available", className="text-center mt-5")
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
    if profit_df.empty: return html.Div("Profit calculation requires Sales, Delivery, and Marketing data.", className="text-center mt-5")
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
    if predictions_df.empty:
        return html.Div([
            html.H4("Churn Prediction Model Not Available"),
            html.P("Please run 'model_trainer.py' to enable this feature.")
        ], className="text-center mt-5")
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

app.layout = dbc.Container([
    dcc.Download(id="download-dataframe-csv"),
    html.H1("Yosri Analytics Hub - v19.1", className='text-center text-primary mb-4'),
    dbc.Tabs(id="tabs-controller", active_tab="predictive-tab", children=[
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

# --- 7. DEFINE CALLBACKS ---
@app.callback(Output('tab-content', 'children'), Input('tabs-controller', 'active_tab'))
def render_tab_content(active_tab):
    layouts = {
        "sales-tab": create_sales_layout, "delivery-tab": create_delivery_layout,
        "customer-tab": create_customer_layout, "competitor-tab": create_competitor_layout,
        "marketing-tab": create_marketing_layout, "profit-tab": create_profit_layout,
        "predictive-tab": create_predictive_layout
    }
    return layouts.get(active_tab, lambda: html.P("This tab doesn't exist."))()

@app.callback(
    [Output('kpi-total-revenue', 'children'), Output('kpi-gross-margin', 'children'), Output('kpi-net-profit', 'children'),
     Output('kpi-total-orders', 'children'), Output('kpi-aov', 'children'), Output('kpi-return-rate', 'children'),
     Output('sales-funnel-chart', 'figure'), Output('sales-over-time-chart', 'figure'),
     Output('sales-by-category-chart', 'figure'), Output('top-products-chart', 'figure'),
     Output('sales-by-channel-chart', 'figure'), Output('sales-by-city-chart', 'figure'),
     Output('sales-by-branch-chart', 'figure')],
    [Input('channel-filter-dropdown', 'value'), Input('sales-date-picker', 'start_date'),
     Input('sales-date-picker', 'end_date'), Input('time-agg-selector', 'value')]
)
def update_sales_dashboard(selected_channel, start_date, end_date, time_agg):
    if sales_df.empty: raise PreventUpdate
    start_date_obj, end_date_obj = pandas.to_datetime(start_date).date(), pandas.to_datetime(end_date).date()
    date_mask = (sales_df['date'] >= start_date_obj) & (sales_df['date'] <= end_date_obj)
    channel_mask = (sales_df['channel'] == selected_channel) if selected_channel != 'All' else True
    filtered_sales = sales_df.loc[date_mask & channel_mask]
    if filtered_sales.empty:
        empty_card = dbc.CardBody([html.H4("No Data"), html.P("-", className="fs-3")]); placeholder_fig = create_placeholder_figure("No data for period")
        return [empty_card]*6 + [placeholder_fig]*7
    total_revenue, total_cogs = filtered_sales['netsale'].sum(), filtered_sales['costofgoodssold'].sum()
    net_profit, total_orders = total_revenue - total_cogs, filtered_sales['orderid'].nunique()
    gross_margin, aov = (net_profit / total_revenue * 100) if total_revenue > 0 else 0, total_revenue / total_orders if total_orders > 0 else 0
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
    time_grouped = filtered_sales.groupby(time_agg)['netsale'].sum().reset_index(); sales_over_time_fig = px.line(time_grouped, x=time_agg, y='netsale', title=f'Net Sales Trend ({time_agg.capitalize()})')
    category_sales = filtered_sales.groupby('category')['netsale'].sum().reset_index(); sales_by_cat_fig = px.pie(category_sales, names='category', values='netsale', title='Sales by Category', hole=0.3)
    product_sales = filtered_sales.groupby('productname')['netsale'].sum().nlargest(10).reset_index(); top_prod_fig = px.bar(product_sales, x='netsale', y='productname', orientation='h', title='Top 10 Products').update_layout(yaxis={'categoryorder':'total ascending'})
    channel_sales = filtered_sales.groupby('channel')['netsale'].sum().reset_index(); sales_by_channel_fig = px.pie(channel_sales, names='channel', values='netsale', title='Sales by Channel', hole=0.3)
    city_sales = filtered_sales.groupby('city')['netsale'].sum().nlargest(10).reset_index(); sales_by_city_fig = px.bar(city_sales, x='netsale', y='city', orientation='h', title='Top 10 Cities').update_layout(yaxis={'categoryorder':'total ascending'})
    branch_sales = filtered_sales.groupby('locationid')['netsale'].sum().nlargest(10).reset_index(); sales_by_branch_fig = px.bar(branch_sales, x='netsale', y='locationid', orientation='h', title='Top 10 Pharmacy Branches by Sales').update_layout(yaxis={'categoryorder':'total ascending'})
    return kpi_revenue_card, kpi_margin_card, kpi_profit_card, kpi_orders_card, kpi_aov_card, kpi_return_card, funnel_fig, sales_over_time_fig, sales_by_cat_fig, top_prod_fig, sales_by_channel_fig, sales_by_city_fig, sales_by_branch_fig

@app.callback(
    [Output('kpi-on-time-delivery', 'children'), Output('kpi-failed-delivery', 'children'), Output('kpi-avg-delivery-time', 'children'), Output('kpi-avg-delivery-cost', 'children'),
     Output('delivery-pipeline-chart', 'figure'), Output('avg-time-by-city-chart', 'figure'), Output('partner-performance-chart', 'figure')],
    [Input('delivery-partner-filter', 'value'), Input('delivery-date-picker', 'start_date'), Input('delivery-date-picker', 'end_date')]
)
def update_delivery_dashboard(selected_partner, start_date, end_date):
    if delivery_df.empty: raise PreventUpdate
    start_date_obj, end_date_obj = pandas.to_datetime(start_date).date(), pandas.to_datetime(end_date).date()
    date_mask = (delivery_df['date'] >= start_date_obj) & (delivery_df['date'] <= end_date_obj)
    partner_mask = (delivery_df['deliverypartner'] == selected_partner) if selected_partner != 'All' else True
    filtered_df = delivery_df.loc[date_mask & partner_mask].copy()
    if filtered_df.empty:
        empty_card = dbc.CardBody([html.H4("No Data"), html.P("-", className="fs-3")]); placeholder_fig = create_placeholder_figure("No data for this period")
        return empty_card, empty_card, empty_card, empty_card, placeholder_fig, placeholder_fig, placeholder_fig
    total_deliveries = len(filtered_df)
    on_time_rate = (filtered_df['on_time'].sum() / total_deliveries * 100) if total_deliveries > 0 else 0
    failed_rate = ((filtered_df['status'] == 'Failed').sum() / total_deliveries * 100) if total_deliveries > 0 else 0
    avg_delivery_time, avg_delivery_cost = filtered_df['delivery_time_days'].mean(), filtered_df['deliverycost'].mean()
    kpi_on_time_card = dbc.CardBody([html.H4("On-Time Rate"), html.P(f"{on_time_rate:.2f}%", className="fs-3")])
    kpi_failed_card = dbc.CardBody([html.H4("Failed Delivery Rate"), html.P(f"{failed_rate:.2f}%", className="fs-3")])
    kpi_avg_time_card = dbc.CardBody([html.H4("Avg. Delivery Time"), html.P(f"{avg_delivery_time:.2f} Days", className="fs-3")])
    kpi_avg_cost_card = dbc.CardBody([html.H4("Avg. Cost per Delivery"), html.P(f"{avg_delivery_cost:,.2f} SAR", className="fs-3")])
    status_order = ['Pending', 'Shipped', 'Delivered', 'Failed']; pipeline_counts = filtered_df['status'].value_counts().reindex(status_order).fillna(0)
    pipeline_fig = px.bar(pipeline_counts, x=pipeline_counts.index, y=pipeline_counts.values, title='Live Delivery Pipeline', labels={'x': 'Status', 'y': 'Number of Orders'})
    time_by_city = filtered_df.groupby('city')['delivery_time_days'].mean().reset_index(); time_by_city_fig = px.bar(time_by_city, x='city', y='delivery_time_days', title='Average Delivery Time by City', labels={'delivery_time_days': 'Average Days'})
    partner_perf = filtered_df.groupby('deliverypartner')['on_time'].mean().reset_index(); partner_perf['on_time'] *= 100
    partner_perf_fig = px.bar(partner_perf.sort_values('on_time'), x='on_time', y='deliverypartner', orientation='h', title='On-Time Rate by Partner')
    return kpi_on_time_card, kpi_failed_card, kpi_avg_time_card, kpi_avg_cost_card, pipeline_fig, time_by_city_fig, partner_perf_fig

@app.callback(
    [Output('kpi-total-customers', 'children'), Output('kpi-active-customers', 'children'),
     Output('kpi-dormant-customers', 'children'), Output('kpi-churn-risk', 'children'),
     Output('customer-status-dist-chart', 'figure'), Output('customer-data-table', 'data'),
     Output('customer-data-table', 'columns')],
    [Input('tabs-controller', 'active_tab'), Input('customer-list-selector', 'value')]
)
def update_customer_dashboard(active_tab, selected_list):
    if active_tab != 'customer-tab' or customer_analysis_df.empty: raise PreventUpdate
    status_counts = customer_analysis_df['status'].value_counts()
    total_cust, active_cust = len(customer_analysis_df), status_counts.get('Active', 0)
    dormant_cust, churn_risk_cust = status_counts.get('Dormant (At-Risk)', 0), status_counts.get('Churn Risk', 0)
    kpi_total_card = dbc.CardBody([html.H4("Total Customers"), html.P(f"{total_cust:,}", className="fs-3")])
    kpi_active_card = dbc.CardBody([html.H4("Active Customers"), html.P(f"{active_cust:,}", className="fs-3")])
    kpi_dormant_card = dbc.CardBody([html.H4("Dormant Customers"), html.P(f"{dormant_cust:,}", className="fs-3")])
    kpi_churn_card = dbc.CardBody([html.H4("High Churn Risk"), html.P(f"{churn_risk_cust:,}", className="fs-3")])
    status_dist_fig = px.pie(status_counts, names=status_counts.index, values=status_counts.values, title='Customer Status Distribution', hole=0.3)
    table_df = pandas.DataFrame()
    if selected_list == 'top_value':
        table_df = customer_analysis_df.sort_values('monetary', ascending=False).head(20)[['customerid', 'city', 'segment', 'monetary', 'frequency', 'recency']]
    elif selected_list == 'churn_risk':
        table_df = customer_analysis_df[customer_analysis_df['status'] == 'Churn Risk'][['customerid', 'city', 'segment', 'recency', 'last_purchase_date']]
    elif selected_list == 'new':
        table_df = customer_analysis_df[customer_analysis_df['status'] == 'New'][['customerid', 'city', 'segment', 'joindate']]
    columns = [{"name": i, "id": i} for i in table_df.columns]; data = table_df.to_dict('records')
    return kpi_total_card, kpi_active_card, kpi_dormant_card, kpi_churn_card, status_dist_fig, data, columns

@app.callback(
    Output("download-dataframe-csv", "data"),
    [Input("export-csv-button", "n_clicks"), State("customer-list-selector", "value")],
    prevent_initial_call=True,
)
def export_customer_data(n_clicks, selected_list):
    if n_clicks is None or customer_analysis_df.empty: raise PreventUpdate
    export_df = pandas.DataFrame()
    if selected_list == 'top_value': export_df = customer_analysis_df.sort_values('monetary', ascending=False)
    elif selected_list == 'churn_risk': export_df = customer_analysis_df[customer_analysis_df['status'] == 'Churn Risk']
    elif selected_list == 'new': export_df = customer_analysis_df[customer_analysis_df['status'] == 'New']
    filename = f"{selected_list}_customers_{datetime.now().strftime('%Y-%m-%d')}.csv"
    return dcc.send_data_frame(export_df.to_csv, filename, index=False)

@app.callback(
    [Output('kpi-price-advantage', 'children'), Output('kpi-price-disadvantage', 'children'), Output('kpi-promo-frequency', 'children'),
     Output('price-comparison-scatter-chart', 'figure'), Output('promo-analysis-chart', 'figure'), Output('assortment-overlap-chart', 'figure')],
    [Input('tabs-controller', 'active_tab')]
)
def update_competitor_dashboard(active_tab):
    if active_tab != 'competitor-tab' or price_comparison_df.empty: raise PreventUpdate
    products_cheaper, products_pricier = price_comparison_df[price_comparison_df['price_difference'] < 0].shape[0], price_comparison_df[price_comparison_df['price_difference'] > 0].shape[0]
    promo_rate = (competitor_df['onpromotion'].sum() / len(competitor_df) * 100) if not competitor_df.empty else 0
    kpi_advantage_card = dbc.CardBody([html.H4("Products We Undercut"), html.P(f"{products_cheaper}", className="fs-3")])
    kpi_disadvantage_card = dbc.CardBody([html.H4("Products More Expensive"), html.P(f"{products_pricier}", className="fs-3")])
    kpi_promo_card = dbc.CardBody([html.H4("Avg. Competitor Promo Rate"), html.P(f"{promo_rate:.2f}%", className="fs-3")])
    price_comp_fig = px.scatter(price_comparison_df, x='our_price', y='avg_competitor_price', hover_name='productname', text='productname', size='price_difference', title='Our Price vs. Average Competitor Price')
    price_comp_fig.add_shape(type='line', x0=0, y0=0, x1=price_comparison_df['our_price'].max(), y1=price_comparison_df['our_price'].max(), line=dict(color='red', dash='dash'))
    promo_freq = competitor_df.groupby('competitor')['onpromotion'].mean().reset_index(); promo_freq['onpromotion'] *= 100
    promo_fig = px.bar(promo_freq, x='competitor', y='onpromotion', title='Promotion Frequency by Competitor')
    our_products = set(sales_df['productname'].unique())
    nahdi_products = set(competitor_df[competitor_df['competitor'] == 'Nahdi']['productname'].unique()) if not competitor_df.empty else set()
    dawaa_products = set(competitor_df[competitor_df['competitor'] == 'Al-Dawaa']['productname'].unique()) if not competitor_df.empty else set()
    venn_data = pandas.DataFrame([
        {'sets': ['Ours Only'], 'size': len(our_products - nahdi_products - dawaa_products)},
        {'sets': ['Nahdi Only'], 'size': len(nahdi_products - our_products - dawaa_products)},
        {'sets': ['Al-Dawaa Only'], 'size': len(dawaa_products - our_products - nahdi_products)},
        {'sets': ['Ours & Nahdi'], 'size': len(our_products & nahdi_products - dawaa_products)},
        {'sets': ['Ours & Al-Dawaa'], 'size': len(our_products & dawaa_products - nahdi_products)},
        {'sets': ['Nahdi & Al-Dawaa'], 'size': len(nahdi_products & dawaa_products - our_products)},
        {'sets': ['All Three'], 'size': len(our_products & nahdi_products & dawaa_products)},
    ]); assortment_fig = px.bar(venn_data, x='size', y='sets', orientation='h', title='Product Assortment Overlap')
    return kpi_advantage_card, kpi_disadvantage_card, kpi_promo_card, price_comp_fig, promo_fig, assortment_fig

@app.callback(
    [Output('kpi-total-ad-spend', 'children'), Output('kpi-avg-roas', 'children'),
     Output('kpi-avg-cpa', 'children'), Output('kpi-total-conversions', 'children'),
     Output('roas-by-campaign-chart', 'figure'), Output('cpa-by-campaign-chart', 'figure'),
     Output('conversions-by-channel-chart', 'figure')],
    [Input('tabs-controller', 'active_tab')]
)
def update_marketing_dashboard(active_tab):
    if active_tab != 'marketing-tab' or campaign_performance_df.empty: raise PreventUpdate
    total_spend = campaign_performance_df['totalcost'].sum()
    total_revenue = campaign_performance_df['netsale'].sum()
    avg_roas = total_revenue / total_spend if total_spend > 0 else 0
    total_conversions = campaign_performance_df['conversions'].sum()
    avg_cpa = total_spend / total_conversions if total_conversions > 0 else 0
    kpi_spend_card = dbc.CardBody([html.H4("Total Ad Spend"), html.P(f"{total_spend:,.2f} SAR", className="fs-3")])
    kpi_roas_card = dbc.CardBody([html.H4("Overall ROAS"), html.P(f"{avg_roas:.2f}x", className="fs-3")])
    kpi_cpa_card = dbc.CardBody([html.H4("Average CPA"), html.P(f"{avg_cpa:,.2f} SAR", className="fs-3")])
    kpi_conv_card = dbc.CardBody([html.H4("Attributed Conversions"), html.P(f"{total_conversions:,.0f}", className="fs-3")])
    roas_fig = px.bar(campaign_performance_df, x='campaignname', y='roas', color='channel', title='Return on Ad Spend (ROAS) by Campaign')
    cpa_fig = px.bar(campaign_performance_df, x='campaignname', y='cpa', color='channel', title='Cost Per Acquisition (CPA) by Campaign')
    conv_by_channel = campaign_performance_df.groupby('channel')['conversions'].sum().reset_index()
    conv_channel_fig = px.pie(conv_by_channel, names='channel', values='conversions', title='Attributed Conversions by Channel', hole=0.3)
    return kpi_spend_card, kpi_roas_card, kpi_cpa_card, kpi_conv_card, roas_fig, cpa_fig, conv_channel_fig

@app.callback(
    [Output('kpi-total-net-profit', 'children'), Output('kpi-avg-profit-margin', 'children'),
     Output('kpi-profit-lost-returns', 'children'), Output('profit-by-channel-chart', 'figure'),
     Output('profit-by-category-chart', 'figure'), Output('high-margin-products-chart', 'figure'),
     Output('low-margin-products-chart', 'figure'), Output('automated-recommendations-list', 'children')],
    [Input('tabs-controller', 'active_tab')]
)
def update_profit_dashboard(active_tab):
    if active_tab != 'profit-tab' or profit_df.empty: raise PreventUpdate
    total_net_profit = profit_df['net_profit'].sum()
    avg_profit_margin = profit_df['profit_margin'].mean()
    returned_orders_df = profit_df[profit_df['orderstatus'] == 'Returned']
    profit_lost_to_returns = returned_orders_df['total_cost'].sum() + returned_orders_df['netsale'].sum()
    kpi_profit_card = dbc.CardBody([html.H4("Total Net Profit"), html.P(f"{total_net_profit:,.2f} SAR", className="fs-3")])
    kpi_margin_card = dbc.CardBody([html.H4("Average Profit Margin"), html.P(f"{avg_profit_margin:.2f}%", className="fs-3")])
    kpi_returns_card = dbc.CardBody([html.H4("Profit Lost to Returns"), html.P(f"{profit_lost_to_returns:,.2f} SAR", className="fs-3")])
    profit_by_channel = profit_df.groupby('channel')['net_profit'].sum().reset_index()
    profit_by_channel_fig = px.bar(profit_by_channel, x='channel', y='net_profit', title='Profit Contribution by Channel', color='channel')
    profit_by_category = profit_df.groupby('category')['net_profit'].sum().reset_index()
    profit_by_cat_fig = px.bar(profit_by_category, x='category', y='net_profit', title='Net Profit by Product Category')
    product_profit = profit_df.groupby('productname')['profit_margin'].mean().reset_index()
    high_margin_prods = product_profit.nlargest(10, 'profit_margin')
    low_margin_prods = product_profit.nsmallest(10, 'profit_margin')
    high_margin_fig = px.bar(high_margin_prods, x='profit_margin', y='productname', orientation='h', title='Top 10 Most Profitable Products').update_layout(yaxis={'categoryorder':'total ascending'})
    low_margin_fig = px.bar(low_margin_prods, x='profit_margin', y='productname', orientation='h', title='Top 10 Least Profitable Products').update_layout(yaxis={'categoryorder':'total descending'})
    recommendations = []
    if not pandas.isna(total_net_profit) and total_net_profit > 0 and profit_lost_to_returns > (total_net_profit * 0.1): 
        recommendations.append(html.Li("High profit loss from returns. Investigate causes."))
    unprofitable_channel = profit_by_channel[profit_by_channel['net_profit'] < 0]
    if not unprofitable_channel.empty:
        channel_name = unprofitable_channel.iloc[0]['channel']
        recommendations.append(html.Li(f"Channel '{channel_name}' is unprofitable. Review marketing strategy."))
    if not high_margin_prods.empty:
        top_performer = high_margin_prods.iloc[0]
        recommendations.append(html.Li(f"'{top_performer['productname']}' has a {top_performer['profit_margin']:.2f}% margin. Consider promoting it."))
    recommendation_list = html.Ul(recommendations) if recommendations else html.P("No critical issues detected.")
    return kpi_profit_card, kpi_margin_card, kpi_returns_card, profit_by_channel_fig, profit_by_cat_fig, high_margin_fig, low_margin_fig, recommendation_list

@app.callback(
    Output("download-dataframe-csv", "data", allow_duplicate=True),
    [Input("export-csv-button", "n_clicks"), State("customer-list-selector", "value")],
    prevent_initial_call=True,
)
def export_actionable_list(n_clicks, selected_list):
    if n_clicks is None or customer_analysis_df.empty: raise PreventUpdate
    export_df = pandas.DataFrame()
    if selected_list == 'top_value': export_df = customer_analysis_df.sort_values('monetary', ascending=False)
    elif selected_list == 'churn_risk': export_df = customer_analysis_df[customer_analysis_df['status'] == 'Churn Risk']
    elif selected_list == 'new': export_df = customer_analysis_df[customer_analysis_df['status'] == 'New']
    filename = f"{selected_list}_customers_{datetime.now().strftime('%Y-%m-%d')}.csv"
    return dcc.send_data_frame(export_df.to_csv, filename, index=False)

@app.callback(
    Output("download-dataframe-csv", "data", allow_duplicate=True),
    Input("export-churn-button", "n_clicks"),
    prevent_initial_call=True
)
def export_churn_list(n_clicks):
    if n_clicks is None or predictions_df.empty:
        raise PreventUpdate
    
    churn_list_df = predictions_df[predictions_df['churn_probability'] > 0.7]
    filename = f"high_churn_risk_customers_{datetime.now().strftime('%Y-%m-%d')}.csv"
    return dcc.send_data_frame(churn_list_df.to_csv, filename, index=False)

# --- 8. RUN THE APP ---
if __name__ == '__main__':
    app.run(debug=True)

