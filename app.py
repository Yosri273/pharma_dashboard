# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# E-commerce Data Analyst App - V12.1 (Final Deployment Fix)
#
# This version fixes a deployment error by robustly handling different
# database URL formats from cloud providers.
# -----------------------------------------------------------------------------

# --- 1. IMPORT LIBRARIES ---
import pandas
import plotly.express as px
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
from dash.exceptions import PreventUpdate
from datetime import datetime
from sqlalchemy import create_engine
import os
import re # NEW: To perform advanced string replacement

# --- 2. DATABASE CONNECTION ---
# This logic now robustly handles various cloud provider URL formats
DATABASE_URL = os.environ.get('DATABASE_URL')

if DATABASE_URL:
    # --- FIX IS HERE ---
    # Cloud providers might use 'postgres://' or even 'https://'.
    # SQLAlchemy requires 'postgresql://'. We use a regular expression
    # to replace the part before the '@' sign with the correct dialect.
    DATABASE_URL = re.sub(r"^(postgres|https)://", "postgresql://", DATABASE_URL)
else:
    # For local development (Docker)
    DB_NAME = 'pharma_db'
    DB_USER = 'mohamedyousri'
    DB_HOST = 'host.docker.internal'
    DATABASE_URL = f"postgresql://{DB_USER}@{DB_HOST}:5432/{DB_NAME}"

engine = create_engine(DATABASE_URL)

# ... (The rest of the file is unchanged) ...
# --- 3. LOAD AND PREPARE DATA ---
try:
    print("Connecting to database and loading data...")
    sales_df = pandas.read_sql('SELECT * FROM sales', engine)
    sales_df.columns = [col.lower() for col in sales_df.columns]
    sales_df['timestamp'] = pandas.to_datetime(sales_df['timestamp'])
    sales_df['date'] = sales_df['timestamp'].dt.date
    sales_df['weekday'] = sales_df['timestamp'].dt.day_name()
    print("Sales data loaded.")

    delivery_df = pandas.read_sql('SELECT * FROM deliveries', engine)
    delivery_df.columns = [col.lower() for col in delivery_df.columns]
    delivery_df['orderdate'] = pandas.to_datetime(delivery_df['orderdate'])
    delivery_df['promiseddate'] = pandas.to_datetime(delivery_df['promiseddate'])
    delivery_df['actualdeliverydate'] = pandas.to_datetime(delivery_df['actualdeliverydate'])
    delivery_df['delivery_time_days'] = (delivery_df['actualdeliverydate'] - delivery_df['orderdate']).dt.days
    delivery_df['on_time'] = delivery_df['actualdeliverydate'] <= delivery_df['promiseddate']
    delivery_df['date'] = delivery_df['orderdate'].dt.date
    print("Delivery data loaded.")
    
    customer_df = pandas.read_sql('SELECT * FROM customers', engine)
    customer_df.columns = [col.lower() for col in customer_df.columns]
    customer_df['joindate'] = pandas.to_datetime(customer_df['joindate'])
    customer_df['join_month'] = customer_df['joindate'].dt.to_period('M').astype(str)
    print("Customer data loaded.")

    competitor_df = pandas.read_sql('SELECT * FROM competitors', engine)
    competitor_df.columns = [col.lower() for col in competitor_df.columns]
    competitor_df['date'] = pandas.to_datetime(competitor_df['date'])
    print("Competitor data loaded.")
    
    rfm_df = sales_df.groupby('customerid').agg(
        last_purchase_date=('timestamp', 'max'),
        frequency=('orderid', 'nunique'),
        monetary=('netsale', 'sum')
    ).reset_index()
    
    current_date = datetime.now()
    rfm_df['recency'] = (current_date - rfm_df['last_purchase_date']).dt.days
    rfm_df = rfm_df.merge(customer_df[['customerid', 'segment']], on='customerid', how='left')

except Exception as e:
    print("\n--- DATABASE CONNECTION ERROR ---")
    print("Could not load data. The database might be unavailable or empty.")
    print(f"Error details: {e}")
    sales_df = pandas.DataFrame()
    delivery_df = pandas.DataFrame()
    customer_df = pandas.DataFrame()
    competitor_df = pandas.DataFrame()
    rfm_df = pandas.DataFrame()


# --- 4. INITIALIZE THE APP ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE], suppress_callback_exceptions=True)
server = app.server

# --- 5. DEFINE LAYOUTS AS FUNCTIONS ---
def create_sales_layout():
    if sales_df.empty:
        return dbc.Alert("Sales data could not be loaded. Please run the data loader.", color="danger")
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.Label("Filter by Location ID:"), width=6),
            dbc.Col(html.Label("Filter by Date Range:"), width=6),
        ]),
        dbc.Row([
            dbc.Col(dcc.Dropdown(
                id='location-filter-dropdown',
                options=[{'label': 'All', 'value': 'All'}] + [{'label': loc, 'value': loc} for loc in sorted(sales_df['locationid'].unique())],
                value='All', clearable=False
            ), width=6, className="mb-4"),
            dbc.Col(dcc.DatePickerRange(
                id='sales-date-picker',
                min_date_allowed=sales_df['date'].min(),
                max_date_allowed=sales_df['date'].max(),
                start_date=sales_df['date'].min(),
                end_date=sales_df['date'].max(),
                display_format='YYYY-MM-DD'
            ), width=6, className="mb-4"),
        ]),
        dbc.Row([
            dbc.Col(dbc.Card(id='kpi-net-sales', color="primary", inverse=True, className="mb-4"), width=4),
            dbc.Col(dbc.Card(id='kpi-total-discount', color="warning", inverse=True, className="mb-4"), width=4),
            dbc.Col(dbc.Card(id='kpi-total-orders', color="info", inverse=True, className="mb-4"), width=4),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='sales-over-time-chart'), width=12),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='sales-by-category-chart'), width=6),
            dbc.Col(dcc.Graph(id='top-products-chart'), width=6),
        ]),
    ], fluid=True)

def create_delivery_layout():
    if delivery_df.empty:
        return dbc.Alert("Delivery data could not be loaded. Please run the data loader.", color="danger")
    return dbc.Container([
        dbc.Row([dbc.Col(html.Label("Filter by Order Date Range:"), width=12)]),
        dbc.Row([dbc.Col(dcc.DatePickerRange(
            id='delivery-date-picker',
            min_date_allowed=delivery_df['date'].min(),
            max_date_allowed=delivery_df['date'].max(),
            start_date=delivery_df['date'].min(),
            end_date=delivery_df['date'].max(),
            display_format='YYYY-MM-DD'
        ), width=12, className="mb-4")]),
        dbc.Row([
            dbc.Col(dbc.Card(id='kpi-on-time-delivery', color="success", inverse=True, className="mb-4"), width=4),
            dbc.Col(dbc.Card(id='kpi-avg-delivery-time', color="warning", inverse=True, className="mb-4"), width=4),
            dbc.Col(dbc.Card(id='kpi-total-deliveries', color="info", inverse=True, className="mb-4"), width=4),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='delivery-status-chart'), width=6),
            dbc.Col(dcc.Graph(id='partner-performance-chart'), width=6),
        ]),
    ], fluid=True)

def create_customer_layout():
    if customer_df.empty or rfm_df.empty:
        return dbc.Alert("Customer data could not be loaded. Please run the data loader.", color="danger")
    return dbc.Container([
        dbc.Row([
            dbc.Col(dbc.Card(id='kpi-total-customers', color="primary", inverse=True, className="mb-4"), width=4),
            dbc.Col(dbc.Card(id='kpi-avg-clv', color="info", inverse=True, className="mb-4"), width=4),
            dbc.Col(dbc.Card(id='kpi-new-customers', color="success", inverse=True, className="mb-4"), width=4),
        ]),
        dbc.Row([
            dbc.Col(dbc.Button("Export Customer Segments to CSV", id="export-csv-button", color="primary", className="mb-4"), width={"size": 4, "offset": 8}, style={'textAlign': 'right'})
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='rfm-chart'), width=12),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='sales-by-segment-chart'), width=6),
            dbc.Col(dcc.Graph(id='customer-segment-dist-chart'), width=6),
        ]),
    ], fluid=True)

def create_competitor_layout():
    if competitor_df.empty:
        return dbc.Alert("Competitor data could not be loaded. Please run the data loader.", color="danger")
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.Label("Filter by Product:"), width=6),
            dbc.Col(html.Label("Filter by Date Range:"), width=6),
        ]),
        dbc.Row([
            dbc.Col(dcc.Dropdown(
                id='product-filter-dropdown',
                options=[{'label': prod, 'value': prod} for prod in sorted(competitor_df['productname'].unique())],
                value=competitor_df['productname'].unique()[0],
                clearable=False
            ), width=6, className="mb-4"),
            dbc.Col(dcc.DatePickerRange(
                id='competitor-date-picker',
                min_date_allowed=competitor_df['date'].min().date(),
                max_date_allowed=competitor_df['date'].max().date(),
                start_date=competitor_df['date'].min().date(),
                end_date=competitor_df['date'].max().date(),
                display_format='YYYY-MM-DD'
            ), width=6, className="mb-4"),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='price-trend-chart'), width=12),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='promotion-activity-chart'), width=12),
        ])
    ], fluid=True)

# ==== MAIN APP LAYOUT ====
app.layout = dbc.Container([
    dcc.Download(id="download-dataframe-csv"),
    html.H1("Pharma Analytics Hub - v12.1 (Live)", className='text-center text-primary mb-4'),
    dbc.Tabs(id="tabs-controller", active_tab="sales-tab", children=[
        dbc.Tab(label="Sales Performance", tab_id="sales-tab"),
        dbc.Tab(label="Delivery Performance", tab_id="delivery-tab"),
        dbc.Tab(label="Customer Analytics", tab_id="customer-tab"),
        dbc.Tab(label="Competitor Analysis", tab_id="competitor-tab"),
    ]),
    html.Div(id='tab-content', className="mt-4")
], fluid=True)

# --- 6. DEFINE CALLBACKS ---
@app.callback(Output('tab-content', 'children'), Input('tabs-controller', 'active_tab'))
def render_tab_content(active_tab):
    if sales_df.empty and active_tab != 'sales-tab':
        return dbc.Alert("Data could not be loaded. Please check the logs and run the data loader.", color="danger")
    if active_tab == "sales-tab": return create_sales_layout()
    elif active_tab == "delivery-tab": return create_delivery_layout()
    elif active_tab == "customer-tab": return create_customer_layout()
    elif active_tab == "competitor-tab": return create_competitor_layout()
    return html.P("This tab doesn't exist.")

@app.callback(
    [Output('kpi-net-sales', 'children'), Output('kpi-total-discount', 'children'),
     Output('kpi-total-orders', 'children'), Output('sales-over-time-chart', 'figure'),
     Output('sales-by-category-chart', 'figure'), Output('top-products-chart', 'figure')],
    [Input('location-filter-dropdown', 'value'), Input('sales-date-picker', 'start_date'),
     Input('sales-date-picker', 'end_date')]
)
def update_sales_dashboard(selected_location, start_date, end_date):
    if not start_date or not end_date or sales_df.empty: raise PreventUpdate
    start_date_obj, end_date_obj = pandas.to_datetime(start_date).date(), pandas.to_datetime(end_date).date()
    date_mask = (sales_df['date'] >= start_date_obj) & (sales_df['date'] <= end_date_obj)
    loc_mask = sales_df['locationid'] == selected_location if selected_location != 'All' else True
    filtered_df = sales_df.loc[date_mask & loc_mask]
    net_sales, total_discount, total_orders = filtered_df['netsale'].sum(), filtered_df['discountvalue'].sum(), filtered_df['orderid'].nunique()
    kpi_netsales_card = dbc.CardBody([html.H4("Total Net Sales"), html.P(f"{net_sales:,.2f} SAR", className="fs-3")])
    kpi_discount_card = dbc.CardBody([html.H4("Total Discount"), html.P(f"{total_discount:,.2f} SAR", className="fs-3")])
    kpi_orders_card = dbc.CardBody([html.H4("Total Orders"), html.P(f"{total_orders:,}", className="fs-3")])
    daily_sales = filtered_df.groupby('date')['netsale'].sum().reset_index()
    sales_over_time_fig = px.line(daily_sales, x='date', y='netsale', title='Daily Net Sales')
    category_sales = filtered_df.groupby('category')['netsale'].sum().reset_index()
    sales_by_cat_fig = px.pie(category_sales, names='category', values='netsale', title='Sales by Category', hole=0.3)
    product_sales = filtered_df.groupby('productname')['netsale'].sum().nlargest(10).reset_index()
    top_prod_fig = px.bar(product_sales, x='netsale', y='productname', orientation='h', title='Top 10 Products by Net Sales').update_layout(yaxis={'categoryorder':'total ascending'})
    return kpi_netsales_card, kpi_discount_card, kpi_orders_card, sales_over_time_fig, sales_by_cat_fig, top_prod_fig

@app.callback(
    [Output('kpi-on-time-delivery', 'children'), Output('kpi-avg-delivery-time', 'children'),
     Output('kpi-total-deliveries', 'children'), Output('delivery-status-chart', 'figure'),
     Output('partner-performance-chart', 'figure')],
    [Input('delivery-date-picker', 'start_date'), Input('delivery-date-picker', 'end_date')]
)
def update_delivery_dashboard(start_date, end_date):
    if not start_date or not end_date or delivery_df.empty: raise PreventUpdate
    start_date_obj, end_date_obj = pandas.to_datetime(start_date).date(), pandas.to_datetime(end_date).date()
    mask = (delivery_df['date'] >= start_date_obj) & (delivery_df['date'] <= end_date_obj)
    filtered_df = delivery_df.loc[mask].copy()
    on_time_pct = (filtered_df['on_time'].sum() / len(filtered_df) * 100) if not filtered_df.empty else 0
    avg_delivery_time, total_deliveries = filtered_df['delivery_time_days'].mean(), len(filtered_df)
    kpi_on_time_card = dbc.CardBody([html.H4("On-Time Delivery Rate"), html.P(f"{on_time_pct:.2f}%", className="fs-3")])
    kpi_avg_time_card = dbc.CardBody([html.H4("Avg. Delivery Time"), html.P(f"{avg_delivery_time:.2f} Days", className="fs-3")])
    kpi_total_del_card = dbc.CardBody([html.H4("Total Deliveries"), html.P(f"{total_deliveries:,}", className="fs-3")])
    status_counts = filtered_df['status'].value_counts().reset_index()
    status_chart_fig = px.pie(status_counts, names='status', values='count', title='Delivery Status Breakdown', hole=0.3)
    partner_perf = filtered_df.groupby('deliverypartner')['on_time'].mean().reset_index()
    partner_perf['on_time'] *= 100
    partner_perf_chart_fig = px.bar(partner_perf.sort_values('on_time'), x='on_time', y='deliverypartner', orientation='h', title='On-Time Rate by Delivery Partner', labels={'on_time': 'On-Time %'})
    return kpi_on_time_card, kpi_avg_time_card, kpi_total_del_card, status_chart_fig, partner_perf_chart_fig

@app.callback(
    [Output('kpi-total-customers', 'children'), Output('kpi-avg-clv', 'children'),
     Output('kpi-new-customers', 'children'), Output('rfm-chart', 'figure'),
     Output('sales-by-segment-chart', 'figure'), Output('customer-segment-dist-chart', 'figure')],
    [Input('tabs-controller', 'active_tab')]
)
def update_customer_dashboard(active_tab):
    if active_tab != 'customer-tab' or rfm_df.empty: raise PreventUpdate
    total_customers, avg_clv = customer_df['customerid'].nunique(), rfm_df['monetary'].mean()
    new_customer_count = customer_df[customer_df['joindate'] > (datetime.now() - pandas.Timedelta(days=90))].shape[0]
    kpi_total_cust_card = dbc.CardBody([html.H4("Total Customers"), html.P(f"{total_customers:,}", className="fs-3")])
    kpi_clv_card = dbc.CardBody([html.H4("Avg. Customer Value"), html.P(f"{avg_clv:,.2f} SAR", className="fs-3")])
    kpi_new_cust_card = dbc.CardBody([html.H4("New Customers (90d)"), html.P(f"{new_customer_count:,}", className="fs-3")])
    rfm_fig = px.scatter(
        rfm_df, x='recency', y='frequency', size='monetary', color='segment',
        hover_name='customerid', size_max=60, title='RFM Customer Segmentation',
        labels={'recency': 'Recency (Days)', 'frequency': 'Frequency', 'segment': 'Segment'}
    ).update_layout(xaxis_autorange='reversed')
    sales_by_segment = rfm_df.groupby('segment')['monetary'].sum().reset_index()
    sales_by_segment_fig = px.bar(sales_by_segment, x='segment', y='monetary', title='Total Net Sales by Customer Segment', color='segment')
    segment_dist = customer_df['segment'].value_counts().reset_index()
    segment_dist_fig = px.pie(segment_dist, names='segment', values='count', title='Customer Segment Distribution', hole=0.3)
    return kpi_total_cust_card, kpi_clv_card, kpi_new_cust_card, rfm_fig, sales_by_segment_fig, segment_dist_fig

@app.callback(
    [Output('price-trend-chart', 'figure'), Output('promotion-activity-chart', 'figure')],
    [Input('product-filter-dropdown', 'value'), Input('competitor-date-picker', 'start_date'),
     Input('competitor-date-picker', 'end_date')]
)
def update_competitor_dashboard(selected_product, start_date, end_date):
    if not selected_product or not start_date or not end_date or competitor_df.empty: raise PreventUpdate
    start_date_obj, end_date_obj = pandas.to_datetime(start_date), pandas.to_datetime(end_date)
    mask = (competitor_df['productname'] == selected_product) & \
           (competitor_df['date'] >= start_date_obj) & \
           (competitor_df['date'] <= end_date_obj)
    filtered_df = competitor_df.loc[mask]
    price_trend_fig = px.line(filtered_df, x='date', y='price', color='competitor',
                              title=f'Price Trend for {selected_product}',
                              labels={'price': 'Price (SAR)', 'date': 'Date'})
    promo_activity = filtered_df.groupby('competitor')['onpromotion'].value_counts(normalize=True).unstack().fillna(0) * 100
    promo_activity.rename(columns={True:'On Promotion', False:'Not on Promotion'}, inplace=True)
    promo_fig = px.bar(promo_activity, barmode='stack', title=f'Promotional Activity for {selected_product}',
                       labels={'value': '% of Time', 'competitor': 'Competitor'})
    return price_trend_fig, promo_fig

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("export-csv-button", "n_clicks"),
    prevent_initial_call=True,
)
def export_customer_data(n_clicks):
    if n_clicks is None or rfm_df.empty: raise PreventUpdate
    return dcc.send_data_frame(rfm_df.to_csv, "customer_segments.csv", index=False)

# --- 7. RUN THE APP ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(host='0.0.0.0', port=port, debug=False)

