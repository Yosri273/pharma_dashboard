# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# E-commerce Data Analyst App - V9.0 (Advanced Data)
#
# This major upgrade handles a more detailed and realistic data schema.
# It introduces new KPIs like Net Sales and new visualizations for
# location-based analysis and customer segmentation.
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

# --- 2. DATABASE CONNECTION ---
DB_NAME = 'pharma_db'
DB_USER = 'mohamedyousri' # Replace with your Mac username if different
DB_CONNECTION_STRING = f"postgresql://{DB_USER}@localhost:5432/{DB_NAME}"
engine = create_engine(DB_CONNECTION_STRING)

# --- 3. LOAD AND PREPARE DATA ---
try:
    print("Connecting to database and loading data...")
    # Load Sales Data
    sales_df = pandas.read_sql('SELECT * FROM sales', engine)
    sales_df.columns = [col.lower() for col in sales_df.columns]
    sales_df['timestamp'] = pandas.to_datetime(sales_df['timestamp'])
    sales_df['date'] = sales_df['timestamp'].dt.date
    print("Sales data loaded.")

    # Load Delivery Data
    delivery_df = pandas.read_sql('SELECT * FROM deliveries', engine)
    delivery_df.columns = [col.lower() for col in delivery_df.columns]
    delivery_df['orderdate'] = pandas.to_datetime(delivery_df['orderdate'])
    delivery_df['promiseddate'] = pandas.to_datetime(delivery_df['promiseddate'])
    delivery_df['actualdeliverydate'] = pandas.to_datetime(delivery_df['actualdeliverydate'])
    delivery_df['delivery_time_days'] = (delivery_df['actualdeliverydate'] - delivery_df['orderdate']).dt.days
    delivery_df['on_time'] = delivery_df['actualdeliverydate'] <= delivery_df['promiseddate']
    delivery_df['date'] = delivery_df['orderdate'].dt.date
    print("Delivery data loaded.")
    
    # Load Customer Data
    customer_df = pandas.read_sql('SELECT * FROM customers', engine)
    customer_df.columns = [col.lower() for col in customer_df.columns]
    customer_df['joindate'] = pandas.to_datetime(customer_df['joindate'])
    customer_df['join_month'] = customer_df['joindate'].dt.to_period('M').astype(str)
    print("Customer data loaded.")
    
    # --- Data Enrichment for RFM ---
    # RFM is now based on 'netsale' for more accurate valuation
    rfm_df = sales_df.groupby('customerid').agg(
        last_purchase_date=('timestamp', 'max'),
        frequency=('orderid', 'nunique'),
        monetary=('netsale', 'sum')
    ).reset_index()
    
    current_date = datetime.now()
    rfm_df['recency'] = (current_date - rfm_df['last_purchase_date']).dt.days
    # Join with customer segment data
    rfm_df = rfm_df.merge(customer_df[['customerid', 'segment']], on='customerid', how='left')


except Exception as e:
    print("\n--- DATABASE CONNECTION ERROR ---")
    print("Could not load data. Please ensure you have run the latest `load_data.py` successfully.")
    print(f"Error details: {e}")
    exit()

# --- 4. INITIALIZE THE APP ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE], suppress_callback_exceptions=True)
server = app.server

# --- 5. DEFINE REUSABLE COMPONENTS AND LAYOUTS ---

# ==== SALES DASHBOARD LAYOUT ====
sales_layout = dbc.Container([
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
        dbc.Col(dcc.Graph(id='gross-vs-net-chart'), width=6),
        dbc.Col(dcc.Graph(id='sales-by-location-chart'), width=6),
    ])
], fluid=True)

# ==== DELIVERY DASHBOARD LAYOUT (Unchanged) ====
delivery_layout = dbc.Container([
    # ... (layout is the same as before) ...
], fluid=True)

# ==== CUSTOMER DASHBOARD LAYOUT ====
customer_layout = dbc.Container([
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
        dbc.Col(dcc.Graph(id='customer-segment-dist-chart'), width=6),
        dbc.Col(dcc.Graph(id='customer-by-city-chart'), width=6),
    ])
], fluid=True)

# ==== MAIN APP LAYOUT WITH TABS ====
app.layout = dbc.Container([
    dcc.Download(id="download-dataframe-csv"),
    html.H1("Pharma Analytics Hub - v9.0 (Advanced)", className='text-center text-primary mb-4'),
    dbc.Tabs(id="tabs-controller", active_tab="sales-tab", children=[
        dbc.Tab(label="Sales Performance", tab_id="sales-tab"),
        dbc.Tab(label="Delivery Performance", tab_id="delivery-tab"),
        dbc.Tab(label="Customer Analytics", tab_id="customer-tab"),
    ]),
    html.Div(id='tab-content', className="mt-4")
], fluid=True)

# --- 6. DEFINE CALLBACKS ---

# Callback to render the correct tab content
@app.callback(Output('tab-content', 'children'), Input('tabs-controller', 'active_tab'))
def render_tab_content(active_tab):
    if active_tab == "sales-tab":
        delivery_layout.children = [] # Clear other layouts to avoid ID conflicts
        customer_layout.children = []
        return sales_layout
    elif active_tab == "delivery-tab":
        sales_layout.children = []
        customer_layout.children = []
        # Re-populate delivery layout since it's now active
        delivery_layout.children = [
            dbc.Row([dbc.Col(html.Label("Filter by Order Date Range:"), width=12)]),
            dbc.Row([dbc.Col(dcc.DatePickerRange(
                id='delivery-date-picker', min_date_allowed=delivery_df['date'].min(),
                max_date_allowed=delivery_df['date'].max(), start_date=delivery_df['date'].min(),
                end_date=delivery_df['date'].max(), display_format='YYYY-MM-DD'
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
        ]
        return delivery_layout
    elif active_tab == "customer-tab":
        sales_layout.children = []
        delivery_layout.children = []
        return customer_layout
    return html.P("This tab doesn't exist.")

# ADVANCED Callback for Sales Dashboard
@app.callback(
    [Output('kpi-net-sales', 'children'), Output('kpi-total-discount', 'children'),
     Output('kpi-total-orders', 'children'), Output('sales-over-time-chart', 'figure'),
     Output('gross-vs-net-chart', 'figure'), Output('sales-by-location-chart', 'figure')],
    [Input('location-filter-dropdown', 'value'), Input('sales-date-picker', 'start_date'),
     Input('sales-date-picker', 'end_date'), Input('tabs-controller', 'active_tab')]
)
def update_sales_dashboard(selected_location, start_date, end_date, active_tab):
    if active_tab != 'sales-tab' or not start_date or not end_date:
        raise PreventUpdate
    
    start_date_obj = pandas.to_datetime(start_date).date()
    end_date_obj = pandas.to_datetime(end_date).date()
    date_mask = (sales_df['date'] >= start_date_obj) & (sales_df['date'] <= end_date_obj)
    
    if selected_location != 'All':
        loc_mask = sales_df['locationid'] == selected_location
        combined_mask = date_mask & loc_mask
    else:
        combined_mask = date_mask
        
    filtered_df = sales_df.loc[combined_mask]

    net_sales = filtered_df['netsale'].sum()
    total_discount = filtered_df['discountvalue'].sum()
    total_orders = filtered_df['orderid'].nunique()

    kpi_netsales_card = dbc.CardBody([html.H4("Total Net Sales"), html.P(f"{net_sales:,.2f} SAR", className="fs-3")])
    kpi_discount_card = dbc.CardBody([html.H4("Total Discount"), html.P(f"{total_discount:,.2f} SAR", className="fs-3")])
    kpi_orders_card = dbc.CardBody([html.H4("Total Orders"), html.P(f"{total_orders:,}", className="fs-3")])

    daily_sales = filtered_df.groupby('date')[['grossvalue', 'netsale']].sum().reset_index()
    sales_over_time_fig = px.line(daily_sales, x='date', y='netsale', title='Daily Net Sales')
    
    gross_vs_net_fig = px.bar(daily_sales.sum().to_frame().T, x=['grossvalue', 'netsale'], y=0, barmode='group', title='Gross vs. Net Sales', labels={'value':'Amount', 'variable':'Value Type'}).update_xaxes(title_text=None)
    
    location_sales = filtered_df.groupby('locationid')['netsale'].sum().nlargest(10).reset_index()
    sales_by_loc_fig = px.bar(location_sales, x='netsale', y='locationid', orientation='h', title='Top 10 Locations by Net Sales').update_layout(yaxis={'categoryorder':'total ascending'})

    return kpi_netsales_card, kpi_discount_card, kpi_orders_card, sales_over_time_fig, gross_vs_net_fig, sales_by_loc_fig

# ADVANCED Callback for Customer Dashboard
@app.callback(
    [Output('kpi-total-customers', 'children'), Output('kpi-avg-clv', 'children'),
     Output('kpi-new-customers', 'children'), Output('rfm-chart', 'figure'),
     Output('customer-segment-dist-chart', 'figure'), Output('customer-by-city-chart', 'figure')],
    [Input('tabs-controller', 'active_tab')]
)
def update_customer_dashboard(active_tab):
    if active_tab != 'customer-tab':
        raise PreventUpdate

    total_customers = customer_df['customerid'].nunique()
    avg_clv = rfm_df['monetary'].mean()
    new_customer_count = customer_df[customer_df['joindate'] > (datetime.now() - pandas.Timedelta(days=90))].shape[0]

    kpi_total_cust_card = dbc.CardBody([html.H4("Total Customers"), html.P(f"{total_customers:,}", className="fs-3")])
    kpi_clv_card = dbc.CardBody([html.H4("Avg. Customer Value"), html.P(f"{avg_clv:,.2f} SAR", className="fs-3")])
    kpi_new_cust_card = dbc.CardBody([html.H4("New Customers (90d)"), html.P(f"{new_customer_count:,}", className="fs-3")])

    rfm_fig = px.scatter(
        rfm_df, x='recency', y='frequency', size='monetary', color='segment',
        hover_name='customerid', size_max=60, title='RFM Customer Segmentation',
        labels={'recency': 'Recency (Days)', 'frequency': 'Frequency', 'segment': 'Segment'}
    ).update_layout(xaxis_autorange='reversed')

    segment_dist = customer_df['segment'].value_counts().reset_index()
    segment_dist_fig = px.pie(segment_dist, names='segment', values='count', title='Customer Segment Distribution', hole=0.3)

    city_dist = customer_df['city'].value_counts().reset_index()
    city_dist_fig = px.pie(city_dist, names='city', values='count', title='Customer Distribution by City', hole=0.3)

    return kpi_total_cust_card, kpi_clv_card, kpi_new_cust_card, rfm_fig, segment_dist_fig, city_dist_fig

# Callback for Delivery Dashboard (re-defined to ensure it works with dynamic layouts)
@app.callback(
    [Output('kpi-on-time-delivery', 'children'), Output('kpi-avg-delivery-time', 'children'),
     Output('kpi-total-deliveries', 'children'), Output('delivery-status-chart', 'figure'),
     Output('partner-performance-chart', 'figure')],
    [Input('delivery-date-picker', 'start_date'), Input('delivery-date-picker', 'end_date'),
     Input('tabs-controller', 'active_tab')]
)
def update_delivery_dashboard(start_date, end_date, active_tab):
    if active_tab != 'delivery-tab' or not start_date or not end_date:
        raise PreventUpdate

    start_date_obj = pandas.to_datetime(start_date).date()
    end_date_obj = pandas.to_datetime(end_date).date()
    mask = (delivery_df['date'] >= start_date_obj) & (delivery_df['date'] <= end_date_obj)
    filtered_df = delivery_df.loc[mask].copy()

    on_time_pct = (filtered_df['on_time'].sum() / len(filtered_df) * 100) if not filtered_df.empty else 0
    avg_delivery_time = filtered_df['delivery_time_days'].mean()
    total_deliveries = len(filtered_df)

    kpi_on_time_card = dbc.CardBody([html.H4("On-Time Delivery Rate"), html.P(f"{on_time_pct:.2f}%", className="fs-3")])
    kpi_avg_time_card = dbc.CardBody([html.H4("Avg. Delivery Time"), html.P(f"{avg_delivery_time:.2f} Days", className="fs-3")])
    kpi_total_del_card = dbc.CardBody([html.H4("Total Deliveries"), html.P(f"{total_deliveries:,}", className="fs-3")])

    status_counts = filtered_df['status'].value_counts().reset_index()
    status_chart_fig = px.pie(status_counts, names='status', values='count', title='Delivery Status Breakdown', hole=0.3)
    
    partner_perf = filtered_df.groupby('deliverypartner')['on_time'].mean().reset_index()
    partner_perf['on_time'] = partner_perf['on_time'] * 100
    partner_perf_chart_fig = px.bar(partner_perf.sort_values('on_time'), x='on_time', y='deliverypartner', orientation='h', title='On-Time Rate by Delivery Partner', labels={'on_time': 'On-Time %'})

    return kpi_on_time_card, kpi_avg_time_card, kpi_total_del_card, status_chart_fig, partner_perf_chart_fig

# Callback for Exporting CSV
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("export-csv-button", "n_clicks"),
    prevent_initial_call=True,
)
def export_customer_data(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    return dcc.send_data_frame(rfm_df.to_csv, "customer_segments.csv", index=False)


# --- 7. RUN THE APP ---
if __name__ == '__main__':
    app.run(debug=True)

