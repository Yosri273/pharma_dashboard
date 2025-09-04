# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# E-commerce Data Analyst App - V2.0 FINAL & WORKING
#
# Corrected the critical typo from DateRangePicker to DatePickerRange.
# All functionality has been restored.
# -----------------------------------------------------------------------------

# --- 1. IMPORT LIBRARIES ---
import pandas
import plotly.express as px
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
from datetime import date

# --- 2. LOAD AND PREPARE THE DATA ---
DATA_FILE_PATH = 'sales_data.csv'
try:
    df = pandas.read_csv(DATA_FILE_PATH)
    df['Timestamp'] = pandas.to_datetime(df['Timestamp'])
    df['TotalSale'] = df['Quantity'] * df['Price']
    df['Date'] = df['Timestamp'].dt.date
except FileNotFoundError:
    print(f"Error: The data file was not found. Make sure '{DATA_FILE_PATH}' is in the same folder as app.py.")
    exit()

# --- 3. INITIALIZE THE DASH WEB APPLICATION ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])
server = app.server

# --- 4. DEFINE THE LAYOUT OF THE APPLICATION ---
app.layout = dbc.Container([
    # Row 1: The Title
    dbc.Row([
        dbc.Col(html.H1("Interactive Pharmaceutical Dashboard", className='text-center text-primary, mb-4'), width=12)
    ]),
    # Row 2: The Filters
    dbc.Row([
        dbc.Col([
            html.Label("Filter by City:"),
            dcc.Dropdown(
                id='city-filter-dropdown',
                options=[{'label': 'All', 'value': 'All'}] + [{'label': city, 'value': city} for city in sorted(df['City'].unique())],
                value='All',
                clearable=False
            )
        ], width=6, className="mb-4"),
        dbc.Col([
            html.Label("Filter by Date Range:"),
            dcc.DatePickerRange( # CORRECTED NAME
                id='date-range-picker',
                min_date_allowed=df['Date'].min(),
                max_date_allowed=df['Date'].max(),
                start_date=df['Date'].min(),
                end_date=df['Date'].max(),
                display_format='YYYY-MM-DD'
            )
        ], width=6, className="mb-4"),
    ]),
    # Row 3: KPI Cards
    dbc.Row([
        dbc.Col(dbc.Card(id='kpi-total-revenue', color="primary", inverse=True, className="mb-4"), width=4),
        dbc.Col(dbc.Card(id='kpi-total-orders', color="info", inverse=True, className="mb-4"), width=4),
        dbc.Col(dbc.Card(id='kpi-avg-transaction', color="success", inverse=True, className="mb-4"), width=4),
    ]),
    # Row 4 & 5: Charts
    dbc.Row([
        dbc.Col(dcc.Graph(id='sales-over-time-chart'), width=12),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='category-distribution-chart'), width=6),
        dbc.Col(dcc.Graph(id='sales-by-product-chart'), width=6),
    ])
], fluid=True)

# --- 5. DEFINE THE CALLBACK FOR INTERACTIVITY ---
@app.callback(
    [Output('kpi-total-revenue', 'children'),
     Output('kpi-total-orders', 'children'),
     Output('kpi-avg-transaction', 'children'),
     Output('sales-over-time-chart', 'figure'),
     Output('category-distribution-chart', 'figure'),
     Output('sales-by-product-chart', 'figure')],
    [Input('city-filter-dropdown', 'value'),
     Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date')]
)
def update_dashboard(selected_city, start_date, end_date):
    if start_date is None or end_date is None:
        return [dash.no_update] * 6

    filtered_df = df.copy()
    if selected_city != 'All':
        filtered_df = filtered_df[filtered_df['City'] == selected_city]

    start_date_obj = date.fromisoformat(start_date)
    end_date_obj = date.fromisoformat(end_date)
    
    mask = (filtered_df['Date'] >= start_date_obj) & (filtered_df['Date'] <= end_date_obj)
    filtered_df = filtered_df.loc[mask]

    total_revenue = filtered_df['TotalSale'].sum()
    total_orders = len(filtered_df)
    average_transaction_value = total_revenue / total_orders if total_orders > 0 else 0

    kpi_revenue_card = dbc.CardBody([
        html.H4("Total Revenue", className="card-title"),
        html.P(f"{total_revenue:,.2f} SAR", className="card-text fs-3")
    ])
    kpi_orders_card = dbc.CardBody([
        html.H4("Total Orders", className="card-title"),
        html.P(f"{total_orders:,}", className="card-text fs-3")
    ])
    kpi_avg_card = dbc.CardBody([
        html.H4("Avg. Transaction Value", className="card-title"),
        html.P(f"{average_transaction_value:,.2f} SAR", className="card-text fs-3")
    ])

    daily_sales = filtered_df.groupby('Date')['TotalSale'].sum().reset_index()
    sales_over_time_fig = px.line(
        daily_sales, x='Date', y='TotalSale', title=f'Daily Sales in {selected_city}',
        labels={'TotalSale': 'Total Revenue (SAR)', 'Date': 'Date'}
    )

    category_sales = filtered_df.groupby('Category')['TotalSale'].sum().reset_index()
    category_distribution_fig = px.pie(
        category_sales, names='Category', values='TotalSale',
        title=f'Sales Distribution in {selected_city}', hole=0.3
    )
    
    product_sales = filtered_df.groupby('ProductName')['TotalSale'].sum().nlargest(10).reset_index()
    sales_by_product_fig = px.bar(
        product_sales, x='TotalSale', y='ProductName', orientation='h',
        title=f'Top 10 Products in {selected_city}',
        labels={'TotalSale': 'Total Revenue (SAR)', 'ProductName': 'Product'}
    ).update_layout(yaxis={'categoryorder':'total ascending'})

    return kpi_revenue_card, kpi_orders_card, kpi_avg_card, sales_over_time_fig, category_distribution_fig, sales_by_product_fig

# --- 6. RUN THE APPLICATION ---
if __name__ == '__main__':
    app.run(debug=True)

