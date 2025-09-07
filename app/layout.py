# app/layout.py
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import dcc, html
import pandas as pd
from typing import Dict, Any

# --- Plotting Helpers (from original utils.py) ---
# These functions are part of the 'view'/'presentation' layer

def create_indicator(title: str, value_str: str, id: str) -> dbc.Card:
    """Creates a KPI indicator card component."""
    return dbc.Card(
        dbc.CardBody([
            html.P(title, className="card-title"),
            html.H2(value_str, className="card-subtitle", id=id),
        ]),
        className="text-center",
        color="light"
    )

def create_time_series_chart(df: pd.DataFrame, title: str) -> go.Figure:
    """Creates a standardized time series bar chart."""
    fig = go.Figure()
    if not df.empty:
        fig.add_trace(go.Bar(
            x=df['order_date'],
            y=df['total_sales'],
            name='Sales'
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Total Sales",
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def create_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> go.Figure:
    """Creates a standardized horizontal bar chart."""
    fig = go.Figure()
    if not df.empty:
        fig.add_trace(go.Bar(
            y=df[x_col],  # Horizontal
            x=df[y_col],
            orientation='h',
        ))
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Total Sales",
        yaxis_title=x_col.replace('_', ' ').title(),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    if not df.empty:
        fig.update_yaxes(categoryorder="total ascending") # Show highest at top
    return fig

def create_pie_chart(df: pd.DataFrame, names_col: str, values_col: str, title: str) -> go.Figure:
    """Creates a standardized pie chart."""
    fig = go.Figure()
    if not df.empty:
        fig.add_trace(go.Pie(
            labels=df[names_col],
            values=df[values_col],
            hole=.3
        ))
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# --- Main Layout Definition (from original layouts.py) ---

def create_sidebar() -> html.Div:
    """Creates the navigation sidebar."""
    return html.Div(
        [
            html.H2("E-Com Dash", className="display-4"),
            html.Hr(),
            html.P("Analytics for E-Commerce Sales", className="lead"),
            dbc.Nav(
                [
                    dbc.NavLink("Overview", href="/", active="exact"),
                    # Add more links here as the app grows
                ],
                vertical=True,
                pills=True,
            ),
        ],
        className="p-4",
        style={"background-color": "#f8f9fa", "height": "100vh"}
    )

def create_main_content() -> html.Div:
    """Creates the main content area with placeholders for graphs and KPIs."""
    return html.Div(
        [
            # Header Row with Date Picker
            dbc.Row([
                dbc.Col(html.H1("Sales Overview"), md=8),
                dbc.Col(
                    dcc.DatePickerRange(
                        id='date-picker-range',
                        display_format='YYYY-MM-DD',
                        start_date_placeholder_text='Start Date',
                        end_date_placeholder_text='End Date',
                        className="w-100"
                    ),
                    md=4,
                ),
            ], className="align-items-center mb-4"),
            
            # KPI Row
            dbc.Row([
                dbc.Col(create_indicator("Total Sales", "...", id="kpi-total-sales"), md=3),
                dbc.Col(create_indicator("Avg. Order Value", "...", id="kpi-avg-order-value"), md=3),
                dbc.Col(create_indicator("Total Orders", "...", id="kpi-total-orders"), md=3),
                dbc.Col(create_indicator("Conversion Rate", "...", id="kpi-conversion-rate"), md=3),
            ], className="mb-4"),
            
            # Charts Row 1
            dbc.Row([
                dbc.Col(dcc.Graph(id='graph-sales-over-time'), md=12),
            ], className="mb-4"),

            # Charts Row 2
            dbc.Row([
                dbc.Col(dcc.Graph(id='graph-top-products'), md=8),
                dbc.Col(dcc.Graph(id='graph-sales-by-region'), md=4),
            ]),
        ],
        className="p-4"
    )

def create_layout() -> dbc.Container:
    """Generates the full application layout."""
    return dbc.Container(
        [
            dbc.Row([
                dbc.Col(create_sidebar(), md=2),
                dbc.Col(create_main_content(), md=10),
            ]),
        ],
        fluid=True,
        className="vh-100"
    )