__all__ = ['get_comprehensive_layout']
from dash import html, dcc
import dash_bootstrap_components as dbc
from app.utils.ui_helpers import create_kpi_card, create_graph_card

def get_comprehensive_layout():
    return dbc.Container([
    # --- Comprehensive Tab Controls ---
        dbc.Row([
            dbc.Col([
                html.Label("Date Range:"),
                dcc.DatePickerRange(
                    id='comp-date-picker',
                    min_date_allowed=None,
                    max_date_allowed=None,
                    start_date=None,
                    end_date=None,
                    className="d-block"
                )
            ], width=12, lg=3, className="mb-2"),
            dbc.Col([
                html.Label("Device Type:"),
                dcc.Dropdown(id='traffic-device-filter', options=[{'label': 'All', 'value': 'All'}, {'label': 'Web', 'value': 'Web'}, {'label': 'Mobile', 'value': 'Mobile'}], value='All', clearable=False)
            ], width=12, lg=2, className="mb-2"),
            dbc.Col([
                html.Label("Traffic Source:"),
                dcc.Dropdown(id='traffic-source-filter', options=[{'label': 'All', 'value': 'All'}, {'label': 'Organic', 'value': 'Organic'}, {'label': 'Paid', 'value': 'Paid'}], value='All', clearable=False)
            ], width=12, lg=2, className="mb-2"),
            dbc.Col([
                html.Label("Session Type:"),
                dcc.Dropdown(id='traffic-session-type-filter', options=[{'label': 'All', 'value': 'All'}, {'label': 'New', 'value': 'New'}, {'label': 'Returning', 'value': 'Returning'}], value='All', clearable=False)
            ], width=12, lg=2, className="mb-2"),
            dbc.Col([
                html.Label("Marketing Channel:"),
                dcc.Dropdown(id='marketing-channel-filter', options=[{'label': 'All', 'value': 'All'}, {'label': 'Email', 'value': 'Email'}, {'label': 'Social', 'value': 'Social'}, {'label': 'Search', 'value': 'Search'}], value='All', clearable=False)
            ], width=12, lg=2, className="mb-2"),
            dbc.Col([
                html.Label("Sales Region:"),
                dcc.Dropdown(id='sales-region-filter', options=[{'label': 'All', 'value': 'All'}, {'label': 'Riyadh', 'value': 'Riyadh'}, {'label': 'Jeddah', 'value': 'Jeddah'}, {'label': 'Dammam', 'value': 'Dammam'}], value='All', clearable=False)
            ], width=12, lg=2, className="mb-2"),
            dbc.Col([
                html.Label("Sales Category:"),
                dcc.Dropdown(id='sales-category-filter', options=[{'label': 'All', 'value': 'All'}, {'label': 'Vitamins', 'value': 'Vitamins'}, {'label': 'Personal Care', 'value': 'Personal Care'}, {'label': 'Skincare', 'value': 'Skincare'}], value='All', clearable=False)
            ], width=12, lg=2, className="mb-2"),
            dbc.Col([
                dbc.Button("Apply", id='comp-apply-btn', color="primary", className="w-100")
            ], width=12, lg=2, className="mb-2 align-self-end"),
            # Small validator badge area (populated asynchronously)
            dbc.Col(html.Div(id='metrics-validator-badge', children=[], style={'textAlign':'right', 'paddingTop':'6px'}), width=8, lg=2, className="mb-2"),
            dbc.Col(dbc.Button("Diagnostics", id='comp-diagnostics-btn', color='light', size='sm'), width=4, lg=1, className="mb-2 text-end"),
        ], align="bottom"),
        # Section 1: Traffic & Engagement (Web & Mobile Analytics)
        dbc.Row([
            dbc.Col(html.H4([html.Span("1.", className="section-number"), " Traffic & Engagement (Web & Mobile Analytics)"], className="comp-section-title"), width=12, className="text-center" )
        ], className="mb-3"),
        dbc.Row([
            create_kpi_card("Total Visits/Sessions", "total_visits_kpi", color="primary", width=3),
            create_kpi_card("Unique Visitors/Users", "unique_visitors_kpi", color="info", width=3),
            create_kpi_card("Bounce Rate", "bounce_rate_kpi", color="warning", width=3),
            create_kpi_card("Avg. Session Duration", "avg_session_duration_kpi", color="secondary", width=3),
        ], className="mb-3"),
        dbc.Row([
            create_kpi_card("Pages/Session", "pages_per_session_kpi", color="success", width=3),
            create_kpi_card("Conversion Rate", "conversion_rate_kpi", color="danger", width=3),
        ], className="mb-3"),
        dbc.Row([
            create_graph_card("top_traffic_sources_chart", "Top Traffic Sources", width=12),
        ], className="mb-4"),
        # ...existing code for other sections, ensure all output IDs match those in callbacks.py and expect normalized columns...

        # Section 2: Customer Acquisition & Marketing Performance
        dbc.Row([
            dbc.Col(html.H4([html.Span("2.", className="section-number"), " Customer Acquisition & Marketing Performance"], className="comp-section-title"), width=12, className="text-center" )
        ], className="mb-3"),
        dbc.Row([
            create_kpi_card("Customer Acquisition Cost (CAC)", "cac_kpi", color="primary", width=3),
            create_kpi_card("Return on Ad Spend (ROAS)", "roas_kpi", color="info", width=3),
            create_kpi_card("Click-Through Rate (CTR)", "ctr_kpi", color="warning", width=3),
        ], className="mb-3"),
        dbc.Row([
            create_graph_card("impressions_chart", "Impressions by Platform"),
            create_graph_card("clicks_chart", "Clicks by Platform"),
            create_graph_card("conversions_chart", "Conversions by Platform"),
        ], className="mb-3"),
        dbc.Row([
            create_graph_card("attribution_analysis_chart", "Attribution Analysis"),
            create_graph_card("top_campaigns_chart", "Top Performing Campaigns & Channels"),
        ], className="mb-4"),

        # Section 3: Sales & Revenue (E-commerce Backend / OMS)
        dbc.Row([
            dbc.Col(html.H4([html.Span("3.", className="section-number"), " Sales & Revenue (E-commerce Backend / OMS)"], className="comp-section-title"), width=12, className="text-center" )
        ], className="mb-3"),
        dbc.Row([
            create_kpi_card("Gross Merchandise Value (GMV)", "gmv_kpi", color="primary", width=3),
            create_kpi_card("Net Sales", "net_sales_kpi", color="info", width=3),
            create_kpi_card("Number of Orders", "num_orders_kpi", color="warning", width=3),
            create_kpi_card("Average Order Value (AOV)", "aov_kpi", color="secondary", width=3),
        ], className="mb-3"),
        dbc.Row([
            create_kpi_card("Repeat Purchase Rate", "repeat_purchase_rate_kpi", color="success", width=3),
            create_kpi_card("Cart Abandonment Rate", "cart_abandonment_rate_kpi", color="danger", width=3),
        ], className="mb-3"),
        dbc.Row([
            create_graph_card("product_performance_chart", "Product Performance", width=12),
        ], className="mb-4"),

        # Section 4: Customer Insights & Retention
        dbc.Row([
            dbc.Col(html.H4([html.Span("4.", className="section-number"), " Customer Insights & Retention"], className="comp-section-title"), width=12, className="text-center" )
        ], className="mb-3"),
        dbc.Row([
            create_kpi_card("Customer Lifetime Value (CLV)", "clv_kpi", color="primary", width=3),
            create_kpi_card("Churn Rate", "churn_rate_kpi", color="danger", width=3),
            create_kpi_card("Active Customers", "active_customers_kpi", color="success", width=3),
            create_kpi_card("Dormant Customers", "dormant_customers_kpi", color="secondary", width=3),
        ], className="mb-3"),
        dbc.Row([
            create_kpi_card("New Customers", "new_customers_kpi", color="info", width=3),
            create_kpi_card("Returning Customers", "returning_customers_kpi", color="warning", width=3),
        ], className="mb-3"),
        dbc.Row([
            create_graph_card("segmentation_chart", "Segmentation"),
            create_graph_card("nps_chart", "NPS / Customer Satisfaction Score"),
        ], className="mb-4"),

        # Section 5: Logistics & Fulfillment
        dbc.Row([
            dbc.Col(html.H4([html.Span("5.", className="section-number"), " Logistics & Fulfillment"], className="comp-section-title"), width=12, className="text-center" )
        ], className="mb-3"),
        dbc.Row([
            create_kpi_card("On-Time Delivery Rate", "on_time_delivery_rate_kpi", color="primary", width=3),
            create_kpi_card("Avg. Delivery Time", "avg_delivery_time_kpi", color="info", width=3),
            create_kpi_card("Delivery Cost/Order", "delivery_cost_per_order_kpi", color="warning", width=3),
            create_kpi_card("Return Rate (Logistics)", "return_rate_kpi", color="danger", width=3),
        ], className="mb-3"),
        dbc.Row([
            create_graph_card("support_ticket_volume_chart", "Support Ticket Volume"),
            create_graph_card("resolution_time_chart", "Resolution Time"),
            create_graph_card("top_issues_chart", "Top Issues/Reasons for Support"),
        ], className="mb-4"),

        # Section 6: End-to-End Funnel & Synthesis KPIs
        dbc.Row([
            dbc.Col(html.H4([html.Span("6.", className="section-number"), " End-to-End Funnel & Synthesis KPIs"], className="comp-section-title"), width=12, className="text-center" )
        ], className="mb-3"),
        dbc.Row([
            create_graph_card("funnel_visualization_chart", "Funnel Visualization", width=12),
        ], className="mb-4"),
        dbc.Row([
            create_graph_card("dropoff_rates_chart", "Drop-off Rates at Each Funnel Stage"),
            create_graph_card("journey_mapping_chart", "Cross-platform Journey Mapping"),
            create_kpi_card("CLV:CAC Ratio", "clv_cac_ratio_kpi", color="success", width=3),
            create_graph_card("profitability_waterfall_chart", "Profitability Waterfall", width=6),
        ], className="mb-4"),

        # Section 7: Actionable Connections & Insights
        dbc.Row([
            dbc.Col(html.H4([html.Span("7.", className="section-number"), " Actionable Connections & Insights"], className="comp-section-title"), width=12, className="text-center" )
        ], className="mb-3"),
        dbc.Row([
            create_graph_card("marketing_correlation_chart", "Marketing Spend vs. Traffic/Conversion/Sales"),
            create_graph_card("funnel_bottlenecks_chart", "Funnel Bottlenecks"),
            create_graph_card("high_ltv_channels_chart", "Top Acquisition Channels for High-LTV Customers"),
            create_graph_card("operational_issues_chart", "Operational Issues Impacting Churn/Support"),
            create_graph_card("realtime_alerts_chart", "Real-time Alerts for Anomalies"),
        ], className="mb-4"),
            dbc.Row([
                dbc.Col(html.Div(id='comp-alerts-section'), width=12)
            ], className="mb-4"),
        ], fluid=True)

layout = get_comprehensive_layout()

# Diagnostics modal (hidden by default) to inspect validator and sessions outputs
DIAGNOSTICS_MODAL = dbc.Modal([
    dbc.ModalHeader("Diagnostics"),
    dbc.ModalBody([
        html.Div([
            html.H6("Metrics Validator"),
            dbc.ButtonGroup([
                dbc.Button("Copy", id='comp-diag-copy-metrics', size='sm'),
                dbc.Button("Download", id='comp-diag-download-metrics', size='sm'),
                dbc.Button("Refresh", id='comp-diag-refresh', size='sm', color='secondary')
            ], style={'float': 'right'})
        ], style={'clear': 'both'}),
        html.Pre(id='diag-metrics-validator', style={'whiteSpace': 'pre-wrap', 'maxHeight': '240px', 'overflowY': 'auto', 'backgroundColor':'#0d1117', 'color':'#e6edf3', 'padding':'8px'}),
        html.H6("Sessions Report"),
        dbc.ButtonGroup([
            dbc.Button("Copy", id='comp-diag-copy-sessions', size='sm'),
            dbc.Button("Download", id='comp-diag-download-sessions', size='sm')
        ], style={'float': 'right'}),
        html.Pre(id='diag-sessions-report', style={'whiteSpace': 'pre-wrap', 'maxHeight': '240px', 'overflowY': 'auto', 'backgroundColor':'#0d1117', 'color':'#e6edf3', 'padding':'8px'})
    ]),
    dbc.ModalFooter(dbc.Button("Close", id='comp-diagnostics-close', color='secondary'))
], id='comp-diagnostics-modal', size='lg', is_open=False)

layout.children.insert(0, DIAGNOSTICS_MODAL)