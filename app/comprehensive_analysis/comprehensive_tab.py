
# --- Comprehensive Analysis Tab ---
# Layout, KPI cards, charts, funnel, alerts
# Uses shared UI helpers for consistency
# Exported via get_layout() for integration

from dash import dcc, html
import dash_bootstrap_components as dbc
from app.utils.ui_helpers import create_kpi_card, create_graph_card, create_filter_options
from app.utils.analytics_helpers import set_dark_theme
from app.comprehensive_analysis.layout import get_comprehensive_layout

"""Comprehensive Analysis Tab for Pharma Dashboard"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import os

# --- DATA LOADING ---
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), 'sample_csvs')

def load_sample_data():
    # Placeholder: helper to load sample CSVs for local development if needed.
    # Currently unused in production; kept for developer convenience.
    return None

layout = get_comprehensive_layout()

# --- CALLBACKS ---
def register_comprehensive_callbacks(app):
    """
    Registers all callbacks for the comprehensive analysis tab only.
    """
    from app.comprehensive_analysis.data_sources import load_all_sources
    from app.comprehensive_analysis.metrics import get_kpis, get_funnel_data
    import plotly.express as px
    from dash import html

    @app.callback(
        [
            Output('total_visits_kpi', 'children'),
            Output('unique_visitors_kpi', 'children'),
            Output('bounce_rate_kpi', 'children'),
            Output('avg_session_duration_kpi', 'children'),
            Output('pages_per_session_kpi', 'children'),
            Output('conversion_rate_kpi', 'children'),
            Output('top_traffic_sources_chart', 'figure'),
            Output('impressions_chart', 'figure'),
            Output('clicks_chart', 'figure'),
            Output('conversions_chart', 'figure'),
            Output('attribution_analysis_chart', 'figure'),
            Output('top_campaigns_chart', 'figure'),
            Output('gmv_kpi', 'children'),
            Output('net_sales_kpi', 'children'),
            Output('num_orders_kpi', 'children'),
            Output('aov_kpi', 'children'),
            Output('repeat_purchase_rate_kpi', 'children'),
            Output('cart_abandonment_rate_kpi', 'children'),
            Output('product_performance_chart', 'figure'),
            Output('clv_kpi', 'children'),
            Output('churn_rate_kpi', 'children'),
            Output('active_customers_kpi', 'children'),
            Output('dormant_customers_kpi', 'children'),
            Output('new_customers_kpi', 'children'),
            Output('returning_customers_kpi', 'children'),
            Output('segmentation_chart', 'figure'),
            Output('nps_chart', 'figure'),
            Output('on_time_delivery_rate_kpi', 'children'),
            Output('avg_delivery_time_kpi', 'children'),
            Output('delivery_cost_per_order_kpi', 'children'),
            Output('return_rate_kpi', 'children'),
            Output('support_ticket_volume_chart', 'figure'),
            Output('resolution_time_chart', 'figure'),
            Output('top_issues_chart', 'figure'),
            Output('funnel_visualization_chart', 'figure'),
            Output('dropoff_rates_chart', 'figure'),
            Output('journey_mapping_chart', 'figure'),
            Output('clv_cac_ratio_kpi', 'children'),
            Output('profitability_waterfall_chart', 'figure'),
            Output('comp-alerts-section', 'children'),
        ],
        [
            Input('comp-apply-btn', 'n_clicks'),
            State('comp-date-picker', 'start_date'),
            State('comp-date-picker', 'end_date'),
            State('traffic-device-filter', 'value'),
            State('traffic-source-filter', 'value'),
            State('traffic-session-type-filter', 'value')
        ]
    )
    def update_comprehensive_tab(n_clicks, start_date, end_date, device, source, session_type):
        # Load sources and compute core metrics
        sources = load_all_sources()
        # Prefer canonical KPIs computed by the ETL layer (single source of truth).
        try:
            from etl import transforms
            kpis = transforms.DATA.get('kpis') if transforms.DATA.get('kpis') is not None else get_kpis(sources)
        except Exception:
            # If ETL isn't available or hasn't computed KPIs yet, fall back to local computation.
            kpis = get_kpis(sources)
        funnel_data = get_funnel_data(sources)
        # Import helper metric modules for channel/customer/logistics insights
        from app.comprehensive_analysis.metrics import get_channel_performance, get_customer_insights, get_logistics_support

        channel_perf = get_channel_performance(sources)
        customer_insights = get_customer_insights(sources)
        logistics = get_logistics_support(sources)

        # Format KPIs
        def format_kpi(val, percent=False, currency=False):
            if val is None or (isinstance(val, float) and (pd.isna(val) or np.isnan(val))):
                return html.H3("-", className="mb-0 fw-bold")
            if percent:
                try:
                    return html.H3(f"{val:.2%}", className="mb-0 fw-bold")
                except Exception:
                    return html.H3(str(val), className="mb-0 fw-bold")
            if currency:
                try:
                    return html.H3(f"{val:,.2f} SAR", className="mb-0 fw-bold")
                except Exception:
                    return html.H3(str(val), className="mb-0 fw-bold")
            return html.H3(f"{val:,.2f}" if isinstance(val, float) else str(val), className="mb-0 fw-bold")

        # Alerts
        alerts = []
        if kpis.get('conversion_rate', 0) < 0.02:
            alerts.append(html.Div("⚠️ Conversion Rate is below 2%", style={"color": "red", "fontWeight": "bold"}))
        if kpis.get('cart_abandonment_rate', 0) and kpis['cart_abandonment_rate'] > 0.5:
            alerts.append(html.Div("⚠️ Cart Abandonment Rate is above 50%", style={"color": "orange", "fontWeight": "bold"}))
        promised_delivery = 2
        if kpis.get('avg_delivery_time') and kpis['avg_delivery_time'] > promised_delivery:
            alerts.append(html.Div(f"⚠️ Avg Delivery Time ({kpis['avg_delivery_time']} days) exceeds promised ({promised_delivery} days)", style={"color": "orange", "fontWeight": "bold"}))
        if not alerts:
            alerts.append(html.Div("✅ All key metrics are healthy.", style={"color": "green"}))

        # Funnel visuals (always create small df so UI receives a figure)
        funnel_stages = ['Visits', 'Add to Cart', 'Checkout', 'Purchase', 'Delivery']
        funnel_counts = [funnel_data.get(stage.lower(), 0) for stage in funnel_stages]
        funnel_df = pd.DataFrame({'Stage': funnel_stages, 'Count': funnel_counts})
        funnel_chart = set_dark_theme(px.funnel(funnel_df, x='Count', y='Stage', title='E-commerce Funnel'))
        dropoff_chart = set_dark_theme(px.bar(funnel_df, x='Stage', y='Count', title='Drop-off Rates'))

        # Map outputs to the layout's order
        return (
            format_kpi(kpis.get('total_sessions')),
            format_kpi(kpis.get('unique_users')),
            format_kpi(kpis.get('bounce_rate'), percent=True),
            format_kpi(kpis.get('avg_session_duration')),
            format_kpi(kpis.get('pages_per_session')),
            format_kpi(kpis.get('conversion_rate'), percent=True),
            kpis.get('top_traffic_sources_fig', {}),
            channel_perf.get('impressions_fig', {}),
            channel_perf.get('clicks_fig', {}),
            channel_perf.get('conversions_fig', {}),
            channel_perf.get('attribution_fig', {}),
            channel_perf.get('top_campaigns_fig', {}),
            format_kpi(kpis.get('gmv'), currency=True),
            format_kpi(kpis.get('net_sales'), currency=True),
            format_kpi(kpis.get('num_orders')),
            format_kpi(kpis.get('aov'), currency=True),
            format_kpi(kpis.get('repeat_purchase_rate'), percent=True),
            format_kpi(kpis.get('cart_abandonment_rate'), percent=True),
            kpis.get('product_performance_fig', {}),
            format_kpi(customer_insights.get('avg_clv') or kpis.get('clv'), currency=True),
            format_kpi(customer_insights.get('churn_rate') or kpis.get('churn_rate'), percent=True),
            format_kpi(customer_insights.get('active_customers')),
            format_kpi(customer_insights.get('dormant_customers')),
            format_kpi(customer_insights.get('new_customers')),
            format_kpi(customer_insights.get('returning_customers')),
            customer_insights.get('segmentation_fig', {}),
            customer_insights.get('nps_fig', {}),
            format_kpi(logistics.get('on_time_delivery_rate', kpis.get('on_time_delivery_rate')), percent=True),
            format_kpi(logistics.get('avg_delivery_time', kpis.get('avg_delivery_time'))),
            format_kpi(logistics.get('delivery_cost_per_order'), currency=True),
            format_kpi(logistics.get('return_rate')),
            logistics.get('support_ticket_volume_fig', {}),
            logistics.get('resolution_time_fig', {}),
            logistics.get('top_issues_fig', {}),
            funnel_data.get('funnel_visualization_fig', funnel_chart),
            funnel_data.get('dropoff_rates_fig', dropoff_chart),
            funnel_data.get('journey_mapping_fig', {}),
            format_kpi(funnel_data.get('clv_cac_ratio', 0)),
            funnel_data.get('profitability_waterfall_fig', {}),
            alerts,
        )

# --- END Comprehensive Analysis Tab ---
