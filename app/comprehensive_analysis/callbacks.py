# pharma_dashboard/app/comprehensive_analysis/callbacks.py

from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
from datetime import datetime
import pandas as pd
import numpy as np

from app.comprehensive_analysis.metrics import (
    get_kpis, get_funnel_data, get_channel_performance,
    get_customer_insights, get_logistics_support, get_alerts
)
from etl.transforms import DATA as TRANSFORMS_DATA
from app.comprehensive_analysis.data_sources import load_all_sources
from app.utils.ui_helpers import create_graph_card
from app.utils.kpi import create_placeholder_figure
import plotly.graph_objects as go
from app.utils.analytics_helpers import set_dark_theme
import logging

logger = logging.getLogger(__name__)

def register_callbacks(app):
    """Registers all callbacks for the comprehensive analysis tab."""
    @app.callback(
        [
            Output('total_visits_kpi', 'children'),
            Output('unique_visitors_kpi', 'children'),
            Output('bounce_rate_kpi', 'children'),
            Output('avg_session_duration_kpi', 'children'),
            Output('pages_per_session_kpi', 'children'),
            Output('conversion_rate_kpi', 'children'),
            Output('top_traffic_sources_chart', 'figure'),
            Output('cac_kpi', 'children'),
            Output('roas_kpi', 'children'),
            Output('ctr_kpi', 'children'),
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
            Output('marketing_correlation_chart', 'figure'),
            Output('funnel_bottlenecks_chart', 'figure'),
            Output('high_ltv_channels_chart', 'figure'),
            Output('operational_issues_chart', 'figure'),
            Output('realtime_alerts_chart', 'figure')
        ],
        Input('comp-apply-btn', 'n_clicks'),
        [
            State('comp-date-picker', 'start_date'),
            State('comp-date-picker', 'end_date'),
            State('traffic-device-filter', 'value'),
            State('traffic-source-filter', 'value'),
            State('traffic-session-type-filter', 'value'),
            State('marketing-channel-filter', 'value'),
            State('sales-region-filter', 'value'),
            State('sales-category-filter', 'value')
        ]
    )
    def update_comprehensive_dashboard(n, sd, ed, device, source, session_type, channel, region, category):
        try:
            if n is None:
                n = 1
            sources = load_all_sources()
            # Prefer canonical KPIs computed by ETL layer
            from etl.transforms import DATA as TRANSFORM_DATA
            kpis = TRANSFORM_DATA.get('kpis') if TRANSFORM_DATA.get('kpis') else get_kpis(sources)
            funnel = get_funnel_data(sources)
            channel_perf = get_channel_performance(sources)
            if isinstance(channel_perf, list):
                channel_perf = channel_perf[0] if len(channel_perf) > 0 else {}
            customer = get_customer_insights(sources)
            logistics = get_logistics_support(sources)
            alerts = get_alerts(sources)

            def kpi(val, percent=False, currency=False):
                if val is None or (isinstance(val, float) and (pd.isna(val) or np.isnan(val))):
                    return html.H3("-", className="mb-0 fw-bold")
                try:
                    if percent:
                        return html.H3(f"{val:.2%}", className="mb-0 fw-bold")
                    if currency:
                        return html.H3(f"{val:,.2f} SAR", className="mb-0 fw-bold")
                    return html.H3(f"{val:,.2f}" if isinstance(val, float) else str(val), className="mb-0 fw-bold")
                except Exception:
                    return html.H3(str(val), className="mb-0 fw-bold")

            def fig(data, title=None):
                if data is None or (isinstance(data, dict) and not data):
                    return {}
                return data

            # Traffic KPIs
            traffic_kpis = [
                kpi(kpis.get('total_sessions')),
                kpi(kpis.get('unique_users')),
                kpi(kpis.get('bounce_rate'), percent=True),
                kpi(kpis.get('avg_session_duration')),
                kpi(kpis.get('pages_per_session')),
                kpi(kpis.get('conversion_rate'), percent=True),
                fig(kpis.get('top_traffic_sources_fig'), "Top Traffic Sources")
            ]

            # Marketing KPIs
            marketing_kpis = [
                kpi(channel_perf.get('cac'), currency=True),
                kpi(channel_perf.get('roas'), currency=True),
                kpi(channel_perf.get('ctr'), percent=True),
                fig(channel_perf.get('impressions_fig'), "Impressions"),
                fig(channel_perf.get('clicks_fig'), "Clicks"),
                fig(channel_perf.get('conversions_fig'), "Conversions"),
                fig(channel_perf.get('attribution_fig'), "Attribution Analysis"),
                fig(channel_perf.get('top_campaigns_fig'), "Top Campaigns")
            ]

            # Sales KPIs
            sales_kpis = [
                kpi(kpis.get('gmv'), currency=True),
                kpi(kpis.get('net_sales'), currency=True),
                kpi(kpis.get('num_orders')),
                kpi(kpis.get('aov'), currency=True),
                kpi(kpis.get('repeat_purchase_rate'), percent=True),
                kpi(kpis.get('cart_abandonment_rate'), percent=True),
                fig(kpis.get('product_performance_fig'), "Product Performance")
            ]

            # Customer KPIs
            customer_kpis = [
                kpi(customer.get('clv'), currency=True),
                kpi(customer.get('churn_rate'), percent=True),
                kpi(customer.get('active_customers')),
                kpi(customer.get('dormant_customers')),
                kpi(customer.get('new_customers')),
                kpi(customer.get('returning_customers')),
                fig(customer.get('segmentation_fig'), "Segmentation"),
                fig(customer.get('nps_fig'), "NPS")
            ]

            # Logistics KPIs
            logistics_kpis = [
                kpi(logistics.get('on_time_delivery_rate'), percent=True),
                kpi(logistics.get('avg_delivery_time')),
                kpi(logistics.get('delivery_cost_per_order'), currency=True),
                kpi(logistics.get('return_rate'), percent=True),
                fig(logistics.get('support_ticket_volume_fig'), "Support Ticket Volume"),
                fig(logistics.get('resolution_time_fig'), "Resolution Time"),
                fig(logistics.get('top_issues_fig'), "Top Issues")
            ]

            # Synthesis & Funnel KPIs — compute CLV:CAC as a numeric KPI (fallback to DATA['synthesis_kpis'])
            fallback_used = False
            try:
                clv_val = funnel.get('clv_cac_ratio') if funnel is not None else None
                if clv_val is None or (isinstance(clv_val, float) and pd.isna(clv_val)):
                    # fallback to global synthesis_kpis computed during ETL
                    from etl.transforms import DATA as GLOBAL_DATA
                    clv_val = GLOBAL_DATA.get('synthesis_kpis', {}).get('clv_cac_ratio', 0)
                    fallback_used = True
                clv_numeric = float(clv_val) if clv_val is not None else 0.0
            except Exception:
                clv_numeric = 0.0
                fallback_used = True

            # Build KPI component; if fallback was used and value is zero, show a short hint to the user
            clv_kpi_component = kpi(clv_numeric)
            if fallback_used and (clv_numeric == 0 or pd.isna(clv_numeric)):
                clv_kpi_component = html.Div([
                    clv_kpi_component,
                    html.Small(
                        "CLV:CAC unavailable: ETL could not compute this metric because prediction LTV or conversion data is missing. Provide predictions with 'churn_probability' and 'Estimated_LTV' and campaign conversion data to compute.",
                        className="text-muted",
                        style={"display": "block", "marginTop": "4px", "fontSize": "0.75rem"}
                    )
                ])

            synthesis_kpis = [
                fig(funnel.get('funnel_visualization_fig'), "Funnel Visualization"),
                fig(funnel.get('dropoff_rates_fig'), "Drop-off Rates"),
                fig(funnel.get('journey_mapping_fig'), "Journey Mapping"),
                clv_kpi_component,
                fig(funnel.get('profitability_waterfall_fig'), "Profitability Waterfall")
            ]

            # Alerts
            if not alerts:
                alerts_section = html.Div("✅ All key metrics are healthy.", style={"color": "green"})
            else:
                alerts_section = html.Ul([html.Li(a) for a in alerts])

            # Section 7: Actionable Connections & Insights (cohort, retention, LTV)
            # Use figures from customer insights when available
            marketing_corr_fig = customer.get('cohort_fig') if customer.get('cohort_fig') is not None else create_placeholder_figure('No cohort data')
            funnel_bottlenecks_fig = customer.get('retention_curve_fig') if customer.get('retention_curve_fig') is not None else create_placeholder_figure('No retention data')
            high_ltv_channels_fig = customer.get('ltv_distribution_fig') if customer.get('ltv_distribution_fig') is not None else create_placeholder_figure('No LTV data')
            operational_issues_fig = create_placeholder_figure('No operational issues data')
            realtime_alerts_fig = create_placeholder_figure('No realtime alerts data')

            outputs = [
                *traffic_kpis,
                *marketing_kpis,
                *sales_kpis,
                *customer_kpis,
                *logistics_kpis,
                *synthesis_kpis,
                alerts_section,
                marketing_corr_fig,
                funnel_bottlenecks_fig,
                high_ltv_channels_fig,
                operational_issues_fig,
                realtime_alerts_fig
            ]

            return tuple(outputs)

    

        except Exception as e:
            logger.exception("Error updating comprehensive dashboard: %s", e)
            empty_kpi = html.H3("-", className="mb-0 fw-bold")
            empty_fig = {}
            empty_alerts = html.Div("Data unavailable", style={"color": "gray"})
            defaults = [
                empty_kpi,  # total_visits_kpi
                empty_kpi,  # unique_visitors_kpi
                empty_kpi,  # bounce_rate_kpi
                empty_kpi,  # avg_session_duration_kpi
                empty_kpi,  # pages_per_session_kpi
                empty_kpi,  # conversion_rate_kpi
                empty_fig,  # top_traffic_sources_chart
                empty_kpi,  # cac_kpi
                empty_kpi,  # roas_kpi
                empty_kpi,  # ctr_kpi
                empty_fig,  # impressions_chart
                empty_fig,  # clicks_chart
                empty_fig,  # conversions_chart
                empty_fig,  # attribution_analysis_chart
                empty_fig,  # top_campaigns_chart
                empty_kpi,  # gmv_kpi
                empty_kpi,  # net_sales_kpi
                empty_kpi,  # num_orders_kpi
                empty_kpi,  # aov_kpi
                empty_kpi,  # repeat_purchase_rate_kpi
                empty_kpi,  # cart_abandonment_rate_kpi
                empty_fig,  # product_performance_chart
                empty_kpi,  # clv_kpi
                empty_kpi,  # churn_rate_kpi
                empty_kpi,  # active_customers_kpi
                empty_kpi,  # dormant_customers_kpi
                empty_kpi,  # new_customers_kpi
                empty_kpi,  # returning_customers_kpi
                empty_fig,  # segmentation_chart
                empty_fig,  # nps_chart
                empty_kpi,  # on_time_delivery_rate_kpi
                empty_kpi,  # avg_delivery_time_kpi
                empty_kpi,  # delivery_cost_per_order_kpi
                empty_kpi,  # return_rate_kpi
                empty_fig,  # support_ticket_volume_chart
                empty_fig,  # resolution_time_chart
                empty_fig,  # top_issues_chart
                empty_fig,  # funnel_visualization_chart
                empty_fig,  # dropoff_rates_chart
                empty_fig,  # journey_mapping_chart
                empty_kpi,  # clv_cac_ratio_kpi
                empty_fig,  # profitability_waterfall_chart
                empty_alerts,  # comp-alerts-section
                empty_fig,  # marketing_correlation_chart
                empty_fig,  # funnel_bottlenecks_chart
                empty_fig,  # high_ltv_channels_chart
                empty_fig,  # operational_issues_chart
                empty_fig   # realtime_alerts_chart
            ]
            return tuple(defaults)

        def fig(data, title):
            if not data or (isinstance(data, dict) and not data):
                return {}
            return data

        # Traffic KPIs
        traffic_kpis = [
            kpi(kpis.get('total_sessions')),
            kpi(kpis.get('unique_users')),
            kpi(kpis.get('bounce_rate'), percent=True),
            kpi(kpis.get('avg_session_duration')),
            kpi(kpis.get('pages_per_session')),
            kpi(kpis.get('conversion_rate'), percent=True),
            fig(kpis.get('top_traffic_sources_fig'), "Top Traffic Sources")
        ]

        # Marketing KPIs
        marketing_kpis = [
            kpi(channel_perf.get('cac'), currency=True),
            kpi(channel_perf.get('roas'), currency=True),
            kpi(channel_perf.get('ctr'), percent=True),
            fig(channel_perf.get('impressions_fig'), "Impressions"),
            fig(channel_perf.get('clicks_fig'), "Clicks"),
            fig(channel_perf.get('conversions_fig'), "Conversions"),
            fig(channel_perf.get('attribution_fig'), "Attribution Analysis"),
            fig(channel_perf.get('top_campaigns_fig'), "Top Campaigns")
        ]

        # Sales KPIs
        sales_kpis = [
            kpi(kpis.get('gmv'), currency=True),
            kpi(kpis.get('net_sales'), currency=True),
            kpi(kpis.get('num_orders')),
            kpi(kpis.get('aov'), currency=True),
            kpi(kpis.get('repeat_purchase_rate'), percent=True),
            kpi(kpis.get('cart_abandonment_rate'), percent=True),
            fig(kpis.get('product_performance_fig'), "Product Performance")
        ]

        # Customer KPIs
        customer_kpis = [
            kpi(customer.get('clv'), currency=True),
            kpi(customer.get('churn_rate'), percent=True),
            kpi(customer.get('active_customers')),
            kpi(customer.get('dormant_customers')),
            kpi(customer.get('new_customers')),
            kpi(customer.get('returning_customers')),
            fig(customer.get('segmentation_fig'), "Segmentation"),
            fig(customer.get('nps_fig'), "NPS")
        ]

        # Logistics KPIs
        logistics_kpis = [
            kpi(logistics.get('on_time_delivery_rate'), percent=True),
            kpi(logistics.get('avg_delivery_time')),
            kpi(logistics.get('delivery_cost_per_order'), currency=True),
            kpi(logistics.get('return_rate'), percent=True),
            fig(logistics.get('support_ticket_volume_fig'), "Support Ticket Volume"),
            fig(logistics.get('resolution_time_fig'), "Resolution Time"),
            fig(logistics.get('top_issues_fig'), "Top Issues")
        ]

        # Synthesis & Funnel KPIs
        synthesis_kpis = [
            fig(funnel.get('funnel_visualization_fig'), "Funnel Visualization"),
            fig(funnel.get('dropoff_rates_fig'), "Drop-off Rates"),
            fig(funnel.get('journey_mapping_fig'), "Journey Mapping"),
            kpi(funnel.get('clv_cac_ratio'), currency=True),
            fig(funnel.get('profitability_waterfall_fig'), "Profitability Waterfall")
        ]

        # Alerts
        if not alerts:
            alerts_section = html.Div("✅ All key metrics are healthy.", style={"color": "green"})
        else:
            alerts_section = html.Ul([html.Li(a) for a in alerts])

        return (
            *traffic_kpis,
            *marketing_kpis,
            *sales_kpis,
            *customer_kpis,
            *logistics_kpis,
            *synthesis_kpis,
            alerts_section
        )

    # Badge updater: shows metrics validator status (PASS/FAIL/running)
    @app.callback(
        Output('metrics-validator-badge', 'children'),
        Input('comp-apply-btn', 'n_clicks')
    )
    def update_validator_badge(_n_clicks):
        try:
            mv = TRANSFORMS_DATA.get('metrics_validation', {})
            # If validator hasn't run or has empty results, render nothing (very subtle)
            if not mv:
                return html.Span("", style={'display': 'none'})
            # Count failing checks where ok is False
            failed = [k for k,v in mv.items() if isinstance(v, dict) and not v.get('ok', False)]
            if not failed:
                # All good; keep hidden to reduce UI noise
                return html.Span("", style={'display': 'none'})
            # Show a small unobtrusive icon with a tooltip-like title attribute
            cnt = len(failed)
            return html.Span(
                html.I(className='bi bi-exclamation-triangle-fill', title=f"Metrics validator: {cnt} issues — click to view", style={'color':'#f0ad4e', 'fontSize':'0.95rem'}),
                style={'opacity': 0.85, 'fontSize': '0.9rem', 'paddingRight': '6px'}
            )
        except Exception:
            # Fail silently and remain unobtrusive
            return html.Span("", style={'display': 'none'})

    # Diagnostics modal toggles and content population
    @app.callback(
        Output('comp-diagnostics-modal', 'is_open'),
        [Input('comp-diagnostics-btn', 'n_clicks'), Input('comp-diagnostics-close', 'n_clicks')],
        [State('comp-diagnostics-modal', 'is_open')]
    )
    def toggle_diagnostics_modal(open_btn, close_btn, is_open):
        if open_btn or close_btn:
            return not bool(is_open)
        return bool(is_open)

    @app.callback(
        [Output('diag-metrics-validator', 'children'), Output('diag-sessions-report', 'children')],
        [Input('comp-diagnostics-btn', 'n_clicks'), Input('comp-diag-refresh', 'n_clicks')]
    )
    def populate_diagnostics(_n, _refresh):
        # Try to fetch via internal endpoints first; if unavailable, fall back to runtime store
        metrics_text = "No diagnostics available"
        sessions_text = "No sessions report available"
        try:
            import requests
            base = ''
            try:
                r = requests.get(f"{base}/api/metrics_validation", timeout=2)
                if r.status_code == 200:
                    metrics_text = r.text
            except Exception:
                pass
            try:
                r2 = requests.get(f"{base}/api/sessions_report", timeout=2)
                if r2.status_code == 200:
                    sessions_text = r2.text
            except Exception:
                pass
        except Exception:
            # Requests may not be installed or network blocked; fall back to TRANSFORMS_DATA
            try:
                mv = TRANSFORMS_DATA.get('metrics_validation', {})
                sr = TRANSFORMS_DATA.get('sessions_report', {})
                import json
                metrics_text = json.dumps(mv, indent=2, default=str)
                sessions_text = json.dumps(sr, indent=2, default=str)
            except Exception:
                metrics_text = "Diagnostics unavailable"
                sessions_text = "Sessions report unavailable"

        return metrics_text, sessions_text