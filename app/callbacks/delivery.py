# -*- coding: utf-8 -*-
"""
Callbacks for the Internal Logistics (Delivery) Dashboard tab.
"""
from dash import Input, Output, State, dcc
from dash.exceptions import PreventUpdate
from datetime import datetime

from app.utils.analytics_helpers import generate_delivery_analytics
from app.utils import create_placeholder_figure, create_kpi_body
from app.reporting import generate_pdf_report
from app.analysis.recommendation_engine import generate_contextual_recommendations
from app.analysis.ui_helpers import render_recommendations
from app.analysis.kpi_utils import normalize_kpis

def register_delivery_callbacks(app):
    """Registers all callbacks for the delivery dashboard."""

    @app.callback(
        [
            Output('kpi-on-time-delivery', 'children'),
            Output('kpi-failed-delivery', 'children'),
            Output('kpi-avg-delivery-time', 'children'),
            Output('kpi-avg-delivery-cost', 'children'),
            Output('delivery-pipeline-chart', 'figure'),
            Output('driver-leaderboard-chart', 'figure'),
            Output('vehicle-efficiency-chart', 'figure'),
            Output('avg-time-by-city-chart', 'figure'),
            Output('delivery-recommendations-list', 'children')
        ],
    Input('delivery-apply-btn', 'n_clicks'),
        Input('delivery-refresh-recs-btn', 'n_clicks'),
    Input('delivery-thresholds-saved-signal', 'data'),
        [
            State('driver-filter', 'value'),
            State('vehicle-type-filter', 'value'),
            State('delivery-date-picker', 'start_date'),
            State('delivery-date-picker', 'end_date'),
        State('delivery-region-filter', 'value'),
        State('delivery-rec-severity-filter', 'value')
        ]
    )
    def update_delivery_dashboard(n, refresh_click, saved_signal, sd, sv, start_d, end_d, sr, severity_filter):
        _ = (refresh_click, saved_signal)
        analytics = generate_delivery_analytics(sd, sv, start_d, end_d, sr)
        
        if analytics.get("error"):
            error_kpi = create_kpi_body("Error", "Schema Mismatch")
            error_fig = create_placeholder_figure(analytics["error"])
            return [error_kpi] * 4 + [error_fig] * 4

        if analytics["is_empty"]:
            ph = create_placeholder_figure("No data for filters")
            ek = create_kpi_body("No Data", "-")
            return [ek] * 4 + [ph] * 4

        figs = analytics["figures"]

        # Prefer ETL-provided canonical KPIs when available
        try:
            from etl import transforms
            etl_kpis = transforms.DATA.get('kpis', {}) or {}
        except Exception:
            etl_kpis = {}

        if etl_kpis:
            on_time = etl_kpis.get('on_time_delivery_rate') or etl_kpis.get('on_time_delivery_rate_percent') or etl_kpis.get('on_time_delivery_rate')
            failed = None  # not always computed centrally
            avg_time = etl_kpis.get('avg_delivery_time') or etl_kpis.get('avg_delivery_time_days')
            avg_cost = etl_kpis.get('delivery_cost_per_order')
            kpi_list = [
                create_kpi_body("On-Time Rate", f"{on_time:.2%}" if isinstance(on_time, float) else (str(on_time) if on_time is not None else "-")),
                create_kpi_body("Failed Delivery", "-"),
                create_kpi_body("Avg. Delivery Time", f"{avg_time:.2f} Days" if isinstance(avg_time, (int, float)) else (str(avg_time) if avg_time is not None else "-")),
                create_kpi_body("Avg Delivery Cost", f"{avg_cost:,.2f} SAR" if isinstance(avg_cost, (int, float)) else (str(avg_cost) if avg_cost is not None else "-")),
            ]
        else:
            kpi_list = list(analytics["kpis"].values())

        # Lightweight tab insights
        tab_insights = []
        try:
            if analytics.get('kpis'):
                k = analytics['kpis']
                if 'kpi_on_time' in k:
                    tab_insights.append(f"On-time rate is {k['kpi_on_time'].children[0]}")
        except Exception:
            pass

        # Normalize KPIs using shared helper
        kpis_src = analytics.get('kpis', {}) or {}
        normalized = normalize_kpis(kpis_src)
        cross_context = {'kpis': normalized}
        rec_objs = generate_contextual_recommendations('delivery', tab_insights, cross_context)
        if severity_filter and str(severity_filter).lower() != 'all':
            whitelist = [str(severity_filter).lower()]
        else:
            whitelist = None
        rec_component = render_recommendations(rec_objs, accordion_id='delivery-recs-accordion', severity_whitelist=whitelist)

        return (
            kpi_list +
            [
                figs['pipeline_fig'],
                figs['driver_leaderboard_fig'],
                figs['vehicle_efficiency_fig'],
                figs['time_by_city_fig'],
                rec_component
            ]
        )

    @app.callback(
        Output('download-dashboard-pdf', 'data', allow_duplicate=True),
        Input('delivery-export-btn', 'n_clicks'),
        [
            State('driver-filter', 'value'),
            State('vehicle-type-filter', 'value'),
            State('delivery-date-picker', 'start_date'),
            State('delivery-date-picker', 'end_date'),
            State('delivery-region-filter', 'value')
        ],
        prevent_initial_call=True
    )
    def export_delivery_pdf(n, sd, sv, start_d, end_d, sr):
        if n is None:
            raise PreventUpdate
        analytics = generate_delivery_analytics(sd, sv, start_d, end_d, sr)
        if analytics.get("error") or analytics["is_empty"]:
            raise PreventUpdate

        kpi_data = {k.replace("kpi_", "").replace("_", " ").title(): v.children[1].children for k, v in analytics["kpis"].items()}
        filter_context = {"Start Date": start_d, "End Date": end_d, "Regions": sr, "Driver": sd, "Vehicle": sv}
        pdf_bytes = generate_pdf_report(
            kpi_data=kpi_data,
            filters_dict=filter_context,
            main_dataframe=analytics["tables"]["driver_performance"],
            figures_list=list(analytics["figures"].values()),
            report_title="Internal Logistics Report",
            table_title="Driver Leaderboard"
        )
        return dcc.send_bytes(pdf_bytes.getvalue(), f"Logistics_Report_{datetime.now().strftime('%Y%m%d')}.pdf")
