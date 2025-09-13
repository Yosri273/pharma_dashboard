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
            Output('avg-time-by-city-chart', 'figure')
        ],
        Input('delivery-apply-btn', 'n_clicks'),
        [
            State('driver-filter', 'value'),
            State('vehicle-type-filter', 'value'),
            State('delivery-date-picker', 'start_date'),
            State('delivery-date-picker', 'end_date'),
            State('delivery-region-filter', 'value')
        ]
    )
    def update_delivery_dashboard(n, sd, sv, start_d, end_d, sr):
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

        return (
            kpi_list +
            [
                figs['pipeline_fig'],
                figs['driver_leaderboard_fig'],
                figs['vehicle_efficiency_fig'],
                figs['time_by_city_fig']
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
