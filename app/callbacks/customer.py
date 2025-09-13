# -*- coding: utf-8 -*-
"""
Callbacks for the Customer Analytics Dashboard tab.
"""
import pandas as pd
from dash import Input, Output, State, dcc
from dash.exceptions import PreventUpdate
from datetime import datetime

from etl.transforms import DATA
from app.utils.analytics_helpers import generate_customer_analytics
from app.utils import create_placeholder_figure, create_kpi_body
from app.reporting import generate_pdf_report

def register_customer_callbacks(app):
    """Registers all callbacks for the customer dashboard."""

    @app.callback(
        [
            Output('kpi-total-customers', 'children'),
            Output('kpi-active-customers', 'children'),
            Output('kpi-retention-rate', 'children'),
            Output('kpi-repeat-purchase-rate', 'children'),
            Output('kpi-dormant-customers', 'children'),
            Output('kpi-churn-risk', 'children'),
            Output('customer-status-dist-chart', 'figure'),
            Output('rfm-bubble-chart', 'figure'),
            Output('customer-data-table', 'data'),
            Output('customer-data-table', 'columns')
        ],
        Input('customer-apply-btn', 'n_clicks'),
        [
            State('customer-list-selector', 'value'),
            State('customer-date-picker', 'start_date'),
            State('customer-date-picker', 'end_date'),
            State('customer-region-filter', 'value'),
            State('customer-segment-filter', 'value')
        ]
    )
    def update_customer_dashboard(n, sl, sd, ed, sr, ss):
        analytics = generate_customer_analytics(sl, sd, ed, sr, ss)
        if analytics["is_empty"]:
            ph = create_placeholder_figure("No data")
            ek = create_kpi_body("No Data", "-")
            return [ek] * 6 + [ph] * 2 + [[], []]

        # Prefer canonical KPIs from ETL runtime store when available
        try:
            from etl import transforms
            etl_kpis = transforms.DATA.get('kpis', {}) or {}
            synthesis_kpis = transforms.DATA.get('synthesis_kpis', {}) or DATA.get('synthesis_kpis', {})
        except Exception:
            etl_kpis = {}
            synthesis_kpis = DATA.get('synthesis_kpis', {})

        kpi_retention = create_kpi_body("Retention Rate", f"{(synthesis_kpis.get('retention_rate') or etl_kpis.get('retention_rate') or 0):.1f}%")
        kpi_repeat = create_kpi_body("Repeat Purchase Rate", f"{(synthesis_kpis.get('repeat_purchase_rate') or etl_kpis.get('repeat_purchase_rate') or 0):.1f}%")

        table_df = analytics["tables"]["customer_list"]
        data = table_df.to_dict('records')
        columns = [{"name": i, "id": i} for i in table_df.columns]
        
        kpis = analytics['kpis']
        # If ETL provides authoritative counts, use them for total/active/dormant
        if etl_kpis:
            total_customers = etl_kpis.get('active_customers') or etl_kpis.get('total_customers') or None
            active_customers = etl_kpis.get('active_customers') or None
            dormant_customers = etl_kpis.get('dormant_customers') or None
            kpi_total = create_kpi_body("Total Customers", f"{total_customers:,}" if total_customers is not None else "-")
            kpi_active = create_kpi_body("Active Customers", f"{active_customers:,}" if active_customers is not None else "-")
            kpi_dormant = create_kpi_body("Dormant Customers", f"{dormant_customers:,}" if dormant_customers is not None else "-")
        else:
            kpi_total = kpis['kpi_total']
            kpi_active = kpis['kpi_active']
            kpi_dormant = kpis['kpi_dormant']

        return [
            kpi_total, kpi_active, kpi_retention, kpi_repeat,
            kpi_dormant, kpis['kpi_churn'],
            analytics['figures']['status_dist_fig'], analytics['figures']['rfm_bubble_fig'],
            data, columns
        ]

    @app.callback(
        Output('download-dashboard-pdf', 'data', allow_duplicate=True),
        Input('customer-export-btn', 'n_clicks'),
        [
            State('customer-list-selector', 'value'),
            State('customer-date-picker', 'start_date'),
            State('customer-date-picker', 'end_date'),
            State('customer-region-filter', 'value'),
            State('customer-segment-filter', 'value')
        ],
        prevent_initial_call=True
    )
    def export_customer_pdf(n, sl, sd, ed, sr, ss):
        if n is None:
            raise PreventUpdate
        analytics = generate_customer_analytics(sl, sd, ed, sr, ss)
        if analytics["is_empty"]:
            raise PreventUpdate

        kpi_data = {k.replace("kpi_", "").replace("_", " ").title(): v.children[1].children for k, v in analytics["kpis"].items()}
        filter_context = {"Join Date Start": sd, "Join Date End": ed, "Regions": sr, "Segments": ss, "List": analytics["selected_list_title"]}
        
        table_df = analytics["tables"]["customer_list"]
        if 'joindate' in table_df.columns:
            table_df['joindate'] = table_df['joindate'].dt.strftime('%Y-%m-%d')
        if 'last_purchase_date' in table_df.columns:
            table_df['last_purchase_date'] = table_df['last_purchase_date'].dt.strftime('%Y-%m-%d')
            
        pdf_bytes = generate_pdf_report(
            kpi_data=kpi_data,
            filters_dict=filter_context,
            main_dataframe=table_df,
            figures_list=[analytics["figures"]["status_dist_fig"]],
            report_title="Customer Report",
            table_title=f"Customer List: {analytics['selected_list_title']}"
        )
        return dcc.send_bytes(pdf_bytes.getvalue(), f"Customer_Report_{datetime.now().strftime('%Y%m%d')}.pdf")
        
    @app.callback(
        Output("download-dataframe-csv", "data"),
        Input("export-csv-button", "n_clicks"),
        State("customer-list-selector", "value"),
        prevent_initial_call=True
    )
    def export_data_as_csv(n, sl):
        if n is None:
            raise PreventUpdate
        
        customer_analysis_df = DATA.get('customer_analysis_df', pd.DataFrame())
        if customer_analysis_df.empty:
            raise PreventUpdate
            
        if sl == 'top_value':
            df_to_export = customer_analysis_df.sort_values('monetary', ascending=False)
        elif sl == 'churn_risk':
            df_to_export = customer_analysis_df[customer_analysis_df['status'] == 'Churn Risk']
        elif sl == 'new':
            df_to_export = customer_analysis_df[customer_analysis_df['status'] == 'New']
        else:
            raise PreventUpdate

        if not df_to_export.empty:
            return dcc.send_data_frame(df_to_export.to_csv, f"{sl}_customers_{datetime.now().strftime('%Y-%m-%d')}.csv", index=False)
        
        raise PreventUpdate
