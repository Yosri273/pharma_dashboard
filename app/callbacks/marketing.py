# -*- coding: utf-8 -*-
"""
Callbacks for the Marketing Analytics Dashboard tab.
"""
from dash import Input, Output, State, dcc
from dash.exceptions import PreventUpdate
from datetime import datetime

from etl.transforms import DATA
from app.utils.analytics_helpers import generate_marketing_analytics
from app.utils import create_placeholder_figure, create_kpi_body
from app.reporting import generate_pdf_report
import pandas as pd
from app.utils.ui_helpers import create_multi_filter_options

# Server-side typeahead limit for products
TOP_PRODUCT_OPTIONS = 200

def register_marketing_callbacks(app):
    """Registers all callbacks for the marketing dashboard."""

    @app.callback(
        [
            Output('kpi-total-ad-spend', 'children'),
            Output('kpi-avg-roas', 'children'),
            Output('kpi-avg-cpa', 'children'),
            Output('kpi-clv-cac-ratio', 'children'),
            Output('kpi-total-conversions', 'children'),
            Output('clv-by-channel-chart', 'figure'),
            Output('roas-by-campaign-chart', 'figure'),
            Output('cpa-by-campaign-chart', 'figure'),
            Output('conversions-by-channel-chart', 'figure')
        ],
        Input('marketing-apply-btn', 'n_clicks'),
        [
            State('marketing-date-picker', 'start_date'),
            State('marketing-date-picker', 'end_date'),
            State('marketing-channel-filter', 'value'),
            State('marketing-product-filter', 'value')
        ]
    )
    def update_marketing_dashboard(n, sd, ed, sc, product_filter):
        product_filter = product_filter or ['All']
        analytics = generate_marketing_analytics(sd, ed, sc, product_filter)
        if analytics["is_empty"]:
            ph = create_placeholder_figure("No data")
            ek = create_kpi_body("No Data", "-")
            return [ek] * 5 + [ph] * 4

        # Prefer ETL-provided canonical and synthesis KPIs when available
        try:
            from etl import transforms
            etl_kpis = transforms.DATA.get('kpis', {}) or {}
            synthesis_kpis = transforms.DATA.get('synthesis_kpis', {}) or {}
        except Exception:
            etl_kpis = {}
            synthesis_kpis = DATA.get('synthesis_kpis', {})

        kpi_clv_cac_val = synthesis_kpis.get('clv_cac_ratio') or etl_kpis.get('clv_cac_ratio') or 0
        kpi_clv_cac = create_kpi_body("CLV to CAC Ratio", f"{kpi_clv_cac_val:.2f}")

        kpis = analytics["kpis"]
        # If ETL provides spend/roas/cpa use formatted values; otherwise use analytics output
        final_kpi_order = [
            (create_kpi_body("Total Ad Spend", f"{etl_kpis.get('total_ad_spend', kpis.get('kpi_spend').children[0] if 'kpi_spend' in kpis else 0):,.2f} SAR") if etl_kpis else kpis["kpi_spend"]),
            (create_kpi_body("Overall ROAS", f"{etl_kpis.get('overall_roas', kpis.get('kpi_roas').children[0] if 'kpi_roas' in kpis else 0):.2f}x") if etl_kpis else kpis["kpi_roas"]),
            (create_kpi_body("Average CPA (CAC)", f"{etl_kpis.get('avg_cpa', kpis.get('kpi_cpa').children[0] if 'kpi_cpa' in kpis else 0):,.2f} SAR") if etl_kpis else kpis["kpi_cpa"]),
            kpi_clv_cac,
            kpis["kpi_conv"]
        ]
        
        figs = analytics["figures"]
        return final_kpi_order + [
            figs['clv_by_channel_fig'],
            figs['roas_fig'],
            figs['cpa_fig'],
            figs['conv_channel_fig']
        ]

    @app.callback(
        Output('download-dashboard-pdf', 'data', allow_duplicate=True),
        Input('marketing-export-btn', 'n_clicks'),
        [
            State('marketing-date-picker', 'start_date'),
            State('marketing-date-picker', 'end_date'),
            State('marketing-channel-filter', 'value'),
            State('marketing-product-filter', 'value')
        ],
        prevent_initial_call=True
    )
    def export_marketing_pdf(n, sd, ed, sc, product_filter):
        if n is None:
            raise PreventUpdate
        product_filter = product_filter or ['All']
        analytics = generate_marketing_analytics(sd, ed, sc, product_filter)
        if analytics["is_empty"]:
            raise PreventUpdate

        kpi_data = {k.replace("kpi_", "").replace("_", " ").title(): v.children[1].children for k, v in analytics["kpis"].items()}
        synthesis_kpis = DATA.get('synthesis_kpis', {})
        kpi_data["Clv To Cac Ratio"] = f"{synthesis_kpis.get('clv_cac_ratio', 0):.2f}"
        
        filter_context = {"Start Date": sd, "End Date": ed, "Channel": sc, "Products": product_filter}
        pdf_bytes = generate_pdf_report(
            kpi_data=kpi_data,
            filters_dict=filter_context,
            main_dataframe=analytics["tables"]["campaign_performance"],
            figures_list=list(analytics["figures"].values()),
            report_title="Marketing Report",
            table_title="Campaign Performance"
        )
        return dcc.send_bytes(pdf_bytes.getvalue(), f"Marketing_Report_{datetime.now().strftime('%Y%m%d')}.pdf")

    # Server-side typeahead for marketing product filter
    @app.callback(
        Output('marketing-product-filter', 'options'),
        Input('marketing-product-filter', 'search_value'),
        [
            State('marketing-date-picker', 'start_date'),
            State('marketing-date-picker', 'end_date'),
            State('marketing-channel-filter', 'value')
        ]
    )
    def update_marketing_product_options(search_value, sd, ed, channel):
        df = DATA.get('campaign_performance_df', pd.DataFrame())
        if df.empty or 'productname' not in df.columns:
            return []

        mask = pd.Series(True, index=df.index)
        if sd:
            try:
                mask &= pd.to_datetime(df['startdate']).dt.date >= pd.to_datetime(sd).date()
            except Exception:
                pass
        if ed:
            try:
                mask &= pd.to_datetime(df['enddate']).dt.date <= pd.to_datetime(ed).date()
            except Exception:
                pass
        if channel and channel != 'All':
            mask &= df['channel'] == channel

        scoped = df.loc[mask]
        products = list(pd.unique(scoped['productname']))
        if search_value:
            sv = str(search_value).lower()
            products = [p for p in products if sv in str(p).lower()]

        products = sorted(products)[:TOP_PRODUCT_OPTIONS]
        return create_multi_filter_options(products)
