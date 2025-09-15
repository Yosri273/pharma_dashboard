# -*- coding: utf-8 -*-
"""
Callbacks for the Profit Optimization Dashboard tab.
"""
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
from datetime import datetime

from app.utils.analytics_helpers import generate_profit_analytics
from app.utils import create_placeholder_figure, create_kpi_body
from app.reporting import generate_pdf_report
from etl.transforms import DATA
import pandas as pd
from app.utils.ui_helpers import create_multi_filter_options
from app.analysis.recommendation_engine import generate_contextual_recommendations
from app.analysis.ui_helpers import render_recommendations
from app.analysis.kpi_utils import normalize_kpis

# Server-side typeahead limit for products
TOP_PRODUCT_OPTIONS = 200

def register_profit_callbacks(app):
    """Registers all callbacks for the profit dashboard."""

    @app.callback(
        [
            Output('kpi-total-net-profit', 'children'),
            Output('kpi-avg-profit-margin', 'children'),
            Output('kpi-profit-lost-returns', 'children'),
            Output('profit-waterfall-chart', 'figure'),
            Output('profit-by-channel-chart', 'figure'),
            Output('profit-by-category-chart', 'figure'),
            Output('high-margin-products-chart', 'figure'),
            Output('low-margin-products-chart', 'figure'),
            Output('automated-recommendations-list', 'children')
        ],
    Input('profit-apply-btn', 'n_clicks'),
        Input('profit-refresh-recs-btn', 'n_clicks'),
    Input('profit-thresholds-saved-signal', 'data'),
        [
            State('profit-date-picker', 'start_date'),
            State('profit-date-picker', 'end_date'),
            State('profit-region-filter', 'value'),
            State('profit-category-filter', 'value'),
            State('profit-product-filter', 'value'),
        State('profit-branch-filter', 'value'),
        State('profit-rec-severity-filter', 'value')
        ]
    )
    def update_profit_dashboard(n, refresh_click, saved_signal, sd, ed, sr, sca, product_filter, branch_filter, severity_filter):
        # Either clicking Apply or Refresh Recommendations triggers recompute using current state.
        _ = (refresh_click, saved_signal)
        product_filter = product_filter or ['All']
        branch_filter = branch_filter or ['All']
        analytics = generate_profit_analytics(sd, ed, sr, sca, product_filter, branch_filter)
        if analytics["is_empty"]:
            ph = create_placeholder_figure("No data")
            ek = create_kpi_body("No Data", "-")
            return [ek] * 3 + [ph] * 5 + [html.P("No data.")]

        figs = analytics["figures"]

        try:
            from etl import transforms
            etl_kpis = transforms.DATA.get('kpis', {}) or {}
        except Exception:
            etl_kpis = {}

        if etl_kpis:
            total_net_profit = etl_kpis.get('net_profit') or None
            avg_margin = etl_kpis.get('gross_margin') or etl_kpis.get('avg_profit_margin') or None
            lost_returns = etl_kpis.get('lost_to_returns') or None

            kpi_list = [
                create_kpi_body("Total Net Profit", f"{total_net_profit:,.2f} SAR" if isinstance(total_net_profit, (int, float)) else "-"),
                create_kpi_body("Avg Profit Margin", f"{avg_margin:.2f}%" if isinstance(avg_margin, (int, float)) else "-"),
                create_kpi_body("Profit Lost to Returns", f"{lost_returns:,.2f} SAR" if isinstance(lost_returns, (int, float)) else "-"),
            ]
        else:
            kpi_list = list(analytics["kpis"].values())
        # Build cross_context and render recommendations
        # Normalize KPI dict for engine using shared helper
        # Prefer ETL raw KPIs (numeric) for recommendation engine; fallback to analytics numeric kpi_values; then UI KPIs
        kpis_src = etl_kpis or {}
        if not kpis_src:
            kpis_src = analytics.get('kpi_values', {}) or analytics.get('kpis', {}) or {}
        normalized = normalize_kpis(kpis_src)

        # Include synthesis KPIs (e.g., clv_cac_ratio) when available
        try:
            from etl import transforms as _T
            synth = _T.DATA.get('synthesis_kpis', {}) or {}
        except Exception:
            synth = {}

        cross_context = {
            'kpis': normalized,
            'product_margins': analytics.get('product_margins', {}),
            'sales_attribution': analytics.get('sales_attribution', {}),
            'marketing_campaigns': analytics.get('marketing_campaigns', {}),
            'synthesis_kpis': synth,
        }
        rec_objs = generate_contextual_recommendations('profit', ["Profit analysis available"], cross_context)
        if severity_filter and str(severity_filter).lower() != 'all':
            whitelist = [str(severity_filter).lower()]
        else:
            whitelist = None
        rec_component = render_recommendations(rec_objs, accordion_id='profit-recs-accordion', severity_whitelist=whitelist)

        return (
            kpi_list +
            [
                figs['profit_waterfall_fig'],
                figs['profit_by_channel_fig'],
                figs['profit_by_cat_fig'],
                figs['high_margin_fig'],
                figs['low_margin_fig'],
                rec_component
            ]
        )

    @app.callback(
        Output('download-dashboard-pdf', 'data', allow_duplicate=True),
        Input('profit-export-btn', 'n_clicks'),
        [
            State('profit-date-picker', 'start_date'),
            State('profit-date-picker', 'end_date'),
            State('profit-region-filter', 'value'),
            State('profit-category-filter', 'value'),
            State('profit-product-filter', 'value'),
            State('profit-branch-filter', 'value')
        ],
        prevent_initial_call=True
    )
    def export_profit_pdf(n, sd, ed, sr, sca, product_filter, branch_filter):
        if n is None:
            raise PreventUpdate
        product_filter = product_filter or ['All']
        branch_filter = branch_filter or ['All']
        analytics = generate_profit_analytics(sd, ed, sr, sca, product_filter, branch_filter)
        if analytics["is_empty"]:
            raise PreventUpdate

        kpi_data = {k.replace("kpi_", "").replace("_", " ").title(): v.children[1].children for k, v in analytics["kpis"].items()}
        filter_context = {"Start Date": sd, "End Date": ed, "Regions": sr, "Categories": sca, "Products": product_filter, "Branches": branch_filter}

        table_df = analytics["tables"]["high_margin_products"]
        table_df['profit_margin'] = table_df['profit_margin'].round(2)

        pdf_bytes = generate_pdf_report(
            kpi_data=kpi_data,
            filters_dict=filter_context,
            main_dataframe=table_df,
            figures_list=list(analytics["figures"].values()),
            report_title="Profit Report",
            table_title="Top 10 Most Profitable Products"
        )
        return dcc.send_bytes(pdf_bytes.getvalue(), f"Profit_Report_{datetime.now().strftime('%Y%m%d')}.pdf")

    # Server-side typeahead for profit product filter
    @app.callback(
        Output('profit-product-filter', 'options'),
        Input('profit-product-filter', 'search_value'),
        [
            State('profit-date-picker', 'start_date'),
            State('profit-date-picker', 'end_date'),
            State('profit-region-filter', 'value'),
            State('profit-category-filter', 'value')
        ]
    )
    def update_profit_product_options(search_value, sd, ed, regions, categories):
        df = DATA.get('profit_df', pd.DataFrame())
        if df.empty or 'productname' not in df.columns:
            return []

        mask = pd.Series(True, index=df.index)
        if sd:
            try:
                mask &= pd.to_datetime(df['date']) >= pd.to_datetime(sd)
            except Exception:
                pass
        if ed:
            try:
                mask &= pd.to_datetime(df['date']) <= pd.to_datetime(ed)
            except Exception:
                pass
        if regions and 'All' not in (regions or []):
            mask &= df['city'].isin(regions)
        if categories and 'All' not in (categories or []):
            mask &= df['category'].isin(categories)

        scoped = df.loc[mask]
        products = list(pd.unique(scoped['productname']))
        if search_value:
            sv = str(search_value).lower()
            products = [p for p in products if sv in str(p).lower()]

        products = sorted(products)[:TOP_PRODUCT_OPTIONS]
        return create_multi_filter_options(products)
