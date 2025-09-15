# pharma_dashboard/app/callbacks/sales.py

from dash import Input, Output, State, dcc
from dash.exceptions import PreventUpdate
from datetime import datetime

from app.utils.analytics_helpers import generate_sales_analytics
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

def register_sales_callbacks(app):
    """Registers all callbacks for the sales dashboard."""

    @app.callback(
        [
            Output('kpi_total_revenue', 'children'),
            Output('kpi_gross_margin', 'children'),
            Output('kpi_net_profit', 'children'),
            Output('kpi_total_orders', 'children'),
            Output('kpi_aov', 'children'),
            Output('kpi_return_rate', 'children'),
            Output('sales-funnel-chart', 'figure'),
            Output('sales-over-time-chart', 'figure'),
            Output('period-growth-chart', 'figure'),
            Output('price-volume-chart', 'figure'),
            Output('sales-by-category-chart', 'figure'),
            Output('top-products-chart', 'figure'),
            Output('sales-by-channel-chart', 'figure'),
            Output('sales-by-city-chart', 'figure'),
            Output('sales-by-branch-chart', 'figure'),
            Output('sales-recommendations-list', 'children')
    ],
    Input('sales-apply-btn', 'n_clicks'),
    Input('sales-refresh-recs-btn', 'n_clicks'),
    Input('sales-thresholds-saved-signal', 'data'),
        [
            State('channel-filter-dropdown', 'value'),
            State('sales-date-picker', 'start_date'),
            State('sales-date-picker', 'end_date'),
            State('time-agg-selector', 'value'),
            State('sales-region-filter', 'value'),
            State('sales-category-filter', 'value'),
            State('sales-product-filter', 'value'),
            State('sales-branch-filter', 'value')
        , State('sales-rec-severity-filter', 'value')
        ]
    )
    def update_sales_dashboard(n, refresh_click, saved_signal, sc, sd, ed, ta, sr, sca, product_filter, branch_filter, severity_filter):
        _ = (refresh_click, saved_signal)
        # product_filter and branch_filter are passed from the layout; ensure they have sensible defaults
        product_filter = product_filter or ['All']
        branch_filter = branch_filter or ['All']
        analytics = generate_sales_analytics(sc, sd, ed, ta, sr, sca, product_filter, branch_filter)
        if analytics["is_empty"]:
            ph = create_placeholder_figure("No data")
            ek = create_kpi_body("No Data", "-")
            return [ek] * 6 + [ph] * 9
        figs = analytics["figures"]

        # Prefer canonical KPIs computed by the ETL layer if available to ensure a single
        # source of truth across the app. Fall back to analytics' KPIs when ETL values
        # are missing.
        try:
            from etl import transforms
            etl_kpis = transforms.DATA.get('kpis') or {}
        except Exception:
            etl_kpis = {}

        def _format_val(val, percent=False, currency=False):
            # Handle None/NaN
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return "-"
            try:
                if percent:
                    # Accept either fraction (0.12) or percent (12.0)
                    if abs(val) <= 1:
                        return f"{val*100:.2f}%"
                    return f"{val:.2f}%"
                if currency:
                    return f"{val:,.2f} SAR"
                # default numeric formatting
                if isinstance(val, float):
                    return f"{val:,.2f}"
                return str(val)
            except Exception:
                return str(val)

        # Build KPI bodies: try ETL values first
        if etl_kpis:
            total_revenue = etl_kpis.get('net_sales') or etl_kpis.get('gmv') or 0
            net_profit = (etl_kpis.get('net_profit') if 'net_profit' in etl_kpis else None)
            total_orders = etl_kpis.get('num_orders') or etl_kpis.get('order_count') or None
            aov = etl_kpis.get('aov') or (total_revenue / total_orders if total_orders else None)
            return_kpi = etl_kpis.get('return_rate') or etl_kpis.get('return_rate_percent') or None

            kpi_list = [
                create_kpi_body("Total Revenue", _format_val(total_revenue, currency=True)),
                create_kpi_body("Gross Margin", analytics["kpis"]["kpi_margin"].children[0] if "kpi_margin" in analytics["kpis"] else create_kpi_body("Gross Margin", "-")),
                create_kpi_body("Net Profit", _format_val(net_profit, currency=True)),
                create_kpi_body("Total Orders", _format_val(total_orders)),
                create_kpi_body("Avg Order Value", _format_val(aov, currency=True)),
                create_kpi_body("Return Rate", _format_val(return_kpi, percent=True)),
            ]
        else:
            # Use analytics' already-formatted KPI card bodies
            kpi_vals = list(analytics["kpis"].values())
            # ensure exactly 6 KPI bodies
            kpi_list = kpi_vals[:6] if len(kpi_vals) >= 6 else kpi_vals + [create_kpi_body("N/A", "-")] * (6 - len(kpi_vals))

        # Build tab-level insights (lightweight, factual strings)
        tab_insights: list[str] = []
        try:
            # Example insights derived from analytics dict
            if analytics.get('kpis'):
                k = analytics['kpis']
                # Access some common KPI children safely
                if 'kpi_margin' in k:
                    tab_insights.append(f"Gross margin is {k['kpi_margin'].children[0]}")
                if 'kpi_roas' in k:
                    tab_insights.append(f"ROAS is {k['kpi_roas'].children[0]}")
        except Exception:
            pass

        # Pass deterministic KPI objects to the engine for richer rules
        try:
            from etl import transforms
            etl_kpis = transforms.DATA.get('kpis', {}) or {}
        except Exception:
            etl_kpis = {}

        # Normalize KPIs and build cross_context
        kpis_src = etl_kpis or analytics.get('kpis', {}) or {}
        normalized_kpis = normalize_kpis(kpis_src)
        cross_context = {'kpis': normalized_kpis}

        rec_objs = generate_contextual_recommendations('sales', tab_insights, cross_context)
        if severity_filter and str(severity_filter).lower() != 'all':
            whitelist = [str(severity_filter).lower()]
        else:
            whitelist = None
        rec_component = render_recommendations(rec_objs, accordion_id='sales-recs-accordion', severity_whitelist=whitelist)

        return (
            kpi_list +
            [
                figs['funnel_fig'],
                figs['sales_over_time_fig'],
                figs['period_growth_fig'],
                figs['price_volume_fig'],
                figs['sales_by_cat_fig'],
                figs['top_prod_fig'],
                figs['sales_by_channel_fig'],
                figs['sales_by_city_fig'],
                figs['sales_by_branch_fig'],
                rec_component
            ]
        )

    @app.callback(
        Output('download-dashboard-pdf', 'data', allow_duplicate=True),
        Input('sales-export-btn', 'n_clicks'),
        [
            State('channel-filter-dropdown', 'value'),
            State('sales-date-picker', 'start_date'),
            State('sales-date-picker', 'end_date'),
            State('time-agg-selector', 'value'),
            State('sales-region-filter', 'value'),
            State('sales-category-filter', 'value'),
            State('sales-product-filter', 'value'),
            State('sales-branch-filter', 'value')
        ],
        prevent_initial_call=True
    )
    def export_sales_pdf(n, sc, sd, ed, ta, sr, sca, product_filter, branch_filter):
        if n is None:
            raise PreventUpdate
        product_filter = product_filter or ['All']
        branch_filter = branch_filter or ['All']
        analytics = generate_sales_analytics(sc, sd, ed, ta, sr, sca, product_filter, branch_filter)
        if analytics["is_empty"]:
            raise PreventUpdate

        kpi_data = {k.replace("kpi_", "").replace("_", " ").title(): v.children[1].children for k, v in analytics["kpis"].items()}
        filter_context = {"Start Date": sd, "End Date": ed, "Regions": sr, "Categories": sca, "Channel": sc, "Products": product_filter, "Branches": branch_filter}
        pdf_bytes = generate_pdf_report(
            kpi_data=kpi_data,
            filters_dict=filter_context,
            main_dataframe=analytics["tables"]["top_products"],
            figures_list=list(analytics["figures"].values()),
            report_title="Sales Report",
            table_title="Top 10 Products"
        )
        return dcc.send_bytes(pdf_bytes.getvalue(), f"Sales_Report_{datetime.now().strftime('%Y%m%d')}.pdf")

    # Server-side typeahead: update product options as the user types (search_value)
    @app.callback(
        Output('sales-product-filter', 'options'),
        Input('sales-product-filter', 'search_value'),
        [
            State('sales-date-picker', 'start_date'),
            State('sales-date-picker', 'end_date'),
            State('sales-region-filter', 'value'),
            State('sales-category-filter', 'value')
        ]
    )
    def update_sales_product_options(search_value, sd, ed, regions, categories):
        # Build a scoped product list using current filters to keep the DOM small
        sales_df = DATA.get('sales', pd.DataFrame())
        if sales_df.empty or 'productname' not in sales_df.columns:
            return []

        mask = pd.Series(True, index=sales_df.index)
        if sd:
            try:
                mask &= pd.to_datetime(sales_df['date']) >= pd.to_datetime(sd)
            except Exception:
                pass
        if ed:
            try:
                mask &= pd.to_datetime(sales_df['date']) <= pd.to_datetime(ed)
            except Exception:
                pass
        if regions and 'All' not in (regions or []):
            mask &= sales_df['city'].isin(regions)
        if categories and 'All' not in (categories or []):
            mask &= sales_df['category'].isin(categories)

        scoped = sales_df.loc[mask]
        products = list(pd.unique(scoped['productname']))
        if search_value:
            sv = str(search_value).lower()
            products = [p for p in products if sv in str(p).lower()]

        products = sorted(products)[:TOP_PRODUCT_OPTIONS]
        return create_multi_filter_options(products)