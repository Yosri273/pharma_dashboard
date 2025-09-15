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
from app.analysis.recommendation_engine import generate_contextual_recommendations
from dash import html
from app.analysis.ui_helpers import render_recommendations
from app.analysis.kpi_utils import normalize_kpis


def _extract_kpi_text(kpi_comp, default="-"):
    """Safely extract the displayed text from a KPI CardBody component.

    The KPI component is usually a dbc.CardBody([html.H3(value)]) so we
    defensively walk children and return the inner string when possible.
    """
    try:
        # CardBody -> children list -> html.H3 -> children (the text)
        if hasattr(kpi_comp, 'children') and kpi_comp.children:
            first = kpi_comp.children[0]
            if hasattr(first, 'children'):
                return first.children
            return first
        # Fallback: if it's already a string/number
        return str(kpi_comp)
    except Exception:
        return default


def _extract_kpi_numeric(kpi_comp, default=0.0):
    """Try to parse a numeric value out of a KPI component or string.

    Strips thousands separators and non-numeric suffixes (like ' SAR' or 'x').
    """
    import re
    try:
        text = _extract_kpi_text(kpi_comp, default='')
        if text is None:
            return float(default)
        # Remove any non-digit/period/minus characters
        s = re.sub(r"[^0-9.\-]", "", str(text))
        return float(s) if s not in ("", "-") else float(default)
    except Exception:
        try:
            return float(kpi_comp)
        except Exception:
            return float(default)

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
                ,
                Output('marketing-recommendations-list', 'children')
        ],
    Input('marketing-apply-btn', 'n_clicks'),
        Input('marketing-refresh-recs-btn', 'n_clicks'),
    Input('marketing-thresholds-saved-signal', 'data'),
        [
            State('marketing-date-picker', 'start_date'),
            State('marketing-date-picker', 'end_date'),
            State('marketing-channel-filter', 'value'),
            State('marketing-product-filter', 'value'),
            State('marketing-rec-severity-filter', 'value')
        ]
    )
    def update_marketing_dashboard(n, refresh_click, saved_signal, sd, ed, sc, product_filter, severity_filter):
        _ = (refresh_click, saved_signal)
        product_filter = product_filter or ['All']
        analytics = generate_marketing_analytics(sd, ed, sc, product_filter)
        if analytics["is_empty"]:
            ph = create_placeholder_figure("No data")
            ek = create_kpi_body("No Data", "-")
            # Return the same number of outputs as the decorator (5 KPI bodies,
            # 4 figure placeholders, and 1 recommendations placeholder)
            return [ek] * 5 + [ph] * 4 + [html.P("No recommendations available.")]

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
        # If ETL provides spend/roas/cpa use those numeric values; otherwise
        # fall back to the pre-rendered KPI components returned by analytics.
        # Use defensive extraction when we need to parse values out of components.
        def _build_kpi_from_etl_or_comp(etl_key, comp_key, fmt, suffix=""):
            if etl_kpis and etl_kpis.get(etl_key) is not None:
                try:
                    val = float(etl_kpis.get(etl_key) or 0)
                    # Use standard thousand-separated formatting
                    formatted = "{:, .2f}".replace(' ', '')
                    formatted = f"{val:,.2f}{suffix}"
                    return create_kpi_body(fmt, formatted)
                except Exception:
                    # fallback to comp
                    pass
            # fallback: use existing component if present
            return kpis.get(comp_key)

        final_kpi_order = [
            _build_kpi_from_etl_or_comp('total_ad_spend', 'kpi_spend', "Total Ad Spend", " SAR"),
            _build_kpi_from_etl_or_comp('overall_roas', 'kpi_roas', "Overall ROAS", "x"),
            _build_kpi_from_etl_or_comp('avg_cpa', 'kpi_cpa', "Average CPA (CAC)", " SAR"),
            kpi_clv_cac,
            kpis.get("kpi_conv")
        ]
        
        figs = analytics["figures"]
        # Build simple tab insights
        tab_insights = []
        try:
            if analytics.get('kpis'):
                k = analytics['kpis']
                if 'kpi_spend' in k:
                    tab_insights.append(f"Ad spend is {_extract_kpi_text(k['kpi_spend'])}")
        except Exception:
            pass

        # Use shared normalizer to produce canonical numeric KPI dict
        try:
            from etl import transforms
            etl_kpis_num = transforms.DATA.get('kpis', {}) or {}
        except Exception:
            etl_kpis_num = {}
        kpi_values = analytics.get('kpi_values', {}) or {}
        raw_kpis = etl_kpis_num or kpi_values or analytics.get('kpis', {}) or {}
        numeric_kpis = normalize_kpis(raw_kpis)

        # Cross-context: include numeric KPIs, synthesis KPIs, and lightweight
        # attribution/campaign metadata from ETL so engine can reason across domains.
        cross_context = {
            'kpis': numeric_kpis,
            'synthesis_kpis': DATA.get('synthesis_kpis', {}),
            'marketing_campaigns': DATA.get('marketing_campaigns', {}),
            'sales_attribution': DATA.get('marketing_attribution', {}),
            'product_margins': DATA.get('product_margins', {})
        }
        rec_objs = generate_contextual_recommendations('marketing', tab_insights, cross_context)
        # Map filter value to whitelist expected by render_recommendations
        if severity_filter and str(severity_filter).lower() != 'all':
            whitelist = [str(severity_filter).lower()]
        else:
            whitelist = None
        rec_component = render_recommendations(rec_objs, accordion_id='marketing-recs-accordion', severity_whitelist=whitelist)

        # Defensive figure extraction: provide placeholders when ETL/analytics
        # didn't produce the expected figures.
        return final_kpi_order + [
            figs.get('clv_by_channel_fig', create_placeholder_figure("Not enough data for CLV by Channel")),
            figs.get('roas_fig', create_placeholder_figure("No ROAS data")),
            figs.get('cpa_fig', create_placeholder_figure("No CPA data")),
            figs.get('conv_channel_fig', create_placeholder_figure("No conversion data")),
            rec_component
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

        # Safely extract KPI text values for the PDF (defensive against varying component shapes)
        kpi_data = {}
        for k, v in analytics.get('kpis', {}).items():
            key_name = k.replace("kpi_", "").replace("_", " ").title()
            kpi_data[key_name] = _extract_kpi_text(v)
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
