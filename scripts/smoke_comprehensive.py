"""Smoke test for comprehensive tab metrics.
Runs data loader and each metrics function, prints key presence and sample values.
Exits non-zero when critical KPIs or figures are missing so CI can fail fast.
"""
import sys
import pprint
import plotly.graph_objs as go
from app.comprehensive_analysis.data_sources import load_all_sources
from app.comprehensive_analysis.metrics import (
    get_kpis, get_funnel_data, get_channel_performance,
    get_customer_insights, get_logistics_support, get_alerts
)

pp = pprint.PrettyPrinter(indent=2)

def fail(msg, code=2):
    print('\nERROR:', msg)
    sys.exit(code)


if __name__ == '__main__':
    sources = load_all_sources()
    print('\nLoaded source tables:')
    for k, v in sources.items():
        print(f"- {k}: rows={len(v)} columns={list(v.columns)[:8]}")

    print('\nRunning get_kpis()...')
    kpis = get_kpis(sources)
    print('KPI keys:', sorted(kpis.keys()))

    # Critical KPI presence & non-null checks for CI
    required_kpis = [
        'total_sessions', 'unique_users', 'conversion_rate', 'gmv',
        'net_sales', 'cac', 'roas', 'on_time_delivery_rate', 'avg_delivery_time'
    ]
    missing = [k for k in required_kpis if k not in kpis or kpis.get(k) is None]
    if missing:
        fail(f'Missing or null critical KPIs: {missing}', code=3)

    for k in required_kpis:
        print(k, '=>', kpis.get(k))

    print('\nRunning get_funnel_data()...')
    funnel = get_funnel_data(sources)
    pp.pprint(funnel)

    print('\nRunning get_channel_performance()...')
    ch = get_channel_performance(sources)
    print('Type:', type(ch))
    if isinstance(ch, list):
        print('Sample record keys:', list(ch[0].keys()) if ch else 'empty')
    else:
        print(ch)

    print('\nRunning get_customer_insights()...')
    cust = get_customer_insights(sources)
    print('Keys:', sorted(cust.keys()))
    print('Sample values:')
    for k in ['new_customers', 'repeat_customers', 'clv']:
        print(k, '=>', cust.get(k))

    # Ensure critical figures exist and are Plotly figures
    required_figs = ['cohort_fig', 'retention_curve_fig', 'ltv_distribution_fig']
    missing_figs = [f for f in required_figs if f not in cust or not isinstance(cust.get(f), go.Figure)]
    if missing_figs:
        fail(f'Missing or invalid figures in customer insights: {missing_figs}', code=4)

    print('\nRunning get_logistics_support()...')
    logistics = get_logistics_support(sources)
    print('Keys:', sorted(logistics.keys()))

    print('\nRunning get_alerts()...')
    alerts = get_alerts(sources)
    print('Alerts:', alerts)

    print('\nSmoke test complete. All critical KPIs and figures present.')
    sys.exit(0)
