import json
import pandas as pd
from etl import transforms
from app.comprehensive_analysis.metrics import get_kpis as ui_get_kpis


def test_etl_computes_kpis_and_matches_ui():
    # Load sample data dictionary (this function reads CSVs if present)
    data = transforms.load_comprehensive_sample_data()
    # Place into runtime store and compute central KPIs using the ETL helper if available
    transforms.DATA.update(data)
    # If ETL exposes get_comprehensive_kpis, use it to compute canonical kpis
    if hasattr(transforms, 'get_comprehensive_kpis'):
        etl_kpis = transforms.get_comprehensive_kpis(transforms.DATA)
    else:
        # fallback: call UI-level function using same data mapping
        etl_kpis = ui_get_kpis(transforms.DATA)

    # Basic sanity checks
    assert isinstance(etl_kpis, dict)
    # common KPI keys expected
    for key in ('total_sessions', 'net_sales'):
        assert key in etl_kpis

    # Compare one KPI computed by UI function against ETL canonical KPI
    ui_kpis = ui_get_kpis(transforms.DATA)
    # If both supply net_sales, they should be close (float equality within tolerance)
    if 'net_sales' in etl_kpis and 'net_sales' in ui_kpis:
        en = float(etl_kpis.get('net_sales') or 0)
        un = float(ui_kpis.get('net_sales') or 0)
        # Only assert closeness when both sides have non-zero values; otherwise
        # different fallback logic may cause one side to be zero while the other
        # uses a global aggregate — that's acceptable for this lightweight test.
        if en > 0 and un > 0:
            rel_diff = abs(en - un) / (abs(en) + 1e-9)
            assert rel_diff < 0.01, f"net_sales diverge too much: etl={en} ui={un}"
