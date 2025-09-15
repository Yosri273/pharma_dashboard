from app.callbacks.marketing import _extract_kpi_text, _extract_kpi_numeric
from dash import html


def test_extract_kpi_text_from_component():
    comp = html.Div([html.H3("12,345.67 SAR")])
    assert _extract_kpi_text(comp) == "12,345.67 SAR"


def test_extract_kpi_numeric_parsing():
    comp = html.Div([html.H3("12,345.67 SAR")])
    val = _extract_kpi_numeric(comp)
    assert abs(val - 12345.67) < 0.001


def test_extract_kpi_numeric_from_string():
    assert abs(_extract_kpi_numeric("1,234.00 SAR") - 1234.00) < 0.001
