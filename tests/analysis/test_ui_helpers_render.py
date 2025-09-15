from app.analysis.ui_helpers import render_recommendations
import dash_bootstrap_components as dbc


def test_render_orders_by_severity_and_handles_missing_text():
    recs = [
        {"text": "Low priority note", "severity": "info"},
        {"text": "Urgent: fix payments", "severity": "critical"},
        {"text": "Check returns", "severity": "warning"},
        {},  # missing text
        None,
        "A plain string recommendation"
    ]

    comp = render_recommendations(recs, accordion_id='test-acc')
    # Should return a dbc.Accordion when there are valid items
    assert isinstance(comp, dbc.Accordion)
    # Items should be ordered with critical first then warning then info
    titles = [it.title for it in comp.children]
    assert any('Urgent' in t for t in titles)


def test_render_returns_placeholder_on_empty():
    comp = render_recommendations([], accordion_id='empty-acc')
    # Placeholder card when empty
    assert isinstance(comp, dbc.Card)
