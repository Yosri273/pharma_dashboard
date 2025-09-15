from app.analysis.ui_helpers import render_recommendations


def test_render_recommendations_basic():
    recs = [{'text': 'Test rec', 'severity': 'info'}]
    comp = render_recommendations(recs, accordion_id='test-acc')
    # Expect a component with id
    assert hasattr(comp, 'id')
    assert comp.id == 'test-acc'
