from app.analysis.recommendation_engine import generate_contextual_recommendations


def test_generate_recs_empty():
    recs = generate_contextual_recommendations('sales', [], {})
    # When no numeric KPIs supplied and no cross-context numeric triggers, engine returns empty list
    assert isinstance(recs, list)
    assert len(recs) == 0


def test_cross_functional_rule_triggers():
    # campaign drives sales for low-margin products
    cross_context = {
        'marketing_campaigns': {
            'Summer Sale': {'spend': 20000, 'roas': 1.2}
        },
        'sales_attribution': {
            'Summer Sale': {'sales_volume': 500, 'top_products': ['A', 'B']}
        },
        'product_margins': {'A': 0.05, 'B': 0.08}
    }
    recs = generate_contextual_recommendations('marketing', ['Ad spend is high'], cross_context)
    assert any('reduc' in r['text'].lower() or 'reallocat' in r['text'].lower() for r in recs)
