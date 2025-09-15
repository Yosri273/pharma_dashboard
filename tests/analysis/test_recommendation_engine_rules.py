from app.analysis.recommendation_engine import generate_contextual_recommendations


def test_aov_low_triggers_info():
    cross = {'kpis': {'aov': 15}}
    recs = generate_contextual_recommendations('sales', [], cross)
    assert any('average order value' in r['text'].lower() or 'aov' in r['text'].lower() for r in recs)


def test_return_rate_critical():
    cross = {'kpis': {'return_rate': 0.16}}
    recs = generate_contextual_recommendations('sales', [], cross)
    assert any('return rate' in r['text'].lower() and r['severity'] == 'critical' for r in recs)


def test_churn_warning():
    cross = {'kpis': {'churn_rate': 0.06}}
    recs = generate_contextual_recommendations('customer', [], cross)
    assert any('churn' in r['text'].lower() for r in recs)


def test_clv_cac_critical_and_warning():
    cross = {'kpis': {'clv_cac_ratio': 0.9}}
    recs = generate_contextual_recommendations('marketing', [], cross)
    assert any('clv:cac' in r['text'].lower() or 'clv:cac' in r['text'].lower() for r in recs)


def test_campaign_scale_suggestion():
    cross = {
        'marketing_campaigns': {'TestCam': {'spend': 3000, 'roas': 3.5}},
        'sales_attribution': {'TestCam': {'sales_volume': 50}},
        'product_margins': {}
    }
    recs = generate_contextual_recommendations('marketing', [], cross)
    assert any('scale' in r['text'].lower() or 'high roas' in r['text'].lower() for r in recs)
