import logging
from pprint import pprint

# Lower logging noise
logging.getLogger().setLevel(logging.ERROR)

from etl.transforms import DATA
from app.analysis.recommendation_engine import generate_contextual_recommendations

k = DATA.get('kpis', {}) if DATA else {}
print('KPI sample (len):', len(k))
# show a few prefixed keys
prefixed = [kk for kk in k.keys() if kk.startswith('kpi_')][:50]
print('prefixed sample (<=50):')
for kk in prefixed:
    print(' -', kk)

if not k:
    # build a small synthetic KPI sample to exercise rules
    k = {
        'roas': 0.8,
        'spend': 15000,
        'conversion_rate': 0.035,
        'aov': 18.0,
        'gross_margin': 0.18,
        'return_rate': 0.09,
        'clv_cac_ratio': 1.2,
        'avg_cpa': 250
    }
    print('\nUsing synthetic KPI sample for testing:')
    from pprint import pprint as _p
    _p(k)

recs = generate_contextual_recommendations('debug', [], {'kpis': k}, None)
print('\nGenerated recommendations count:', len(recs))
for i, r in enumerate(recs[:100], 1):
    print(f"\n=== Recommendation #{i} ===")
    pprint(r)

# If there are no recommendations, print thresholds used in last audit file entry for context
if not recs:
    try:
        import json
        with open('model_store/recommendations.jsonl', 'r') as fh:
            lines = fh.read().strip().split('\n')
            if lines:
                last = json.loads(lines[-1])
                print('\nLast audit entry thresholds_used keys:', list(last.get('thresholds_used', {}).keys())[:50])
    except Exception:
        pass
