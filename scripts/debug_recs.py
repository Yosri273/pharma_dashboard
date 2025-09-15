from etl.transforms import DATA
from app.analysis.recommendation_engine import generate_contextual_recommendations

def main():
    k = DATA.get('kpis', {}) or {}
    print('KPI keys sample:', list(k.keys())[:50])
    print('KPI sample values (first 20):')
    for i,(kk,vv) in enumerate(k.items()):
        if i>19: break
        print(f'  {kk} -> {vv}')
    recs = generate_contextual_recommendations('debug', [], {'kpis': k}, None)
    print('\nGenerated recommendations count:', len(recs))
    for r in recs:
        print('-', r)

if __name__ == '__main__':
    main()
