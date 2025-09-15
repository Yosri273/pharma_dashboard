"""Shared KPI normalization helpers used by tab callbacks and the recommendation engine.

Exports:
 - normalize_kpis(raw_kpis) -> dict of canonical numeric KPIs
 - coerce_threshold_value(v) -> int/float/None
"""
from typing import Dict, Any

def _extract_text_from_component(comp):
    try:
        # dash html components often have .children
        ch = getattr(comp, 'children', None)
        if ch is None:
            return str(comp)
        # children may be list or scalar
        if isinstance(ch, (list, tuple)):
            if not ch:
                return ''
            # first child's children commonly holds the value
            first = ch[0]
            return _extract_text_from_component(first)
        return str(ch)
    except Exception:
        return str(comp)


def _coerce_num(v):
    if v is None or v == '':
        return None
    try:
        if isinstance(v, (int, float)):
            return v
        # unwrap dash component text if needed
        s = _extract_text_from_component(v)
        s_clean = s.replace('%', '').replace('x', '').replace(',', '').strip()
        if s_clean == '':
            return None
        f = float(s_clean)
        # prefer int when integer-valued
        if f.is_integer():
            return int(f)
        return f
    except Exception:
        return None


def normalize_kpis(raw_kpis: Dict[str, Any]) -> Dict[str, float]:
    """Normalize KPI dict keys and coerce numeric values.

    - strips 'kpi_' prefix
    - maps short names to canonical names
    - coerces numeric strings like '12.3%', '1,234', '2.5x'
    - returns only keys with numeric values (int/float)
    """
    key_map = {
        'conv': 'conversion_rate', 'conv_rate': 'conversion_rate', 'conversion': 'conversion_rate', 'conversion_rate': 'conversion_rate',
        'roas': 'roas', 'roa': 'roas', 'overall_roas': 'roas',
        'cpa': 'cpa', 'avg_cpa': 'cpa',
        'spend': 'total_ad_spend', 'ad_spend': 'total_ad_spend',
        'aov': 'aov', 'avg_order_value': 'aov',
        'gross_margin': 'gross_margin', 'margin': 'gross_margin',
        'return_rate': 'return_rate', 'returns': 'return_rate',
        'clv_cac_ratio': 'clv_cac_ratio', 'clv_cac': 'clv_cac_ratio',
        'net_profit': 'net_profit', 'avg_delivery_time': 'avg_delivery_time', 'on_time_delivery_rate': 'on_time_delivery_rate'
    }

    out = {}
    if not raw_kpis:
        return out
    for kk, vv in (raw_kpis or {}).items():
        try:
            k = str(kk).lower()
            if k.startswith('kpi_'):
                k = k[4:]
            k = k.replace(' ', '_')
            mapped = key_map.get(k, k)
            num = _coerce_num(vv)
            if num is not None:
                out[mapped] = num
        except Exception:
            continue
    return out


def coerce_threshold_value(v):
    """Coerce threshold input into int/float or return None for empty/invalid."""
    return _coerce_num(v)
