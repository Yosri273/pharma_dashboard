from typing import List, Dict, Any
from datetime import datetime, timezone
import json
import os

from .recommender import generate_recommendations as simple_recs


def _wrap_text_rec(text: str, severity: str = "info") -> Dict[str, Any]:
    return {"text": text, "severity": severity}


def _persist_recommendations(tab: str, recs: List[Dict[str, Any]], context: Dict[str, Any], thresholds: Dict[str, Any] = None):
    """Append recommendations to a JSONL file for audit/tracing.

    Each line: {ts, tab, context_snapshot, recommendations}
    """
    try:
        out_dir = os.path.join(os.getcwd(), 'model_store')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'recommendations.jsonl')
        payload = {
            'ts': datetime.now(timezone.utc).isoformat(),
            'tab': tab,
            'context': {k: (v if isinstance(v, (int, float, str, list, dict, bool)) else str(type(v))) for k, v in (context or {}).items()},
            'recs': recs,
            'thresholds_used': (thresholds or {})
        }
        with open(path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + '\n')
    except Exception:
        # Swallow persistence errors; recommendation generation should not crash the app
        pass


def generate_contextual_recommendations(tab: str, tab_insights: list[str], cross_context: Dict[str, object], thresholds: Dict[str, float] = None) -> List[Dict[str, Any]]:
    """Generate structured, context-aware recommendations.

    Returns a list of dicts: {"text": str, "severity": one of ('info','warning','critical')}

    Rules reference numeric KPI thresholds. Threshold values come from (in order of precedence):
    1) explicit `thresholds` argument passed to this function
    2) `cross_context.get('rec_thresholds')` when present
    3) per-tab on-disk override at `model_store/thresholds_{tab}.json` if present
    4) global on-disk override at `model_store/recommendation_thresholds.json` if present
    5) built-in defaults defined below
    """

    # Default thresholds (sensible defaults; editable via file or by passing `thresholds`)
    DEFAULT_THRESHOLDS = {
        # Sales / conversion
        'conversion_rate_critical': 0.02,
        'conversion_rate_warning': 0.05,
        # Margins
        'gross_margin_critical': 0.10,
        'gross_margin_warning': 0.20,
        # AOV
        'aov_info': 20.0,
        'aov_warning': 35.0,
        # Returns
        'return_rate_critical': 0.15,
        'return_rate_warning': 0.08,
        # Churn / repeat
        'churn_warning': 0.05,
        'repeat_rate_info': 0.25,
        # Logistics
        'on_time_delivery_warning': 0.90,
        'avg_delivery_time_warning_days': 5.0,
        # Marketing spend / ROAS
        'marketing_spend_high_critical': 20000.0,
        'marketing_spend_high_warning': 10000.0,
        'roas_critical': 1.0,
        'roas_warning': 1.5,
        'low_spend_scale_roas': 3.0,
        'low_spend_threshold': 5000.0,
        # CPA
        'cpa_warning': 200.0,
        # CLV:CAC
        'clv_cac_critical': 1.0,
        'clv_cac_warning': 1.5,
        # Cross-rules
        'campaign_low_margin_threshold': 0.10,
        'campaign_low_margin_ratio_warn': 0.5,
        'campaign_high_spend_threshold': 10000.0,
        # Campaign scaling rule
        'campaign_scale_spend_threshold': 5000.0,
        'campaign_scale_roas_threshold': 2.5,
        'campaign_scale_volume_threshold': 100,
    }

    # Resolve thresholds precedence (merge in increasing priority, so later sources override earlier ones)
    resolved_thresholds = dict(DEFAULT_THRESHOLDS)
    try:
        # 4) global file (lowest precedence among external sources)
        try:
            global_path = os.path.join(os.getcwd(), 'model_store', 'recommendation_thresholds.json')
            if os.path.exists(global_path):
                with open(global_path, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    resolved_thresholds.update({k: v for k, v in data.items() if v is not None})
        except Exception:
            pass

        # 3) per-tab file
        try:
            if tab:
                per_tab_path = os.path.join(os.getcwd(), 'model_store', f'thresholds_{tab}.json')
                if os.path.exists(per_tab_path):
                    with open(per_tab_path, 'r', encoding='utf-8') as fh:
                        data = json.load(fh)
                    if isinstance(data, dict):
                        resolved_thresholds.update({k: v for k, v in data.items() if v is not None})
        except Exception:
            pass

        # 2) cross_context override
        try:
            if isinstance(cross_context, dict) and cross_context.get('rec_thresholds'):
                resolved_thresholds.update({k: v for k, v in (cross_context.get('rec_thresholds') or {}).items() if v is not None})
        except Exception:
            pass

        # 1) explicit arg (highest precedence)
        if thresholds and isinstance(thresholds, dict):
            resolved_thresholds.update({k: v for k, v in thresholds.items() if v is not None})
    except Exception:
        # fallback to whatever has been resolved so far
        pass

    recs: List[Dict[str, Any]] = []

    # Prioritized KPI-driven deterministic rules (only numeric thresholds produce recommendations)
    # Prioritized KPIs: conversion_rate, net_profit, gross_margin, aov, return_rate,
    # on_time_delivery_rate, avg_delivery_time, total_ad_spend, roas, cpa, clv_cac_ratio
    try:
        raw_kpis = (cross_context or {}).get('kpis') or (cross_context or {}).get('synthesis_kpis') or {}
        # NOTE: callers may pass KPI dicts with UI-prefixed keys (e.g. 'kpi_roas')
        # or component objects. Normalize into a canonical numeric mapping here.
        def _normalize_raw_kpis(raw_kpis_in):
            out = {}
            try:
                for kk, vv in (raw_kpis_in or {}).items():
                    k = str(kk).lower()
                    if k.startswith('kpi_'):
                        k = k[4:]
                    # strip extra whitespace and replace spaces with underscores
                    k = k.replace(' ', '_')
                    # if vv is a component like a CardBody with children, try to coerce numeric
                    try:
                        if isinstance(vv, (int, float)):
                            out[k] = vv
                        else:
                            s = str(vv)
                            s_clean = s.replace('%', '').replace('x', '').replace(',', '').strip()
                            # handle empty strings
                            if s_clean == '':
                                out[k] = None
                            else:
                                out[k] = float(s_clean)
                    except Exception:
                        # keep original value when numeric coercion fails
                        out[k] = vv
            except Exception:
                pass
            return out

        raw_kpis = _normalize_raw_kpis(raw_kpis)
        # Normalize KPI keys: strip common prefixes and map short names to canonical keys
        kpis = {}
        key_map = {
            # stripped forms (after removing 'kpi_' prefix)
            'conv': 'conversion_rate', 'conv_rate': 'conversion_rate', 'conversion': 'conversion_rate',
            'conversion_rate': 'conversion_rate',
            'roas': 'roas', 'roa': 'roas', 'overall_roas': 'roas',
            'cpa': 'cpa', 'avg_cpa': 'cpa',
            'spend': 'total_ad_spend', 'ad_spend': 'total_ad_spend',
            'cpv': 'cost_per_view'
        }
        for kk, vv in (raw_kpis or {}).items():
            k = kk.lower()
            if k.startswith('kpi_'):
                k = k[4:]
            # map known short names
            mapped = key_map.get(k, None)
            if mapped:
                kpis[mapped] = vv
            else:
                # normalize certain common forms
                if k in ('conv', 'conversion'):
                    kpis['conversion_rate'] = vv
                elif k in ('roas', 'overall_roas'):
                    kpis['roas'] = vv
                elif k in ('cpa', 'avg_cpa', 'cost_per_acquisition'):
                    kpis['cpa'] = vv
                else:
                    kpis[k] = vv

        def _to_fraction(val):
            try:
                v = float(val)
                if abs(v) > 1:
                    return v / 100.0
                return v
            except Exception:
                return None

        # Track KPIs we've explicitly evaluated so generic pass doesn't repeat them
        evaluated_kpis = set()

        # conversion_rate
        conv = kpis.get('conversion_rate')
        if conv is not None:
            v = _to_fraction(conv)
            if v is not None:
                if v < resolved_thresholds.get('conversion_rate_critical', 0.02):
                    recs.append(_wrap_text_rec(f"Conversion rate is critically low (<{resolved_thresholds.get('conversion_rate_critical')*100:.2f}%). Run prioritized A/B tests on checkout and messaging.", "critical"))
                elif v < resolved_thresholds.get('conversion_rate_warning', 0.05):
                    recs.append(_wrap_text_rec(f"Conversion rate is below target ({resolved_thresholds.get('conversion_rate_warning')*100:.2f}%). Run checkout UX experiments and review traffic quality.", "warning"))
            evaluated_kpis.add('conversion_rate')

        # net_profit and gross_margin
        net_profit = kpis.get('net_profit')
        if net_profit is not None:
            try:
                if float(net_profit) < 0:
                    recs.append(_wrap_text_rec("Net profit is negative. Review top cost drivers and consider pausing loss-making promotions.", "critical"))
            except Exception:
                pass

        gross_margin = kpis.get('gross_margin') or kpis.get('avg_profit_margin')
        if gross_margin is not None:
            gm = _to_fraction(gross_margin)
            if gm is not None:
                if gm < resolved_thresholds.get('gross_margin_critical', 0.10):
                    recs.append(_wrap_text_rec(f"Gross margin is very low (<{resolved_thresholds.get('gross_margin_critical')*100:.0f}%). Audit product-level profitability and pricing.", "critical"))
                elif gm < resolved_thresholds.get('gross_margin_warning', 0.20):
                    recs.append(_wrap_text_rec(f"Gross margin is below healthy thresholds (<{resolved_thresholds.get('gross_margin_warning')*100:.0f}%). Consider price or cost optimizations.", "warning"))
            evaluated_kpis.add('gross_margin')
            evaluated_kpis.add('avg_profit_margin')

        # Average order value (AOV)
        aov = kpis.get('aov') or kpis.get('avg_order_value')
        if aov is not None:
            try:
                aov_val = float(aov)
                if aov_val < resolved_thresholds.get('aov_info', 20.0):
                    recs.append(_wrap_text_rec(f"Average Order Value is low (<{resolved_thresholds.get('aov_info')}). Test bundling, free-shipping thresholds, and upsell flows.", "info"))
                elif aov_val < resolved_thresholds.get('aov_warning', 35.0):
                    recs.append(_wrap_text_rec(f"AOV is below target (<{resolved_thresholds.get('aov_warning')}); experiment with recommended bundles and promotions to lift AOV.", "warning"))
            except Exception:
                pass

        # Return rate
        rr = kpis.get('return_rate') or kpis.get('return_rate_percent')
        if rr is not None:
            rrv = _to_fraction(rr)
            if rrv is not None:
                if rrv > resolved_thresholds.get('return_rate_critical', 0.15):
                    recs.append(_wrap_text_rec(f"Return rate is very high (>{resolved_thresholds.get('return_rate_critical')*100:.0f}%). Investigate product quality and sizing; consider tightening return policy.", "critical"))
                elif rrv > resolved_thresholds.get('return_rate_warning', 0.08):
                    recs.append(_wrap_text_rec(f"Return rate elevated (>{resolved_thresholds.get('return_rate_warning')*100:.0f}%). Review product imagery and descriptions.", "warning"))
            evaluated_kpis.add('return_rate')
            evaluated_kpis.add('return_rate_percent')

        # Churn & Repeat Purchase KPIs
        churn = kpis.get('churn_rate') or kpis.get('monthly_churn')
        repeat = kpis.get('repeat_purchase_rate') or kpis.get('repeat_rate')
        if churn is not None:
            try:
                ch = _to_fraction(churn)
                if ch is not None and ch > resolved_thresholds.get('churn_warning', 0.05):
                    recs.append(_wrap_text_rec(f"Customer churn is elevated (>{resolved_thresholds.get('churn_warning')*100:.1f}% monthly). Prioritize winback campaigns and analyze onboarding flows.", "warning"))
            except Exception:
                pass

        if repeat is not None:
            try:
                rp = _to_fraction(repeat)
                if rp is not None and rp < resolved_thresholds.get('repeat_rate_info', 0.25):
                    recs.append(_wrap_text_rec(f"Repeat purchase rate is low (<{resolved_thresholds.get('repeat_rate_info')*100:.0f}%). Implement loyalty program or post-purchase offers.", "info"))
            except Exception:
                pass
            evaluated_kpis.add('repeat_purchase_rate')
            evaluated_kpis.add('repeat_rate')

        # Logistics KPIs
        on_time = kpis.get('on_time_delivery_rate')
        if on_time is not None:
            otv = _to_fraction(on_time)
            if otv is not None and otv < resolved_thresholds.get('on_time_delivery_warning', 0.90):
                recs.append(_wrap_text_rec(f"On-time delivery rate is below {int(resolved_thresholds.get('on_time_delivery_warning')*100)}%. Review routing and driver performance.", "warning"))
            evaluated_kpis.add('on_time_delivery_rate')

        avg_delivery = kpis.get('avg_delivery_time')
        if avg_delivery is not None:
            try:
                if float(avg_delivery) > resolved_thresholds.get('avg_delivery_time_warning_days', 5.0):
                    recs.append(_wrap_text_rec(f"Average delivery time exceeds {resolved_thresholds.get('avg_delivery_time_warning_days')} days. Investigate bottlenecks and consider redistribution of fleet.", "warning"))
            except Exception:
                pass
            evaluated_kpis.add('avg_delivery_time')

        # Marketing KPIs
        spend = kpis.get('total_ad_spend')
        roas = kpis.get('overall_roas') or kpis.get('roas')
        if spend is not None and roas is not None:
            try:
                spend_val = float(spend)
                roas_val = float(roas)
                if spend_val > resolved_thresholds.get('marketing_spend_high_critical', 20000.0) and roas_val < resolved_thresholds.get('roas_critical', 1.0):
                    recs.append(_wrap_text_rec("High ad spend with ROAS below critical threshold. Pause the campaign and analyze channel efficiency.", "critical"))
                elif spend_val > resolved_thresholds.get('marketing_spend_high_warning', 10000.0) and roas_val < resolved_thresholds.get('roas_warning', 1.5):
                    recs.append(_wrap_text_rec("High ad spend with low ROAS. Reassess targeting and creatives.", "warning"))
            except Exception:
                pass
            evaluated_kpis.add('total_ad_spend')
            evaluated_kpis.add('roas')

        # Additional marketing heuristics: low spend but high ROAS may be under-invested;
        # recommend scaling when ROAS comfortably exceeds target and volume is low.
        try:
            if spend is not None and roas is not None:
                spend_val = float(spend)
                roas_val = float(roas)
                if spend_val < resolved_thresholds.get('low_spend_threshold', 5000.0) and roas_val > resolved_thresholds.get('low_spend_scale_roas', 3.0):
                    recs.append(_wrap_text_rec(f"Low ad spend with strong ROAS (>{resolved_thresholds.get('low_spend_scale_roas')}). Consider scaling high-performing campaigns gradually.", "info"))
        except Exception:
            pass

        cpa = kpis.get('avg_cpa') or kpis.get('cpa')
        if cpa is not None:
            try:
                if float(cpa) > resolved_thresholds.get('cpa_warning', 200.0):
                    recs.append(_wrap_text_rec(f"CPA is high (>{resolved_thresholds.get('cpa_warning')}). Focus on lower funnel optimizations and creative testing.", "warning"))
            except Exception:
                pass
            evaluated_kpis.add('avg_cpa')
            evaluated_kpis.add('cpa')

        clv_cac = kpis.get('clv_cac_ratio')
        if clv_cac is not None:
            try:
                if float(clv_cac) < resolved_thresholds.get('clv_cac_critical', 1.0):
                    recs.append(_wrap_text_rec("CLV:CAC below critical threshold indicates unprofitable acquisition. Revisit acquisition channels.", "critical"))
                elif float(clv_cac) < resolved_thresholds.get('clv_cac_warning', 1.5):
                    recs.append(_wrap_text_rec("CLV:CAC nearing warning threshold; prioritize retention and higher-LTV cohorts.", "warning"))
            except Exception:
                pass
            evaluated_kpis.add('clv_cac_ratio')

        # Cross-functional rule: marketing spend + sales attribution + product margins
        m = cross_context.get('marketing_campaigns', {}) if isinstance(cross_context, dict) else {}
        s = cross_context.get('sales_attribution', {}) if isinstance(cross_context, dict) else {}
        p = cross_context.get('product_margins', {}) if isinstance(cross_context, dict) else {}

        for campaign, mc in (m or {}).items():
            spend_val = mc.get('spend', 0)
            if spend_val <= 0:
                continue
            sa = (s or {}).get(campaign, {})
            top_products = sa.get('top_products', []) if sa else []
            margins = [p.get(prod) for prod in top_products if prod in (p or {})]
            low_margin_ratio = 0
            if margins:
                low_margin_ratio = sum(1 for mval in margins if (mval is not None and mval < resolved_thresholds.get('campaign_low_margin_threshold', 0.10))) / len(margins)

            if sa and sa.get('sales_volume', 0) > 0 and low_margin_ratio > resolved_thresholds.get('campaign_low_margin_ratio_warn', 0.5) and spend_val > resolved_thresholds.get('campaign_high_spend_threshold', 10000.0):
                recs.append(_wrap_text_rec(
                    f"'{campaign}' campaign drives volume but many attributed products have margins <{resolved_thresholds.get('campaign_low_margin_threshold')*100:.0f}%. Reduce spend or re-target to higher-margin SKUs.",
                    "warning"
                ))

        # Cross-rule: low volume but strong ROAS for campaign -> suggest scaling test
        for campaign, mc in (m or {}).items():
            try:
                sp = float(mc.get('spend', 0) or 0)
                r = float(mc.get('roas', 0) or 0)
                vol = int((s or {}).get(campaign, {}).get('sales_volume', 0) or 0)
                if sp < resolved_thresholds.get('campaign_scale_spend_threshold', 5000.0) and r > resolved_thresholds.get('campaign_scale_roas_threshold', 2.5) and vol < resolved_thresholds.get('campaign_scale_volume_threshold', 100):
                    recs.append(_wrap_text_rec(f"Campaign '{campaign}' shows high ROAS but low volume; run a measured scale test.", "info"))
            except Exception:
                pass
        # Generic KPI threshold rules:
        # Support thresholds with keys like: kpi_name_min, kpi_name_min_critical, kpi_name_max, kpi_name_max_critical
        try:
            for k, v in (kpis or {}).items():
                # skip KPIs we've already explicitly evaluated
                if k in evaluated_kpis:
                    continue
                # Only numeric comparisons
                try:
                    val = None
                    if isinstance(v, (int, float)):
                        val = float(v)
                    else:
                        # try converting from strings like '12.3%' or '1.23x' or '1,234.00'
                        s = str(v)
                        s_clean = s.replace('%', '').replace('x', '').replace(',', '').strip()
                        val = float(s_clean)
                except Exception:
                    continue

                # build threshold keys
                key_base = k.lower()

                # Determine direction: default heuristics or explicit direction key
                dir_key = f"{key_base}_direction"
                direction = None
                if dir_key in resolved_thresholds:
                    direction = resolved_thresholds.get(dir_key)
                else:
                    # heuristics: keywords where higher is bad
                    higher_bad_keywords = ['return', 'refund', 'churn', 'bounce', 'fraud', 'cpa', 'cost', 'spend', 'stockout', 'delay', 'time', 'ticket', 'support']
                    higher_good_keywords = ['conversion', 'aov', 'avg_order_value', 'roas', 'clv', 'margin', 'gross_margin', 'repeat', 'retention', 'active_customer', 'revenue', 'nps', 'csat', 'time_on_site', 'inventory_turnover']
                    if any(w in key_base for w in higher_bad_keywords) and not any(w in key_base for w in higher_good_keywords):
                        direction = 'higher_is_bad'
                    else:
                        direction = 'lower_is_bad'

                # severity keys using suffix _critical/_warning/_info
                crit_key = f"{key_base}_critical"
                warn_key = f"{key_base}_warning"
                info_key = f"{key_base}_info"

                # If thresholds for this KPI are fractional (<=1), and parsed val is >1,
                # adjust parsed val by dividing by 100 so '3.5%' or '3.5' (percent) parses correctly.
                # Find a representative threshold to decide scaling
                rep_thresh = None
                for tk in (crit_key, warn_key, info_key):
                    if tk in resolved_thresholds:
                        rep_thresh = resolved_thresholds.get(tk)
                        break
                if rep_thresh is not None and rep_thresh <= 1 and val is not None and val > 1:
                    try:
                        val = val / 100.0
                    except Exception:
                        pass

                # evaluate critical -> warning -> info according to direction
                try:
                    if direction == 'higher_is_bad':
                        if crit_key in resolved_thresholds and val > resolved_thresholds.get(crit_key):
                            recs.append(_wrap_text_rec(f"{k} is above critical threshold ({resolved_thresholds.get(crit_key)}). Investigate immediately.", "critical"))
                            continue
                        if warn_key in resolved_thresholds and val > resolved_thresholds.get(warn_key):
                            recs.append(_wrap_text_rec(f"{k} is above warning threshold ({resolved_thresholds.get(warn_key)}). Review drivers and consider actions.", "warning"))
                        elif info_key in resolved_thresholds and val > resolved_thresholds.get(info_key):
                            recs.append(_wrap_text_rec(f"{k} is elevated (>{resolved_thresholds.get(info_key)}). Consider related mitigations.", "info"))
                    else:
                        # lower_is_bad
                        if crit_key in resolved_thresholds and val < resolved_thresholds.get(crit_key):
                            recs.append(_wrap_text_rec(f"{k} is below critical threshold ({resolved_thresholds.get(crit_key)}). Investigate immediately.", "critical"))
                            continue
                        if warn_key in resolved_thresholds and val < resolved_thresholds.get(warn_key):
                            recs.append(_wrap_text_rec(f"{k} is below warning threshold ({resolved_thresholds.get(warn_key)}). Review drivers and consider actions.", "warning"))
                        elif info_key in resolved_thresholds and val < resolved_thresholds.get(info_key):
                            recs.append(_wrap_text_rec(f"{k} is low (<{resolved_thresholds.get(info_key)}). Consider tuning related levers.", "info"))
                except Exception:
                    # ignore comparison errors per KPI
                    pass
        except Exception:
            pass
    except Exception:
        # Fail-safe: ignore cross rules if malformed
        pass

    # Deduplicate by text while preserving order
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in recs:
        t = r.get('text')
        if t and t not in seen:
            out.append(r)
            seen.add(t)

    # Sort by severity: critical -> warning -> info
    severity_rank = {'critical': 0, 'warning': 1, 'info': 2}
    out.sort(key=lambda x: severity_rank.get(x.get('severity', 'info'), 2))

    # Persist recommendations for auditing (best-effort)
    try:
        _persist_recommendations(tab, out, cross_context, thresholds=resolved_thresholds)
    except Exception:
        pass

    return out
