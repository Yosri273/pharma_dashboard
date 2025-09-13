"""Sessions report generator utilities.

Provides a function `generate_sessions_report` that reads `web_analytics.csv` and
`mobile_analytics.csv`, computes GA-like metrics, time series and simple cohort
retention, writes reports under `reports/`, and returns a dict with results.
"""
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np


def _normalize(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def _find_col(df, candidates):
    if df is None or df.empty:
        return None
    cols_lc = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lc:
            return cols_lc[cand.lower()]
    return None


def generate_sessions_report(base_path=None, save=True):
    base = base_path or os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    web_path = os.path.join(base, 'web_analytics.csv')
    mobile_path = os.path.join(base, 'mobile_analytics.csv')
    reports_dir = os.path.join(base, 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    web = pd.read_csv(web_path) if os.path.exists(web_path) else pd.DataFrame()
    mobile = pd.read_csv(mobile_path) if os.path.exists(mobile_path) else pd.DataFrame()
    web = _normalize(web)
    mobile = _normalize(mobile)

    combined = pd.concat([web, mobile], ignore_index=True, sort=False)

    # Column candidates
    user_col = _find_col(combined, ['user_id', 'userid', 'customerid', 'client_id'])
    session_col = _find_col(combined, ['sessionid', 'session_id', 'session'])
    ts_col = _find_col(combined, ['timestamp', 'time', 'datetime', 'ts'])
    bounce_col = _find_col(combined, ['bounce', 'is_bounce', 'bounced'])
    pageviews_col = _find_col(combined, ['pageviews', 'pages', 'page_views'])
    duration_col = _find_col(combined, ['session_duration', 'duration_seconds', 'duration'])
    conv_col = _find_col(combined, ['conversion', 'conversions', 'is_conversion', 'converted'])
    source_col = _find_col(combined, ['source', 'utm_source', 'campaign_source'])
    device_col = _find_col(combined, ['device', 'platform', 'os'])

    def safe_mean(s):
        try:
            return float(s.astype(float).mean())
        except Exception:
            return 0.0

    out = {'generated_at': datetime.utcnow().isoformat()}
    out['total_sessions'] = int(combined.shape[0])
    out['unique_users'] = int(combined[user_col].nunique()) if user_col and user_col in combined.columns else 0
    out['bounce_rate'] = safe_mean(combined[bounce_col]) if bounce_col and bounce_col in combined.columns else 0.0
    out['pages_per_session'] = safe_mean(combined[pageviews_col]) if pageviews_col and pageviews_col in combined.columns else 0.0
    out['avg_session_duration_seconds'] = safe_mean(combined[duration_col]) if duration_col and duration_col in combined.columns else 0.0
    out['conversions'] = int(combined[conv_col].sum()) if conv_col and conv_col in combined.columns else 0

    # Sessions time-series (sessions per day)
    if ts_col and ts_col in combined.columns:
        try:
            combined['_ts'] = pd.to_datetime(combined[ts_col], errors='coerce')
            combined['_date'] = combined['_ts'].dt.date
            ts = combined.groupby('_date').size().reset_index(name='sessions')
            ts['date'] = ts['_date'].astype(str)
            out['sessions_by_date'] = ts[['date', 'sessions']].to_dict(orient='records')
        except Exception:
            out['sessions_by_date'] = []
    else:
        out['sessions_by_date'] = []

    # Device breakdown
    if device_col and device_col in combined.columns:
        dev = combined.groupby(device_col).agg(sessions=(session_col if session_col in combined.columns else None, 'count'), users=(user_col if user_col in combined.columns else None, lambda x: x.nunique() if user_col and user_col in combined.columns else 0)).reset_index()
        out['device_breakdown'] = dev.fillna(0).to_dict(orient='records')
    else:
        out['device_breakdown'] = []

    # New vs Returning users (first seen heuristic)
    if user_col and user_col in combined.columns and ts_col and ts_col in combined.columns:
        tmp = combined[[user_col, ts_col]].copy()
        tmp[ts_col] = pd.to_datetime(tmp[ts_col], errors='coerce')
        first_seen = tmp.groupby(user_col)[ts_col].min().reset_index().rename(columns={ts_col: 'first_seen'})
        merged = tmp.merge(first_seen, on=user_col, how='left')
        merged['is_new_session'] = merged[ts_col] == merged['first_seen']
        new_sessions = int(merged['is_new_session'].sum())
        returning_sessions = int(len(merged) - new_sessions)
        out['new_vs_returning'] = {'new_sessions': new_sessions, 'returning_sessions': returning_sessions}
    else:
        out['new_vs_returning'] = {'new_sessions': 0, 'returning_sessions': 0}

    # Simple weekly cohort retention up to 4 weeks
    cohorts = []
    try:
        if user_col and user_col in combined.columns and ts_col and ts_col in combined.columns:
            tmp = combined[[user_col, ts_col]].copy()
            tmp[ts_col] = pd.to_datetime(tmp[ts_col], errors='coerce')
            tmp['cohort_week'] = tmp[ts_col].dt.to_period('W').dt.start_time
            tmp['activity_week'] = tmp[ts_col].dt.to_period('W').dt.start_time
            cohort_pivot = tmp.groupby(['cohort_week', 'activity_week'])[user_col].nunique().reset_index()
            cohort_table = {}
            for cohort, group in cohort_pivot.groupby('cohort_week'):
                base_users = group[group['cohort_week'] == group['activity_week']][user_col].values
                base = int(base_users[0]) if len(base_users) else 0
                retention = []
                weeks = sorted(group['activity_week'].unique())
                for w in weeks[:5]:
                    val = int(group[group['activity_week'] == w][user_col].sum()) if base else 0
                    retention.append({'week': str(w.date()), 'users': val})
                cohort_table[str(cohort.date())] = {'base': base, 'retention': retention}
            out['weekly_cohorts'] = cohort_table
        else:
            out['weekly_cohorts'] = {}
    except Exception:
        out['weekly_cohorts'] = {}

    if save:
        with open(os.path.join(reports_dir, 'sessions_summary.json'), 'w') as f:
            json.dump(out, f, indent=2, default=str)
        # also dump a compact csv of sessions by date
        try:
            if out.get('sessions_by_date'):
                pd.DataFrame(out['sessions_by_date']).to_csv(os.path.join(reports_dir, 'sessions_by_date.csv'), index=False)
        except Exception:
            pass

    return out
