"""
Generate a sessions summary report similar to Google Analytics / Mailchimp
Outputs:
 - reports/sessions_summary.json
 - reports/sessions_by_source.csv

Usage: python3 scripts/generate_sessions_report.py
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WEB = os.path.join(BASE, 'web_analytics.csv')
MOBILE = os.path.join(BASE, 'mobile_analytics.csv')
OUT_DIR = os.path.join(BASE, 'reports')
os.makedirs(OUT_DIR, exist_ok=True)

# Read files if present
web = pd.read_csv(WEB) if os.path.exists(WEB) else pd.DataFrame()
mobile = pd.read_csv(MOBILE) if os.path.exists(MOBILE) else pd.DataFrame()

# Helper to normalize column names
def normalize_cols(df):
    if df is None or df.empty:
        return df
    df = df.copy()
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    df.columns = [c.strip() for c in df.columns]
    return df

web = normalize_cols(web)
mobile = normalize_cols(mobile)

combined = pd.concat([web, mobile], ignore_index=True, sort=False)

# Identify variants
def find_col(df, candidates):
    if df is None or df.empty:
        return None
    cols_lc = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lc:
            return cols_lc[cand.lower()]
    return None

user_col = find_col(combined, ['user_id','userid','customerid','client_id'])
session_col = find_col(combined, ['sessionid','session_id','session'])
bounce_col = find_col(combined, ['bounce','is_bounce','bounced'])
pageviews_col = find_col(combined, ['pageviews','pages','page_views'])
duration_col = find_col(combined, ['session_duration','duration_seconds','duration'])
conv_col = find_col(combined, ['conversion','conversions','is_conversion','converted'])
source_col = find_col(combined, ['source','utm_source','campaign_source'])
device_col = find_col(combined, ['device','platform','os'])

def safe_mean(series):
    try:
        return float(series.astype(float).mean())
    except Exception:
        return 0.0

report = {}
report['generated_at'] = datetime.now(timezone.utc).isoformat()
report['total_sessions'] = combined.shape[0]
report['unique_users'] = int(combined[user_col].nunique()) if user_col and user_col in combined.columns else 0
report['bounce_rate'] = safe_mean(combined[bounce_col]) if bounce_col else 0.0
report['pages_per_session'] = safe_mean(combined[pageviews_col]) if pageviews_col else 0.0
report['avg_session_duration_seconds'] = safe_mean(combined[duration_col]) if duration_col else 0.0
report['conversions'] = int(combined[conv_col].sum()) if conv_col and conv_col in combined.columns else 0

# Aggregate by source
if source_col and source_col in combined.columns:
    by_source = combined.groupby(source_col).agg(
        sessions=('sessionid' if 'sessionid' in combined.columns else session_col, 'count'),
        users=(user_col if user_col in combined.columns else session_col, lambda x: x.nunique() if user_col and user_col in combined.columns else x.count()),
        bounce_rate=(bounce_col if bounce_col in combined.columns else session_col, lambda x: float(x.mean()) if bounce_col in combined.columns else 0),
        pages_per_session=(pageviews_col if pageviews_col in combined.columns else session_col, lambda x: float(x.mean()) if pageviews_col in combined.columns else 0)
    ).reset_index()
    by_source.to_csv(os.path.join(OUT_DIR, 'sessions_by_source.csv'), index=False)
    report['by_source'] = by_source.to_dict(orient='records')
else:
    report['by_source'] = []

# Save JSON
with open(os.path.join(OUT_DIR, 'sessions_summary.json'), 'w') as f:
    json.dump(report, f, indent=2)

print('Wrote reports to', OUT_DIR)
print(json.dumps(report, indent=2))
