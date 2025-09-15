"""
End-to-end metrics validator for the Comprehensive tab.
Checks, per metric:
 - required file exists
 - required columns exist
 - computes metric using canonical logic
 - reports PASS/FAIL with diagnostic info

Run: python3 scripts/validate_metrics_end_to_end.py
"""
import os
import json
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_FILES = {
    'sales': 'sales_data.csv',
    'deliveries': 'delivery_data.csv',
    'web': 'web_analytics.csv',
    'mobile': 'mobile_analytics.csv',
    'campaigns': 'marketing_campaigns.csv',
    'attrib': 'marketing_attribution.csv',
    'crm': 'crm_data.csv',
    'support': 'support_tickets.csv',
    'funnel': 'funnel_data.csv',
    'ad': 'ad_platform_data.csv',
    'preds': os.path.join('model_store','synthetic_customer_churn_predictions.csv')
}

REQUIRED_COLUMNS = {
    'unique_visitors': [('web','sessionid'), ('mobile','sessionid')],
    'unique_users': [('web','customerid'), ('mobile','customerid')],
    'bounce_rate': [('web','bounce'), ('mobile','bounce')],
    'avg_session_duration': [('web','session_duration'), ('mobile','session_duration')],
    'pages_per_session': [('web','pageviews'), ('mobile','pageviews')],
    'conversion_rate': [('web','conversion'), ('mobile','conversion')],
    'top_traffic_sources': [('web','source'), ('mobile','source')],
    'top_campaigns': [('campaigns','conversions')],
    'top_channels': [('campaigns','channel'), ('campaigns','conversions')],
    'net_sales': [('sales','total_price'), ('sales','netsale'), ('sales','grossvalue')],
    'aov': [('sales','total_price'), ('sales','netsale'), ('sales','grossvalue')],
    'cart_abandonment_rate': [('sales','cart_abandoned')],
    'clv_avg': [('crm','clv'), ('preds','Estimated_LTV')],
    'churn_rate': [('preds','churn_probability')],
    'nps': [('crm','nps_score')],
    'active_customers': [('crm','customerid')],
    'dormant_customers': [('sales','timestamp'), ('crm','customerid')],
    'return_rate': [('deliveries','is_returned')],
    'support_volume': [('support','customerid')],
    'avg_resolution_time': [('support','resolution_time')],
    'top_issues': [('support','issue_type')],
    'funnel_dropoffs': [('funnel','visit'), ('funnel','add_to_cart'), ('funnel','checkout'), ('funnel','purchase')],
    'platform_counts': [('ad','platform'), ('ad','impressions')],
    'clv_cac_ratio': [('crm','clv'), ('campaigns','totalcost'), ('campaigns','conversions')],
    'churn_with_support_vs_without': [('support','customerid'), ('preds','customerid'), ('preds','churn_probability')],
}


def load_file(key):
    path = os.path.join(BASE, DATA_FILES[key])
    if not os.path.exists(path):
        return None, path
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return f'__error__:{e}', path
    return df, path


def check_columns(df, required_cols):
    # required_cols: list of tuples like (filekey, colname)
    missing = []
    for fk, col in required_cols:
        if fk not in DATA_FILES:
            missing.append((fk, col, 'file_missing'))
            continue
        df_obj, p = load_file(fk)
        if df_obj is None:
            missing.append((fk, col, 'file_not_found'))
            continue
        if isinstance(df_obj, str) and df_obj.startswith('__error__'):
            missing.append((fk, col, 'file_read_error'))
            continue
        if col not in df_obj.columns:
            missing.append((fk, col, 'col_missing'))
    return missing


REPORT = {}

# Load all files once
FILES = {}
for k in DATA_FILES:
    df, p = load_file(k)
    FILES[k] = {'df': df, 'path': p}

# Helper to get first available column from alternatives in same file

def first_col(df, cols):
    for c in cols:
        if c in df.columns:
            return c
    return None


# Metric checks

# unique_visitors / unique_users / bounce_rate / avg_session_duration / pages_per_session / conversion_rate / top_traffic_sources
sessions = None
if FILES['web']['df'] is None and FILES['mobile']['df'] is None:
    sessions = None
else:
    parts = []
    for k in ('web','mobile'):
        if isinstance(FILES[k]['df'], pd.DataFrame):
            parts.append(FILES[k]['df'])
    sessions = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def metric_unique_visitors():
    key='unique_visitors'
    res={'ok':False,'reason':'','value':None}
    if sessions is None or sessions.empty:
        res['reason']='no_session_data'
        return res
    if 'sessionid' not in sessions.columns:
        res['reason']='sessionid_missing'
        return res
    res['value']=int(sessions['sessionid'].nunique())
    res['ok']=True
    return res


def metric_unique_users():
    key='unique_users'
    res={'ok':False,'reason':'','value':None}
    if sessions is None or sessions.empty:
        res['reason']='no_session_data'
        return res
    if 'customerid' not in sessions.columns:
        res['reason']='customerid_missing'
        return res
    res['value']=int(sessions['customerid'].nunique())
    res['ok']=True
    return res


def metric_bounce_rate():
    res={'ok':False,'reason':'','value':None}
    if sessions is None or sessions.empty:
        res['reason']='no_session_data'
        return res
    if 'bounce' not in sessions.columns:
        res['reason']='bounce_missing'
        return res
    res['value']=float(sessions['bounce'].mean())
    res['ok']=True
    return res


def metric_avg_session_duration():
    res={'ok':False,'reason':'','value':None}
    if sessions is None or sessions.empty:
        res['reason']='no_session_data'
        return res
    if 'session_duration' not in sessions.columns:
        res['reason']='session_duration_missing'
        return res
    res['value']=float(sessions['session_duration'].mean())
    res['ok']=True
    return res


def metric_pages_per_session():
    res={'ok':False,'reason':'','value':None}
    if sessions is None or sessions.empty:
        res['reason']='no_session_data'
        return res
    if 'pageviews' not in sessions.columns:
        res['reason']='pageviews_missing'
        return res
    res['value']=float(sessions['pageviews'].mean())
    res['ok']=True
    return res


def metric_conversion_rate():
    res={'ok':False,'reason':'','value':None}
    if sessions is None or sessions.empty:
        res['reason']='no_session_data'
        return res
    if 'conversion' not in sessions.columns:
        res['reason']='conversion_missing'
        return res
    total=len(sessions)
    res['value']=float(sessions['conversion'].sum()/total) if total>0 else None
    res['ok']=True
    return res


def metric_top_traffic_sources():
    res={'ok':False,'reason':'','value':None}
    if sessions is None or sessions.empty:
        res['reason']='no_session_data'
        return res
    if 'source' not in sessions.columns:
        res['reason']='source_missing'
        return res
    res['value']=sessions['source'].value_counts().head(10).to_dict()
    res['ok']=True
    return res


# top_campaigns / top_channels

def metric_top_campaigns():
    res={'ok':False,'reason':'','value':None}
    df = FILES['campaigns']['df']
    if not isinstance(df, pd.DataFrame):
        res['reason']='campaigns_missing'
        return res
    if 'conversions' not in df.columns:
        res['reason']='conversions_missing'
        return res
    res['value']=df.sort_values('conversions',ascending=False).head(5)[['campaignid','campaignname','conversions']].to_dict(orient='records')
    res['ok']=True
    return res


def metric_top_channels():
    res={'ok':False,'reason':'','value':None}
    df = FILES['campaigns']['df']
    if not isinstance(df, pd.DataFrame):
        res['reason']='campaigns_missing'
        return res
    if 'channel' not in df.columns or 'conversions' not in df.columns:
        res['reason']='channel_or_conversions_missing'
        return res
    res['value']=df.groupby('channel')['conversions'].sum().sort_values(ascending=False).to_dict()
    res['ok']=True
    return res


# net_sales / aov / cart_abandonment_rate

def metric_net_sales():
    res={'ok':False,'reason':'','value':None}
    df = FILES['sales']['df']
    if not isinstance(df, pd.DataFrame):
        res['reason']='sales_missing'
        return res
    col = first_col(df, ['total_price','netsale','grossvalue'])
    if col is None:
        res['reason']='no_amount_col'
        return res
    res['value']=float(df[col].sum())
    res['ok']=True
    return res


def metric_aov():
    res={'ok':False,'reason':'','value':None}
    df = FILES['sales']['df']
    if not isinstance(df, pd.DataFrame):
        res['reason']='sales_missing'
        return res
    col = first_col(df, ['total_price','netsale','grossvalue'])
    if col is None:
        res['reason']='no_amount_col'
        return res
    res['value']=float(df[col].mean())
    res['ok']=True
    return res


def metric_cart_abandonment_rate():
    res={'ok':False,'reason':'','value':None}
    df = FILES['sales']['df']
    if not isinstance(df, pd.DataFrame):
        res['reason']='sales_missing'
        return res
    if 'cart_abandoned' not in df.columns:
        res['reason']='cart_abandoned_missing'
        return res
    res['value']=float(df['cart_abandoned'].mean())
    res['ok']=True
    return res


# clv_avg / churn_rate / nps

def metric_clv_avg():
    res={'ok':False,'reason':'','value':None}
    crm = FILES['crm']['df']
    preds = FILES['preds']['df']
    if isinstance(crm, pd.DataFrame) and 'clv' in crm.columns:
        res['value']=float(crm['clv'].mean())
        res['ok']=True
        return res
    if isinstance(preds, pd.DataFrame) and 'Estimated_LTV' in preds.columns:
        res['value']=float(preds['Estimated_LTV'].mean())
        res['ok']=True
        return res
    res['reason']='no_clv_source'
    return res


def metric_churn_rate():
    res={'ok':False,'reason':'','value':None}
    preds=FILES['preds']['df']
    if not isinstance(preds, pd.DataFrame):
        res['reason']='preds_missing'
        return res
    if 'churn_probability' not in preds.columns:
        res['reason']='churn_probability_missing'
        return res
    res['value']=float(preds['churn_probability'].mean())
    res['ok']=True
    return res


def metric_nps():
    res={'ok':False,'reason':'','value':None}
    crm=FILES['crm']['df']
    if not isinstance(crm, pd.DataFrame):
        res['reason']='crm_missing'
        return res
    if 'nps_score' not in crm.columns:
        res['reason']='nps_missing'
        return res
    res['value']=float(crm['nps_score'].mean())
    res['ok']=True
    return res


# active_customers / dormant_customers

def metric_active_customers():
    res={'ok':False,'reason':'','value':None}
    crm=FILES['crm']['df']
    if not isinstance(crm, pd.DataFrame):
        res['reason']='crm_missing'
        return res
    if 'customerid' not in crm.columns:
        res['reason']='crm_customerid_missing'
        return res
    res['value']=int(crm['customerid'].nunique())
    res['ok']=True
    return res


def metric_dormant_customers():
    res={'ok':False,'reason':'','value':None}
    sales=FILES['sales']['df']
    crm=FILES['crm']['df']
    if not isinstance(sales, pd.DataFrame) or not isinstance(crm, pd.DataFrame):
        res['reason']='sales_or_crm_missing'
        return res
    if 'timestamp' not in sales.columns or 'customerid' not in sales.columns:
        res['reason']='sales_timestamp_or_customerid_missing'
        return res
    try:
        sales['timestamp']=pd.to_datetime(sales['timestamp'],errors='coerce')
    except:
        pass
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=180)
    recent = sales[sales['timestamp']>=cutoff]
    res['value']=int(len(set(crm['customerid']) - set(recent['customerid'])))
    res['ok']=True
    return res


# return_rate

def metric_return_rate():
    res={'ok':False,'reason':'','value':None}
    df = FILES['deliveries']['df']
    if not isinstance(df, pd.DataFrame):
        res['reason']='deliveries_missing'
        return res
    if 'is_returned' not in df.columns:
        res['reason']='is_returned_missing'
        return res
    res['value']=float(df['is_returned'].mean())
    res['ok']=True
    return res


# support metrics

def metric_support_volume():
    res={'ok':False,'reason':'','value':None}
    df = FILES['support']['df']
    if df is None or isinstance(df, str) and df.startswith('__error__'):
        res['reason']='support_missing'
        return res
    if not isinstance(df, pd.DataFrame):
        res['reason']='support_missing'
        return res
    res['value']=int(len(df))
    res['ok']=True
    return res


def metric_avg_resolution_time():
    res={'ok':False,'reason':'','value':None}
    df = FILES['support']['df']
    if not isinstance(df, pd.DataFrame):
        res['reason']='support_missing'
        return res
    if 'resolution_time' not in df.columns:
        res['reason']='resolution_time_missing'
        return res
    res['value']=float(df['resolution_time'].mean())
    res['ok']=True
    return res


def metric_top_issues():
    res={'ok':False,'reason':'','value':None}
    df = FILES['support']['df']
    if not isinstance(df, pd.DataFrame):
        res['reason']='support_missing'
        return res
    if 'issue_type' not in df.columns:
        res['reason']='issue_type_missing'
        return res
    res['value']=df['issue_type'].value_counts().head(5).to_dict()
    res['ok']=True
    return res


# funnel_dropoffs

def metric_funnel_dropoffs():
    res={'ok':False,'reason':'','value':None}
    df = FILES['funnel']['df']
    if not isinstance(df, pd.DataFrame):
        res['reason']='funnel_missing'
        return res
    needed = ['visit','add_to_cart','checkout','purchase']
    if not all(c in df.columns for c in needed):
        res['reason']='funnel_cols_missing'
        return res
    stages = df[needed].sum()
    dropoffs = {}
    for i in range(len(needed)-1):
        a,b = needed[i], needed[i+1]
        dropoffs[f"{a}_to_{b}"] = 1 - (stages[b]/stages[a]) if stages[a]>0 else np.nan
    res['value']=dropoffs
    res['ok']=True
    return res


# platform_counts

def metric_platform_counts():
    res={'ok':False,'reason':'','value':None}
    df=FILES['ad']['df']
    if not isinstance(df, pd.DataFrame):
        res['reason']='ad_missing'
        return res
    if 'platform' not in df.columns or 'impressions' not in df.columns:
        res['reason']='platform_or_impressions_missing'
        return res
    res['value']=df.groupby('platform')['impressions'].sum().to_dict()
    res['ok']=True
    return res


# clv_cac_ratio

def metric_clv_cac_ratio():
    res={'ok':False,'reason':'','value':None}
    crm=FILES['crm']['df']
    campaigns=FILES['campaigns']['df']
    if not isinstance(crm, pd.DataFrame) or not isinstance(campaigns, pd.DataFrame):
        res['reason']='crm_or_campaigns_missing'
        return res
    if 'clv' not in crm.columns or 'totalcost' not in campaigns.columns or 'conversions' not in campaigns.columns:
        res['reason']='clv_or_campaign_cols_missing'
        return res
    avg_clv=float(crm['clv'].mean())
    campaigns['cpa']=campaigns['totalcost'] / campaigns['conversions'].replace(0, np.nan)
    avg_cpa=float(campaigns['cpa'].mean())
    if avg_cpa==0 or np.isnan(avg_cpa):
        res['reason']='avg_cpa_invalid'
        return res
    res['value']=float(avg_clv/avg_cpa)
    res['ok']=True
    return res


# churn_with_support_vs_without

def metric_churn_with_support_vs_without():
    res={'ok':False,'reason':'','value':None}
    support=FILES['support']['df']
    preds=FILES['preds']['df']
    if not isinstance(preds, pd.DataFrame):
        res['reason']='preds_missing'
        return res
    if 'customerid' not in preds.columns or 'churn_probability' not in preds.columns:
        res['reason']='preds_cols_missing'
        return res
    if not isinstance(support, pd.DataFrame) or 'customerid' not in support.columns:
        res['reason']='support_missing_or_customerid'
        return res
    try:
        sup_by_cust = support.groupby('customerid').size().rename('ticket_count')
        merged = preds.set_index('customerid').join(sup_by_cust, how='left')
        avg_with = merged[merged['ticket_count']>0]['churn_probability'].mean()
        avg_without = merged[merged['ticket_count'].fillna(0)==0]['churn_probability'].mean()
        res['value']={'with_tickets': float(avg_with) if not np.isnan(avg_with) else None, 'without': float(avg_without) if not np.isnan(avg_without) else None}
        res['ok']=True
    except Exception as e:
        res['reason']=str(e)
    return res


METRICS = [
    ('unique_visitors', metric_unique_visitors), ('unique_users', metric_unique_users), ('bounce_rate', metric_bounce_rate),
    ('avg_session_duration', metric_avg_session_duration), ('pages_per_session', metric_pages_per_session), ('conversion_rate', metric_conversion_rate),
    ('top_traffic_sources', metric_top_traffic_sources), ('top_campaigns', metric_top_campaigns), ('top_channels', metric_top_channels),
    ('net_sales', metric_net_sales), ('aov', metric_aov), ('cart_abandonment_rate', metric_cart_abandonment_rate),
    ('clv_avg', metric_clv_avg), ('churn_rate', metric_churn_rate), ('nps', metric_nps), ('active_customers', metric_active_customers),
    ('dormant_customers', metric_dormant_customers), ('return_rate', metric_return_rate), ('support_volume', metric_support_volume), ('avg_resolution_time', metric_avg_resolution_time),
    ('top_issues', metric_top_issues), ('funnel_dropoffs', metric_funnel_dropoffs), ('platform_counts', metric_platform_counts), ('clv_cac_ratio', metric_clv_cac_ratio),
    ('churn_with_support_vs_without', metric_churn_with_support_vs_without)
]


def run_all():
    results = {}
    for name,fn in METRICS:
        try:
            results[name]=fn()
        except Exception as e:
            results[name]={'ok':False,'reason':f'exception:{e}','value':None}
    return results


if __name__=='__main__':
    res = run_all()
    print(json.dumps(res, indent=2, default=str))
    # Save diagnostics
    outp = os.path.join(BASE,'cache-directory','metrics_validation.json')
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp,'w') as f:
        json.dump(res,f,indent=2,default=str)
    # Update todo: mark current as completed
    print('\nDiagnostics written to', outp)
