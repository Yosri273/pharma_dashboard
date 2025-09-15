# KPI and Metrics Calculation for E-commerce Comprehensive Analysis Tab

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from app.utils.kpi import create_placeholder_figure
# Use shared plot styling helper from analytics helpers so figures match across tabs
from app.utils.analytics_helpers import set_dark_theme

def get_kpis(data_sources):
    # Calculate and return main KPIs (traffic, sales, retention, logistics, support)
    kpis = {}
    # Traffic & Engagement
    web = data_sources.get('web_analytics', pd.DataFrame())
    mobile = data_sources.get('mobile_analytics', pd.DataFrame())
    # total sessions is simply the number of rows across sources
    kpis['total_sessions'] = (web.shape[0] if not web.empty else 0) + (mobile.shape[0] if not mobile.empty else 0)

    # Helper to find a usable column name from variants (case-insensitive)
    def find_col(df, candidates):
        if df is None or df.empty:
            return None
        cols_lc = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in cols_lc:
                return cols_lc[cand.lower()]
        return None

    # Unique users / visitors: try multiple common names
    user_candidates = ['user_id', 'userid', 'customerid', 'client_id', 'user']
    web_user_col = find_col(web, user_candidates)
    mobile_user_col = find_col(mobile, user_candidates)
    users = set()
    if web_user_col:
        users |= set(web.get(web_user_col, pd.Series()).dropna().astype(str).unique())
    if mobile_user_col:
        users |= set(mobile.get(mobile_user_col, pd.Series()).dropna().astype(str).unique())
    kpis['unique_users'] = len(users)

    # Conversions and conversion rate: accept many variant column names
    conv_candidates = ['conversion', 'conversions', 'is_conversion', 'converted']
    web_conv_col = find_col(web, conv_candidates)
    mobile_conv_col = find_col(mobile, conv_candidates)
    total_conversions = 0
    try:
        if web_conv_col:
            total_conversions += web.get(web_conv_col, pd.Series()).astype(float).sum()
    except Exception:
        pass
    try:
        if mobile_conv_col:
            total_conversions += mobile.get(mobile_conv_col, pd.Series()).astype(float).sum()
    except Exception:
        pass
    kpis['conversion_rate'] = (float(total_conversions) / kpis['total_sessions']) if kpis['total_sessions'] else 0
    # Bounce rate, pages per session, avg session duration: robust lookups
    bounce_candidates = ['bounce', 'is_bounce', 'bounced']
    pageview_candidates = ['pageviews', 'pages', 'page_views']
    session_duration_candidates = ['session_duration', 'duration_seconds', 'duration']
    # compute combined metrics
    combined = pd.concat([web, mobile], ignore_index=True, sort=False)
    if not combined.empty:
        # bounce
        bcol = find_col(combined, bounce_candidates)
        try:
            kpis['bounce_rate'] = float(combined[bcol].mean()) if bcol and bcol in combined.columns else 0
        except Exception:
            kpis['bounce_rate'] = 0
        # pages/session
        pcol = find_col(combined, pageview_candidates)
        try:
            kpis['pages_per_session'] = float(combined[pcol].mean()) if pcol and pcol in combined.columns else 0
        except Exception:
            kpis['pages_per_session'] = 0
        # avg session duration (seconds)
        scol = find_col(combined, session_duration_candidates)
        try:
            kpis['avg_session_duration'] = float(combined[scol].mean()) if scol and scol in combined.columns else 0
        except Exception:
            kpis['avg_session_duration'] = 0
    else:
        kpis['bounce_rate'] = 0
        kpis['pages_per_session'] = 0
        kpis['avg_session_duration'] = 0

    # Acquisition & Marketing
    ad = data_sources.get('ad_platform_data', pd.DataFrame())
    kpis['total_ad_spend'] = ad['spend'].sum() if 'spend' in ad and not ad.empty else 0
    kpis['total_conversions'] = ad['conversions'].sum() if 'conversions' in ad and not ad.empty else 0
    kpis['cac'] = kpis['total_ad_spend'] / kpis['total_conversions'] if kpis['total_conversions'] else 0
    kpis['roas'] = (ad['conversions'].sum() if 'conversions' in ad and not ad.empty else 0) / kpis['total_ad_spend'] if kpis['total_ad_spend'] else 0

    # Sales & Revenue
    sales = data_sources.get('sales_data', pd.DataFrame())
    kpis['gmv'] = sales['GrossValue'].sum() if 'GrossValue' in sales and not sales.empty else 0
    kpis['net_sales'] = sales['NetSale'].sum() if 'NetSale' in sales and not sales.empty else 0
    kpis['num_orders'] = sales.shape[0] if not sales.empty else 0
    kpis['aov'] = kpis['net_sales'] / kpis['num_orders'] if kpis['num_orders'] else 0

    # Retention & Customer
    crm = data_sources.get('crm_data', pd.DataFrame())
    kpis['nps'] = crm['nps_score'].mean() if 'nps_score' in crm and not crm.empty else 0
    kpis['repeat_purchase_rate'] = (sales['CustomerID'].value_counts() > 1).mean() if 'CustomerID' in sales and not sales.empty else 0

    # Logistics
    delivery = data_sources.get('delivery_data', pd.DataFrame())
    # Example: on_time_delivery_rate: percent of Status == 'Delivered'
    kpis['on_time_delivery_rate'] = (delivery['Status'] == 'Delivered').mean() if 'Status' in delivery and not delivery.empty else 0
    # Example: avg_delivery_time: days between PromisedDate and ActualDeliveryDate
    if 'PromisedDate' in delivery and 'ActualDeliveryDate' in delivery and not delivery.empty:
        delivery['PromisedDate'] = pd.to_datetime(delivery['PromisedDate'], errors='coerce')
        delivery['ActualDeliveryDate'] = pd.to_datetime(delivery['ActualDeliveryDate'], errors='coerce')
        kpis['avg_delivery_time'] = (delivery['ActualDeliveryDate'] - delivery['PromisedDate']).dt.days.mean()
        if pd.isna(kpis['avg_delivery_time']):
            kpis['avg_delivery_time'] = 0
    else:
        kpis['avg_delivery_time'] = 0

    # Support
    support = data_sources.get('support_tickets', pd.DataFrame())
    kpis['support_ticket_volume'] = support.shape[0] if not support.empty else 0
    kpis['avg_resolution_time'] = support['resolution_time'].mean() if 'resolution_time' in support and not support.empty else 0

    # Add placeholder figures so the layout can always render a figure object
    # Top traffic sources (web + mobile combined)
    try:
        if not web.empty or not mobile.empty:
            combined = pd.concat([web, mobile], ignore_index=True, sort=False)
            src = combined.groupby('source').size().reset_index(name='sessions').sort_values('sessions', ascending=False)
            kpis['top_traffic_sources_fig'] = set_dark_theme(px.bar(src.head(10), x='source', y='sessions', title='Top Traffic Sources'))
        else:
            kpis['top_traffic_sources_fig'] = create_placeholder_figure('No traffic data')
    except Exception:
        kpis['top_traffic_sources_fig'] = create_placeholder_figure('No traffic data')

    # Product performance (top SKUs by GMV)
    try:
        if 'ProductName' in sales and not sales.empty:
            prod = sales.groupby('ProductName').agg({'GrossValue': 'sum', 'OrderID': 'count'}).rename(columns={'OrderID': 'orders'}).reset_index()
            prod = prod.sort_values('GrossValue', ascending=False).head(12)
            kpis['product_performance_fig'] = set_dark_theme(px.bar(prod, x='ProductName', y='GrossValue', title='Top Products by GMV'))
        else:
            kpis['product_performance_fig'] = create_placeholder_figure('No product data')
    except Exception:
        kpis['product_performance_fig'] = create_placeholder_figure('No product data')
    kpis.setdefault('cart_abandonment_rate', 0)
    kpis.setdefault('pages_per_session', 0)
    kpis.setdefault('bounce_rate', 0)
    kpis.setdefault('avg_session_duration', 0)

    return kpis

def get_funnel_data(data_sources):
    # Calculate funnel visualization data
    funnel = data_sources.get('funnel_data', pd.DataFrame())
    # Provide a consistent return structure with figure placeholders when data is missing
    if funnel.empty:
        return {
            'funnel_visualization_fig': create_placeholder_figure('No funnel data'),
            'dropoff_rates_fig': create_placeholder_figure('No funnel data'),
            'journey_mapping_fig': create_placeholder_figure('No funnel data'),
            'clv_cac_ratio': 0,
            'profitability_waterfall_fig': create_placeholder_figure('No funnel data')
        }

    stages = ['visit', 'add_to_cart', 'checkout', 'purchase', 'delivery']
    funnel_counts = {stage: funnel[stage].sum() if stage in funnel else None for stage in stages}
    drop_offs = {}
    for i in range(len(stages)-1):
        if funnel_counts[stages[i]] and funnel_counts[stages[i+1]]:
            drop_offs[f"{stages[i]}_to_{stages[i+1]}_dropoff"] = 1 - (funnel_counts[stages[i+1]] / funnel_counts[stages[i]])
        else:
            drop_offs[f"{stages[i]}_to_{stages[i+1]}_dropoff"] = None

    # Build funnel visualization (using summed counts across weeks)
    try:
        stages = ['Visits', 'Carts', 'Orders']
        values = [funnel[s].sum() if s in funnel else 0 for s in stages]
        df_f = pd.DataFrame({'stage': stages, 'value': values})
        funnel_fig = set_dark_theme(px.funnel(df_f, x='value', y='stage', title='End-to-end Funnel'))
    except Exception:
        funnel_fig = create_placeholder_figure('Funnel visualization error')

    # Drop-off rates
    try:
        drop_df = pd.DataFrame([{
            'stage': s,
            'dropoff': (
                1 - (funnel[stages[i+1]].sum() / funnel[stages[i]].sum())
            ) if (stages[i] in funnel and stages[i+1] in funnel and funnel[stages[i]].sum() > 0) else 0
        } for i, s in enumerate(['visit_to_add_to_cart', 'add_to_cart_to_checkout', 'checkout_to_purchase'])])
        dropoff_fig = set_dark_theme(px.bar(drop_df, x='stage', y='dropoff', title='Funnel Drop-off Rates'))
    except Exception:
        dropoff_fig = create_placeholder_figure('Drop-off rates unavailable')

    # Journey mapping (simple sankey from source -> device)
    try:
        # Attempt to combine web/mobile if available
        combined = pd.concat([data_sources.get('web_analytics', pd.DataFrame()), data_sources.get('mobile_analytics', pd.DataFrame())], ignore_index=True, sort=False)
        if not combined.empty and 'source' in combined.columns and ('device' in combined.columns or 'app_session_id' in combined.columns):
            combined['device'] = combined.get('device', combined.get('os', 'Unknown'))
            sankey_df = combined.groupby(['source', 'device']).size().reset_index(name='count')
            # build sankey
            labels = list(pd.concat([sankey_df['source'], sankey_df['device']]).unique())
            source_idx = sankey_df['source'].apply(lambda v: labels.index(v))
            target_idx = sankey_df['device'].apply(lambda v: labels.index(v))
            link = dict(source=source_idx.tolist(), target=target_idx.tolist(), value=sankey_df['count'].tolist())
            sankey_fig = go.Figure(data=[go.Sankey(node=dict(label=labels), link=link)])
            sankey_fig.update_layout(title_text='Source -> Device Journey', font_size=10)
            sankey_fig = set_dark_theme(sankey_fig)
        else:
            sankey_fig = create_placeholder_figure('Journey mapping not enough data')
    except Exception:
        sankey_fig = create_placeholder_figure('Journey mapping error')

    # Profitability waterfall: revenue - cogs - delivery - ad spend = profit
    try:
        sales = data_sources.get('sales_data', pd.DataFrame())
        deliveries = data_sources.get('delivery_data', pd.DataFrame())
        ads = data_sources.get('ad_platform_data', pd.DataFrame())
        revenue = sales['GrossValue'].sum() if 'GrossValue' in sales else 0
        cogs = sales['CostOfGoodsSold'].sum() if 'CostOfGoodsSold' in sales else 0
        delivery_cost = deliveries['DeliveryCost'].sum() if 'DeliveryCost' in deliveries else 0
        ad_spend = ads['spend'].sum() if 'spend' in ads else 0
        profit = revenue - cogs - delivery_cost - ad_spend
        wf = go.Figure(go.Waterfall(
            name='Profitability',
            orientation='v',
            measure=['relative', 'relative', 'relative', 'relative', 'total'],
            x=['Revenue', 'COGS', 'Delivery', 'Ad Spend', 'Net Profit'],
            text=[f"{revenue:,.2f}", f"-{cogs:,.2f}", f"-{delivery_cost:,.2f}", f"-{ad_spend:,.2f}", f"{profit:,.2f}"],
            y=[revenue, -cogs, -delivery_cost, -ad_spend, profit]
        ))
        wf.update_layout(title='Profitability Waterfall')
        wf = set_dark_theme(wf)
    except Exception:
        wf = create_placeholder_figure('Profitability waterfall unavailable')

    return {
        'funnel_counts': funnel_counts,
        'drop_offs': drop_offs,
        'funnel_visualization_fig': funnel_fig,
        'dropoff_rates_fig': dropoff_fig,
        'journey_mapping_fig': sankey_fig,
        'clv_cac_ratio': 0,
        'profitability_waterfall_fig': wf
    }

def get_channel_performance(data_sources):
    # Calculate channel/campaign performance metrics
    ad = data_sources.get('ad_platform_data', pd.DataFrame())
    # Return a normalized dict with summary metrics and optional records + figures
    if ad.empty:
        return {
            'records': [],
            'cac': 0,
            'roas': 0,
            'ctr': 0,
            'impressions_fig': create_placeholder_figure('No ad impressions'),
            'clicks_fig': create_placeholder_figure('No ad clicks'),
            'conversions_fig': create_placeholder_figure('No ad conversions'),
            'attribution_fig': create_placeholder_figure('No attribution data'),
            'top_campaigns_fig': create_placeholder_figure('No campaign data')
        }

    grouped = ad.groupby('platform').agg({
        'impressions': 'sum',
        'clicks': 'sum',
        'spend': 'sum',
        'conversions': 'sum'
    }).reset_index()
    grouped['ctr'] = grouped['clicks'] / grouped['impressions'].replace({0: pd.NA})
    grouped['cpc'] = grouped['spend'] / grouped['clicks'].replace({0: pd.NA})
    grouped['cpm'] = grouped['spend'] / grouped['impressions'].replace({0: pd.NA}) * 1000
    grouped['roas'] = grouped['conversions'] / grouped['spend'].replace({0: pd.NA})

    records = grouped.fillna(0).to_dict(orient='records')
    # summary metrics
    total_spend = ad['spend'].sum() if 'spend' in ad else 0
    total_conv = ad['conversions'].sum() if 'conversions' in ad else 0
    cac = (total_spend / total_conv) if total_conv else 0

    # Build figures
    try:
        impressions_fig = set_dark_theme(px.bar(grouped.sort_values('impressions', ascending=False), x='platform', y='impressions', title='Impressions by Platform'))
    except Exception:
        impressions_fig = create_placeholder_figure('Impressions by platform')
    try:
        clicks_fig = set_dark_theme(px.bar(grouped.sort_values('clicks', ascending=False), x='platform', y='clicks', title='Clicks by Platform'))
    except Exception:
        clicks_fig = create_placeholder_figure('Clicks by platform')
    try:
        conversions_fig = set_dark_theme(px.bar(grouped.sort_values('conversions', ascending=False), x='platform', y='conversions', title='Conversions by Platform'))
    except Exception:
        conversions_fig = create_placeholder_figure('Conversions by platform')

    # Top campaigns by conversions
    try:
        top_campaigns = ad.groupby('campaign_id').agg({'conversions': 'sum', 'spend': 'sum'}).reset_index()
        top_campaigns['roas'] = top_campaigns['conversions'] / top_campaigns['spend'].replace({0: pd.NA})
        top_campaigns_fig = set_dark_theme(px.bar(top_campaigns.sort_values('conversions', ascending=False).head(12), x='campaign_id', y='conversions', title='Top Campaigns by Conversions'))
    except Exception:
        top_campaigns_fig = create_placeholder_figure('Top campaigns')

    # Attribution (simple: orders per campaign if marketing_attribution exists)
    try:
        ma = data_sources.get('marketing_attribution', pd.DataFrame())
        if not ma.empty and 'campaignid' in ma.columns:
            att = ma['campaignid'].value_counts().reset_index()
            att.columns = ['campaignid', 'orders']
            attribution_fig = set_dark_theme(px.bar(att.head(12), x='campaignid', y='orders', title='Attribution: Orders by Campaign'))
        else:
            attribution_fig = create_placeholder_figure('No attribution data')
    except Exception:
        attribution_fig = create_placeholder_figure('Attribution error')

    return {
        'records': records,
        'cac': cac,
        'roas': (total_conv / total_spend) if total_spend else 0,
        'ctr': (ad['clicks'].sum() / ad['impressions'].sum()) if 'clicks' in ad and 'impressions' in ad and ad['impressions'].sum() else 0,
        'impressions_fig': impressions_fig,
        'clicks_fig': clicks_fig,
        'conversions_fig': conversions_fig,
        'attribution_fig': attribution_fig,
        'top_campaigns_fig': top_campaigns_fig
    }

def get_customer_insights(data_sources):
    # Calculate customer segmentation, CLV, churn, NPS
    sales = data_sources.get('sales_data', pd.DataFrame())
    crm = data_sources.get('crm_data', pd.DataFrame())
    insights = {}
    if not sales.empty:
        # Segmentation
        insights['new_customers'] = sales['CustomerID'].nunique() if 'CustomerID' in sales else 0
        insights['repeat_customers'] = (sales['CustomerID'].value_counts() > 1).sum() if 'CustomerID' in sales else 0
        # CLV (simple avg netsale per customer)
        if 'NetSale' in sales and 'CustomerID' in sales:
            clv = sales.groupby('CustomerID')['NetSale'].sum().mean()
            insights['avg_clv'] = clv
        # Churn (customers with no purchase in last 90 days)
        if 'Timestamp' in sales and 'CustomerID' in sales:
            sales['Timestamp'] = pd.to_datetime(sales['Timestamp'])
            last_90 = pd.Timestamp.now() - pd.Timedelta(days=90)
            active = sales[sales['Timestamp'] > last_90]['CustomerID'].unique()
            all_customers = sales['CustomerID'].unique()
            insights['churn_rate'] = 1 - (len(active) / len(all_customers)) if len(all_customers) else None
    if not crm.empty:
        insights['nps'] = crm['nps_score'].mean() if 'nps_score' in crm else None

    # Normalize keys expected by the UI
    insights.setdefault('avg_clv', insights.get('avg_clv', 0))
    insights.setdefault('clv', insights.get('avg_clv', 0))
    insights.setdefault('new_customers', insights.get('new_customers', 0))
    insights.setdefault('returning_customers', insights.get('repeat_customers', 0))
    insights.setdefault('active_customers', 0)
    insights.setdefault('dormant_customers', 0)

    # Segmentation chart: show counts per segment (from crm_data 'Segment')
    try:
        if not crm.empty and 'Segment' in crm.columns:
            seg = crm['Segment'].value_counts().reset_index()
            seg.columns = ['segment', 'count']
            insights['segmentation_fig'] = set_dark_theme(px.bar(seg, x='segment', y='count', title='Customer Segments'))
        else:
            insights['segmentation_fig'] = create_placeholder_figure('No segmentation data')
    except Exception:
        insights['segmentation_fig'] = create_placeholder_figure('Segmentation error')

    # NPS figure: trend if JoinDate exists else distribution
    try:
        if not crm.empty and 'nps_score' in crm.columns:
            if 'JoinDate' in crm.columns:
                crm['JoinDate'] = pd.to_datetime(crm['JoinDate'], errors='coerce')
                crm['nps_score'] = pd.to_numeric(crm['nps_score'], errors='coerce')
                nps_ts = crm.dropna(subset=['JoinDate', 'nps_score']).set_index('JoinDate').resample('M')['nps_score'].mean().reset_index()
                if not nps_ts.empty:
                    nps_ts['month'] = nps_ts['JoinDate'].dt.to_period('M').astype(str)
                    insights['nps_fig'] = set_dark_theme(px.line(nps_ts, x='month', y='nps_score', title='NPS Trend (by Join Date)'))
                else:
                    insights['nps_fig'] = set_dark_theme(px.histogram(crm, x='nps_score', nbins=10, title='NPS Distribution'))
            else:
                insights['nps_fig'] = set_dark_theme(px.histogram(crm, x='nps_score', nbins=10, title='NPS Distribution'))
        else:
            insights['nps_fig'] = create_placeholder_figure('No NPS data')
    except Exception:
        insights['nps_fig'] = create_placeholder_figure('NPS visualization error')

    # Cohort analysis & retention curves using sales + crm
    try:
        if not sales.empty and 'CustomerID' in sales and 'Timestamp' in sales:
            sales['Timestamp'] = pd.to_datetime(sales['Timestamp'], errors='coerce')
            # First purchase month per customer
            first_purchase = sales.sort_values('Timestamp').groupby('CustomerID')['Timestamp'].first().reset_index()
            first_purchase['cohort_month'] = first_purchase['Timestamp'].dt.to_period('M').dt.to_timestamp()
            sales = sales.merge(first_purchase[['CustomerID','cohort_month']], on='CustomerID', how='left')
            sales['order_month'] = sales['Timestamp'].dt.to_period('M').dt.to_timestamp()
            cohort_pivot = sales.groupby(['cohort_month','order_month']).agg({'OrderID':'nunique'}).reset_index()
            # Calculate retention as percent of cohort active
            cohort_sizes = cohort_pivot[cohort_pivot['cohort_month']==cohort_pivot['order_month']][['cohort_month','OrderID']].rename(columns={'OrderID':'cohort_size'})
            cohort = cohort_pivot.merge(cohort_sizes, on='cohort_month', how='left')
            cohort['retention'] = cohort['OrderID'] / cohort['cohort_size']
            # build retention table for plotting (wide)
            cohort_table = cohort.pivot_table(index='cohort_month', columns='order_month', values='retention')
            # Cohort heatmap
            try:
                cohort_fig = set_dark_theme(px.imshow(cohort_table.fillna(0).values, x=[str(c.date()) for c in cohort_table.columns], y=[str(c.date()) for c in cohort_table.index], labels=dict(x='Order Month', y='Cohort Month', color='Retention'), aspect='auto', title='Cohort Retention Heatmap'))
            except Exception:
                cohort_fig = create_placeholder_figure('Cohort visualization error')
            # Retention curve: average retention across cohorts per period
            try:
                retention_curve = cohort.groupby('order_month')['retention'].mean().reset_index()
                retention_fig = set_dark_theme(px.line(retention_curve, x='order_month', y='retention', title='Retention Curve (avg across cohorts)'))
            except Exception:
                retention_fig = create_placeholder_figure('Retention curve unavailable')
        else:
            cohort_fig = create_placeholder_figure('No cohort data')
            retention_fig = create_placeholder_figure('No retention data')
    except Exception:
        cohort_fig = create_placeholder_figure('Cohort calc error')
        retention_fig = create_placeholder_figure('Retention calc error')

    # LTV distribution (CLV by customer)
    try:
        if not sales.empty and 'CustomerID' in sales:
            if 'NetSale' in sales:
                clv_series = sales.groupby('CustomerID')['NetSale'].sum()
            elif 'GrossValue' in sales:
                clv_series = sales.groupby('CustomerID')['GrossValue'].sum()
            else:
                clv_series = pd.Series([], dtype=float)
            if not clv_series.empty:
                ltv_df = clv_series.reset_index()
                ltv_df.columns = ['CustomerID','clv']
                ltv_fig = set_dark_theme(px.histogram(ltv_df, x='clv', nbins=40, title='LTV Distribution'))
            else:
                ltv_fig = create_placeholder_figure('No LTV data')
        else:
            ltv_fig = create_placeholder_figure('No LTV data')
    except Exception:
        ltv_fig = create_placeholder_figure('LTV calculation error')

    insights.setdefault('segmentation_fig', insights.get('segmentation_fig', create_placeholder_figure('No segmentation data')))
    insights.setdefault('nps_fig', insights.get('nps_fig', create_placeholder_figure('No NPS data')))
    insights.setdefault('cohort_fig', cohort_fig)
    insights.setdefault('retention_curve_fig', retention_fig)
    insights.setdefault('ltv_distribution_fig', ltv_fig)

    return insights

def get_logistics_support(data_sources):
    # Calculate logistics and support KPIs
    delivery = data_sources.get('delivery_data', pd.DataFrame())
    support = data_sources.get('support_tickets', pd.DataFrame())
    metrics = {}
    if not delivery.empty:
        metrics['on_time_delivery_rate'] = (delivery['Status'] == 'Delivered').mean() if 'Status' in delivery else None
        if 'PromisedDate' in delivery and 'ActualDeliveryDate' in delivery:
            delivery['PromisedDate'] = pd.to_datetime(delivery['PromisedDate'], errors='coerce')
            delivery['ActualDeliveryDate'] = pd.to_datetime(delivery['ActualDeliveryDate'], errors='coerce')
            metrics['avg_delivery_time'] = (delivery['ActualDeliveryDate'] - delivery['PromisedDate']).dt.days.mean()
        else:
            metrics['avg_delivery_time'] = None
        metrics['delivery_cost_per_order'] = delivery['DeliveryCost'].mean() if 'DeliveryCost' in delivery else None
        # No return_rate column in delivery_data.csv
        metrics['return_rate'] = None
    if not support.empty:
        metrics['support_ticket_volume'] = support.shape[0]
        metrics['avg_resolution_time'] = support['resolution_time'].mean() if 'resolution_time' in support else None
        metrics['top_issues'] = support['issue_type'].value_counts().head(3).to_dict() if 'issue_type' in support else None
    # Build support figures
    try:
        if not support.empty and 'issue_type' in support:
            vol = support['issue_type'].value_counts().reset_index()
            vol.columns = ['issue', 'count']
            metrics['support_ticket_volume_fig'] = set_dark_theme(px.bar(vol, x='issue', y='count', title='Support Ticket Volume by Issue'))
        else:
            metrics['support_ticket_volume_fig'] = create_placeholder_figure('No support ticket volume data')
    except Exception:
        metrics['support_ticket_volume_fig'] = create_placeholder_figure('No support ticket volume data')

    try:
        if not support.empty and 'resolution_time' in support:
            metrics['resolution_time_fig'] = set_dark_theme(px.histogram(support, x='resolution_time', nbins=20, title='Resolution Time Distribution'))
        else:
            metrics['resolution_time_fig'] = create_placeholder_figure('No resolution time data')
    except Exception:
        metrics['resolution_time_fig'] = create_placeholder_figure('No resolution time data')

    try:
        if metrics.get('top_issues'):
            top = pd.DataFrame(list(metrics['top_issues'].items()), columns=['issue','count'])
            metrics['top_issues_fig'] = set_dark_theme(px.bar(top, x='issue', y='count', title='Top Support Issues'))
        else:
            metrics['top_issues_fig'] = create_placeholder_figure('No issues data')
    except Exception:
        metrics['top_issues_fig'] = create_placeholder_figure('No issues data')
    return metrics

def get_alerts(data_sources):
    # Generate actionable alerts and recommendations
    alerts = []
    kpis = get_kpis(data_sources)
    if kpis.get('cac', 0) > 50:
        alerts.append('Customer Acquisition Cost is high. Review marketing spend efficiency.')
    if kpis.get('on_time_delivery_rate', 1) < 0.9:
        alerts.append('On-time delivery rate is low. Investigate logistics delays.')
    if kpis.get('churn_rate', 0) > 0.3:
        alerts.append('Churn rate is high. Consider retention campaigns.')
    if kpis.get('conversion_rate', 1) < 0.05:
        alerts.append('Conversion rate is low. Optimize funnel and landing pages.')
    if kpis.get('nps', 10) < 7:
        alerts.append('NPS is below target. Address customer satisfaction issues.')
    return alerts
