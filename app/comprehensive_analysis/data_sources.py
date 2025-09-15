# Data Loading and Merging for E-commerce Comprehensive Analysis Tab

import pandas as pd
import os

def load_all_sources():
    # Load all required tables from the database and return as dict of DataFrames
    from services.db import get_engine
    engine = get_engine()
    table_names = [
        'web_analytics', 'mobile_analytics', 'ad_platform_data', 'support_tickets',
        'crm_data', 'sales_data', 'customer_data', 'delivery_data',
        'marketing_campaigns', 'marketing_attribution', 'funnel_data', 'competitor_data'
    ]
    data = {}
    for table in table_names:
        try:
            data[table] = pd.read_sql_table(table, engine)
        except Exception:
            data[table] = pd.DataFrame()
    return data

def merge_sources(sources):
    # Merge and connect data creatively for analysis
    merged = {}

    # Example: Attribution - connect web/mobile analytics to sales via user_id
    if not sources['sales_data'].empty and not sources['web_analytics'].empty:
        merged['web_sales'] = sources['sales_data'].merge(
            sources['web_analytics'], left_on='CustomerID', right_on='user_id', how='left')

    if not sources['sales_data'].empty and not sources['mobile_analytics'].empty:
        merged['mobile_sales'] = sources['sales_data'].merge(
            sources['mobile_analytics'], left_on='CustomerID', right_on='user_id', how='left')

    # Attribution: connect ad platform data to sales via campaign_id
    if not sources['sales_data'].empty and not sources['ad_platform_data'].empty:
        sales = sources['sales_data'].copy()
        ad = sources['ad_platform_data'].copy()
        # Use CampaignID if available, else Channel/platform
        if 'CampaignID' in sales.columns and 'campaign_id' in ad.columns:
            sales['CampaignID'] = sales['CampaignID'].astype(str)
            ad['campaign_id'] = ad['campaign_id'].astype(str)
            merged['ad_sales'] = sales.merge(ad, left_on='CampaignID', right_on='campaign_id', how='left')
        elif 'Channel' in sales.columns and 'platform' in ad.columns:
            merged['ad_sales'] = sales.merge(ad, left_on='Channel', right_on='platform', how='left')

    # Customer journey: combine funnel, sales, and analytics
    # Funnel data does not have user_id, so merge on Week if possible
    if not sources['funnel_data'].empty and not sources['sales_data'].empty:
        if 'Week' in sources['funnel_data'].columns and 'Timestamp' in sources['sales_data'].columns:
            # Extract week from Timestamp in sales_data
            sales = sources['sales_data'].copy()
            sales['Week'] = pd.to_datetime(sales['Timestamp']).dt.to_period('W').astype(str)
            merged['funnel_sales'] = sources['funnel_data'].merge(
                sales, on='Week', how='left')

    # Support impact: connect support tickets to CRM and sales
    if not sources['support_tickets'].empty and not sources['crm_data'].empty:
        merged['support_crm'] = sources['support_tickets'].merge(
            sources['crm_data'], on='customer_id', how='left')
    if not sources['support_tickets'].empty and not sources['sales_data'].empty:
        merged['support_sales'] = sources['support_tickets'].merge(
            sources['sales_data'], left_on='customer_id', right_on='CustomerID', how='left')

    # Logistics impact: connect delivery to sales and support
    if not sources['delivery_data'].empty and not sources['sales_data'].empty:
        delivery = sources['delivery_data'].copy()
        sales = sources['sales_data'].copy()
        delivery['OrderID'] = delivery['OrderID'].astype(str)
        sales['OrderID'] = sales['OrderID'].astype(str)
        merged['delivery_sales'] = delivery.merge(
            sales, on='OrderID', how='left')
    # delivery_data does not have CustomerID, so skip this merge or use City if relevant
    if not sources['delivery_data'].empty and not sources['support_tickets'].empty:
        if 'City' in sources['delivery_data'].columns and 'city' in sources['support_tickets'].columns:
            merged['delivery_support'] = sources['delivery_data'].merge(
                sources['support_tickets'], left_on='City', right_on='city', how='left')

    # Marketing attribution: connect marketing attribution to campaigns and sales
    if not sources['marketing_attribution'].empty and not sources['marketing_campaigns'].empty:
        merged['attribution_campaign'] = sources['marketing_attribution'].merge(
            sources['marketing_campaigns'], left_on='CampaignID', right_on='CampaignID', how='left')
    if not sources['marketing_attribution'].empty and not sources['sales_data'].empty:
        merged['attribution_sales'] = sources['marketing_attribution'].merge(
            sources['sales_data'], left_on='OrderID', right_on='OrderID', how='left')

    # Competitor benchmarking: connect competitor data to sales
    if not sources['competitor_data'].empty and not sources['sales_data'].empty:
        competitor = sources['competitor_data'].copy()
        sales = sources['sales_data'].copy()
        competitor['ProductID'] = competitor['ProductID'].astype(str) if 'ProductID' in competitor.columns else None
        sales['ProductID'] = sales['ProductID'].astype(str)
        merged['competitor_sales'] = competitor.merge(
            sales, on='ProductID', how='left')

    return merged
