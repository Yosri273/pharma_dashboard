# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Data Loading and Preparation Module - V20.0
#
# This module is the "engine room" of the application. It handles all data
# loading, preparation, cleaning, and complex enrichment logic. The prepared
# DataFrames are stored in a global dictionary for other modules to use.
# -----------------------------------------------------------------------------

import pandas as pd
from datetime import datetime, timedelta
import joblib
from database import get_engine, refresh_all_data

# --- 1. INITIALIZE GLOBAL DATA STORE ---
# This dictionary will hold all our prepared dataframes.
DATA = {}

# --- 2. DATA PREPARATION & ENRICHMENT FUNCTION ---
def prepare_and_enrich_data(dataframes):
    """Performs all calculations and transformations on the loaded data."""
    print("--- Starting data preparation and enrichment ---")
    
    # Safely get each dataframe from the input dictionary
    sales_df = dataframes.get('sales', pd.DataFrame())
    delivery_df = dataframes.get('deliveries', pd.DataFrame())
    customer_df = dataframes.get('customers', pd.DataFrame())
    marketing_campaigns_df = dataframes.get('marketing_campaigns', pd.DataFrame())
    marketing_attribution_df = dataframes.get('marketing_attribution', pd.DataFrame())
    competitor_df = dataframes.get('competitors', pd.DataFrame())
    funnel_df = dataframes.get('sales_funnel', pd.DataFrame())

    # --- Base Data Type Conversion & Feature Creation ---
    if not sales_df.empty:
        sales_df['timestamp'] = pd.to_datetime(sales_df['timestamp'])
        sales_df['date'] = sales_df['timestamp'].dt.date
        sales_df['week'] = sales_df['timestamp'].dt.to_period('W').astype(str)
        sales_df['month'] = sales_df['timestamp'].dt.to_period('M').astype(str)
    dataframes['sales'] = sales_df

    if not delivery_df.empty:
        delivery_df['orderdate'] = pd.to_datetime(delivery_df['orderdate'], errors='coerce')
        delivery_df['date'] = delivery_df['orderdate'].dt.date
        delivery_df['actualdeliverydate'] = pd.to_datetime(delivery_df['actualdeliverydate'], errors='coerce')
        delivery_df['promiseddate'] = pd.to_datetime(delivery_df['promiseddate'], errors='coerce')
        if 'actualdeliverydate' in delivery_df.columns and 'orderdate' in delivery_df.columns:
            delivery_df['delivery_time_days'] = (delivery_df['actualdeliverydate'] - delivery_df['orderdate']).dt.days
        if 'actualdeliverydate' in delivery_df.columns and 'promiseddate' in delivery_df.columns:
            delivery_df['on_time'] = delivery_df['actualdeliverydate'] <= delivery_df['promiseddate']
    dataframes['deliveries'] = delivery_df

    # --- Customer Segmentation ---
    customer_analysis_df = pd.DataFrame()
    if not sales_df.empty and not customer_df.empty:
        customer_df['joindate'] = pd.to_datetime(customer_df['joindate'])
        rfm_df = sales_df.groupby('customerid').agg(
            last_purchase_date=('timestamp', 'max'),
            frequency=('orderid', 'nunique'),
            monetary=('netsale', 'sum')
        ).reset_index()
        current_date = datetime.now()
        rfm_df['recency'] = (current_date - rfm_df['last_purchase_date']).dt.days
        customer_analysis_df = pd.merge(customer_df, rfm_df, on='customerid', how='left')
        def get_status(row):
            join_recency = (current_date - row['joindate']).days
            if join_recency <= 90: return 'New'
            if pd.isna(row['recency']): return 'Never Purchased'
            if row['recency'] <= 90: return 'Active'
            if 90 < row['recency'] <= 180: return 'Dormant (At-Risk)'
            return 'Churn Risk'
        customer_analysis_df['status'] = customer_analysis_df.apply(get_status, axis=1)
    dataframes['customer_analysis'] = customer_analysis_df
    
    # --- Marketing Performance ---
    campaign_performance_df = pd.DataFrame()
    if not all(df.empty for df in [sales_df, marketing_campaigns_df, marketing_attribution_df]):
        order_revenue = sales_df.groupby('orderid')['netsale'].sum().reset_index()
        attributed_revenue = pd.merge(marketing_attribution_df, order_revenue, on='orderid')
        campaign_revenue = attributed_revenue.groupby('campaignid')['netsale'].sum().reset_index()
        campaign_conversions = attributed_revenue.groupby('campaignid')['orderid'].nunique().reset_index().rename(columns={'orderid': 'conversions'})
        campaign_performance_df = pd.merge(marketing_campaigns_df, campaign_revenue, on='campaignid', how='left')
        campaign_performance_df = pd.merge(campaign_performance_df, campaign_conversions, on='campaignid', how='left')
        campaign_performance_df['conversions'] = campaign_performance_df['conversions'].fillna(0)
        campaign_performance_df['netsale'] = campaign_performance_df['netsale'].fillna(0)
        campaign_performance_df['roas'] = campaign_performance_df.apply(lambda r: r['netsale']/r['totalcost'] if r['totalcost']>0 else 0, axis=1)
        campaign_performance_df['cpa'] = campaign_performance_df.apply(lambda r: r['totalcost']/r['conversions'] if r['conversions']>0 else 0, axis=1)
        campaign_performance_df['ctr'] = campaign_performance_df.apply(lambda r: (r['clicks']/r['impressions'])*100 if r['impressions']>0 else 0, axis=1)
    dataframes['campaign_performance'] = campaign_performance_df

    # --- Profitability Analysis ---
    profit_df = pd.DataFrame()
    if not all(df.empty for df in [sales_df, delivery_df, marketing_campaigns_df, marketing_attribution_df]):
        profit_df = pd.merge(sales_df, delivery_df[['orderid', 'deliverycost']], on='orderid', how='left')
        profit_df = pd.merge(profit_df, marketing_attribution_df, on='orderid', how='left')
        if not marketing_attribution_df.empty:
            campaign_costs = marketing_campaigns_df.copy()
            order_counts = marketing_attribution_df['campaignid'].value_counts().reset_index()
            order_counts.columns = ['campaignid', 'orders_in_campaign']
            campaign_costs = pd.merge(campaign_costs, order_counts, on='campaignid', how='left')
            campaign_costs['marketing_cost_per_order'] = campaign_costs.apply(lambda r: r['totalcost']/r['orders_in_campaign'] if pd.notna(r['orders_in_campaign']) and r['orders_in_campaign']>0 else 0, axis=1)
            profit_df = pd.merge(profit_df, campaign_costs[['campaignid', 'marketing_cost_per_order']], on='campaignid', how='left')
        else: profit_df['marketing_cost_per_order'] = 0
        profit_df['deliverycost'] = profit_df['deliverycost'].fillna(delivery_df['deliverycost'].mean())
        profit_df['marketing_cost_per_order'] = profit_df['marketing_cost_per_order'].fillna(0)
        profit_df['total_cost'] = profit_df.get('costofgoodssold', 0) + profit_df.get('deliverycost', 0) + profit_df.get('marketing_cost_per_order', 0)
        profit_df['net_profit'] = profit_df.get('netsale', 0) - profit_df['total_cost']
        profit_df['profit_margin'] = profit_df.apply(lambda r: (r['net_profit']/r['netsale'])*100 if r['netsale']>0 else 0, axis=1)
    dataframes['profit'] = profit_df

    # --- Competitor Price Comparison ---
    price_comparison_df = pd.DataFrame()
    if not competitor_df.empty and not sales_df.empty:
        competitor_df['date'] = pd.to_datetime(competitor_df['date'])
        our_prices = sales_df.groupby('productname')['netsale'].mean().reset_index().rename(columns={'netsale': 'our_price'})
        competitor_avg_prices = competitor_df.groupby('productname')['price'].mean().reset_index().rename(columns={'price': 'avg_competitor_price'})
        price_comparison_df = pd.merge(our_prices, competitor_avg_prices, on='productname', how='inner')
        price_comparison_df['price_difference'] = price_comparison_df['our_price'] - price_comparison_df['avg_competitor_price']
    dataframes['price_comparison'] = price_comparison_df

    # --- Load the trained model and scaler ---
    predictions_df = pd.DataFrame()
    try:
        churn_model = joblib.load('churn_model.pkl')
        scaler = joblib.load('scaler.pkl')
        print("  [SUCCESS] Churn prediction model loaded.")
        if not customer_analysis_df.empty:
            # Recreate the feature set for prediction
            features_df_predict = customer_analysis_df.copy()
            features_df_predict = features_df_predict.rename(columns={'recency':'Recency', 'frequency':'Frequency', 'monetary':'MonetaryValue'})
            features_df_predict['tenure_days'] = (datetime.now() - features_df_predict['joindate']).dt.days
            features_df_predict = features_df_predict.merge(sales_df.groupby('customerid').agg(avg_basket_value=('netsale', 'mean'), total_quantity=('quantity', 'sum')), on='customerid')
            
            model_features = pd.get_dummies(features_df_predict.drop(['customerid', 'joindate', 'last_purchase_date', 'city'], axis=1), columns=['segment'], drop_first=True)
            model_columns = ['Recency', 'Frequency', 'MonetaryValue', 'avg_basket_value', 'total_quantity', 'tenure_days', 'segment_Silver', 'segment_Gold']
            for col in model_columns:
                if col not in model_features.columns:
                    model_features[col] = 0
            model_features = model_features[model_columns]
            
            features_scaled = scaler.transform(model_features)
            churn_probabilities = churn_model.predict_proba(features_scaled)[:, 1]
            
            predictions_df = features_df_predict[['customerid', 'city', 'segment', 'Recency', 'Frequency', 'MonetaryValue']].copy()
            predictions_df['churn_probability'] = churn_probabilities
            predictions_df = predictions_df.sort_values('churn_probability', ascending=False)
            
    except FileNotFoundError:
        print("  [WARNING] 'churn_model.pkl' or 'scaler.pkl' not found.")
    except Exception as e:
        print(f"  [ERROR] Failed to load model or make predictions: {e}")
    dataframes['predictions'] = predictions_df
    
    print("--- All data enrichment and preparation complete. ---")
    return dataframes

# --- 3. INITIAL DATA LOAD ---
# This block runs when the application starts up.
engine = get_engine()
DATA = prepare_and_enrich_data(refresh_all_data(engine))

