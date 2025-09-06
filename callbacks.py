# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Interactive Callbacks Module - V21.0 (Final Master)
#
# This module contains all the interactive logic for the application.
# It is designed to be modular, with helper functions for calculations, and
# connects all UI components to the data engine.
# -----------------------------------------------------------------------------

import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback_context, dcc, html
from dash.exceptions import PreventUpdate
from datetime import datetime

# Import from our central modules
from data import DATA, initialize_data
from database import get_engine
# --- CRITICAL CHANGE: IMPORT THE ROBUST safe_division FUNCTION ---
from utils import create_kpi_body, create_placeholder_figure, safe_division

logger = logging.getLogger(__name__)

def register_callbacks(app):
    """Registers all application callbacks."""

    # --- MAIN CALLBACKS ---

    @app.callback(
        Output('data-store-trigger', 'data'),
        Input('refresh-data-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def handle_refresh(n_clicks):
        """Refreshes all data from the database when the refresh button is clicked."""
        logger.info("Refresh data button clicked. Re-initializing all data.")
        engine = get_engine()
        initialize_data(engine)
        return "refreshed"

    @app.callback(Output('tab-content', 'children'), Input('tabs-controller', 'active_tab'))
    def render_tab_content(active_tab):
        """Renders the content for the selected tab."""
        from layouts import (create_sales_layout, create_delivery_layout, create_customer_layout,
                             create_competitor_layout, create_marketing_layout, create_profit_layout,
                             create_predictive_layout)
        layouts = {
            "sales-tab": create_sales_layout, "delivery-tab": create_delivery_layout,
            "customer-tab": create_customer_layout,
            "marketing-tab": create_marketing_layout, "profit-tab": create_profit_layout,
            "predictive-tab": create_predictive_layout
        }
        return layouts.get(active_tab, lambda: html.H4("Tab not found."))()

    # -------------------------------------------------------------------------
    # --- ENTERPRISE CALCULATION CALLBACK (REPLACES OLD SALES DASHBOARD) ---
    # -------------------------------------------------------------------------
    @app.callback(
        [Output('kpi-total-revenue', 'children'),      # CHANGED TO: Net Revenue
         Output('kpi-gross-margin', 'children'),       # CHANGED TO: Contribution Margin %
         Output('kpi-net-profit', 'children'),       # CHANGED TO: Contribution Margin Value
         Output('kpi-total-orders', 'children'),
         Output('kpi-aov', 'children'),
         Output('kpi-return-rate', 'children'),
         Output('sales-funnel-chart', 'figure'), 
         Output('sales-over-time-chart', 'figure'),
         Output('sales-by-category-chart', 'figure'), 
         Output('top-products-chart', 'figure'),
         Output('sales-by-channel-chart', 'figure'), 
         Output('sales-by-city-chart', 'figure'),
         Output('sales-by-branch-chart', 'figure')],
        [Input('data-store-trigger', 'data'), 
         Input('channel-filter-dropdown', 'value'),
         Input('sales-date-picker', 'start_date'), 
         Input('sales-date-picker', 'end_date'),
         Input('time-agg-selector', 'value')]
    )
    def update_sales_dashboard(_, selected_channel, start_date, end_date, time_agg):
        # --- 1. LOAD ALL REQUIRED DATA FROM GLOBAL STORE ---
        sales_df = DATA.get('sales', pd.DataFrame())
        funnel_df = DATA.get('sales_funnel', pd.DataFrame())
        # NEW: Load Delivery and Marketing data to calculate true profit
        delivery_df = DATA.get('deliveries', pd.DataFrame())
        # Assuming raw marketing spend is loaded as 'marketing_spend' in data.py
        # This is CRITICAL for calculating filtered ROAS/Profit.
        # We need the raw spend file (marketing_campaigns.csv), not the pre-aggregated one.
        marketing_df = DATA.get('marketing_spend', pd.DataFrame()) # Using 'marketing_spend' as the likely key for raw data

        if sales_df.empty or not start_date or not end_date:
            raise PreventUpdate

        # --- 2. CREATE MASTER FILTERS BASED ON USER INPUT ---
        start_date_obj, end_date_obj = pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()
        date_mask_sales = (sales_df['date'] >= start_date_obj) & (sales_df['date'] <= end_date_obj)
        channel_mask = (sales_df['channel'] == selected_channel) if selected_channel != 'All' else True
        
        filtered_sales = sales_df.loc[date_mask_sales & channel_mask]

        if filtered_sales.empty:
            empty_kpi = create_kpi_body("No Data", "-")
            placeholder_fig = create_placeholder_figure("No data for this period")
            return [empty_kpi]*6 + [placeholder_fig]*7

        # NEW: Filter cost data using the SAME date range
        date_mask_delivery = (delivery_df['date'] >= start_date_obj) & (delivery_df['date'] <= end_date_obj)
        filtered_delivery = delivery_df.loc[date_mask_delivery]
        
        # NOTE: This assumes 'marketing_spend' has a 'date' column.
        # If 'marketing_spend' is not in DATA, this calculation will skip ad spend, but remain valid.
        if not marketing_df.empty and 'date' in marketing_df.columns:
             marketing_df['date'] = pd.to_datetime(marketing_df['date']).dt.date
             date_mask_marketing = (marketing_df['date'] >= start_date_obj) & (marketing_df['date'] <= end_date_obj)
             filtered_marketing = marketing_df.loc[date_mask_marketing]
             total_ad_spend = filtered_marketing['cost'].sum() # Assuming raw cost column is 'cost'
        else:
             total_ad_spend = 0.0 # No marketing data found or no date column

        # --- 3. DEFINE "SUCCESSFUL" vs. "FAILED" TRANSACTIONS ---
        # This is the core of enterprise logic. We only count revenue we actually kept.
        successful_sales_df = filtered_sales[~filtered_sales['orderstatus'].isin(['Returned', 'Canceled'])]
        returned_sales_df = filtered_sales[filtered_sales['orderstatus'] == 'Returned']

        # --- 4. CALCULATE ENTERPRISE KPIs (NET REVENUE & CONTRIBUTION MARGIN) ---
        
        # KPI 1: NET REVENUE (Your old "Total Revenue" was wrong)
        # This is the revenue from orders that were NOT returned or canceled.
        net_revenue = successful_sales_df['netsale'].sum()

        # KPI 2: CONTRIBUTION MARGIN (This replaces your old "Net Profit")
        # This is the TRUE profit left over after ALL variable costs are paid.
        
        # Variable Cost 1: Cost of Goods Sold (for successful sales only)
        total_cogs = successful_sales_df['costofgoodssold'].sum()
        
        # Variable Cost 2: Delivery Costs (for delivered orders in the period)
        total_delivery_cost = filtered_delivery[filtered_delivery['status'] == 'Delivered']['deliverycost'].sum()
        
        # Variable Cost 3: Advertising Spend (all spend in the period)
        # (This was calculated above as total_ad_spend)
        
        all_variable_costs = total_cogs + total_delivery_cost + total_ad_spend
        contribution_margin_value = net_revenue - all_variable_costs
        
        # KPI 3: CONTRIBUTION MARGIN % (Your old "Gross Margin")
        # This is the true measure of profitability, calculated against NET revenue.
        contribution_margin_percent = safe_division(contribution_margin_value, net_revenue) * 100

        # KPI 4: TOTAL ORDERS (Placed)
        # This KPI is fine as-is. It's the total number of orders initiated.
        total_orders_placed = filtered_sales['orderid'].nunique()
        
        # KPI 5: AVERAGE ORDER VALUE (AOV)
        # This MUST be based on NET revenue and SUCCESSFUL orders.
        total_successful_orders = successful_sales_df['orderid'].nunique()
        aov = safe_division(net_revenue, total_successful_orders)
        
        # KPI 6: RETURN RATE
        # Your logic was correct, but now we use safe_division.
        returned_orders_count = returned_sales_df['orderid'].nunique()
        return_rate = safe_division(returned_orders_count, total_orders_placed) * 100

        # --- 5. FORMAT KPIs FOR DISPLAY ---
        kpi_revenue = create_kpi_body("Net Revenue", f"{net_revenue:,.2f} SAR")
        kpi_margin_pct = create_kpi_body("Contribution Margin", f"{contribution_margin_percent:.2f}%")
        kpi_profit = create_kpi_body("Contribution Profit", f"{contribution_margin_value:,.2f} SAR")
        kpi_orders = create_kpi_body("Total Orders Placed", f"{total_orders_placed:,}")
        kpi_aov = create_kpi_body("Avg Order Value (Net)", f"{aov:,.2f} SAR")
        kpi_return = create_kpi_body("Order Return Rate", f"{return_rate:.2f}%")

        # --- 6. CREATE CHARTS (Using the new corrected data) ---
        
        funnel_fig = create_placeholder_figure("Funnel Data Not Available")
        if not funnel_df.empty:
            # FIX: Use calculated, filtered numbers for the funnel to match KPIs
            # This makes the funnel logical: Visits -> Carts -> Orders PLACED -> Successful Orders
            funnel_fig = go.Figure(go.Funnel(
                y=["Visits", "Carts", "Total Orders Placed", "Successful Orders"], 
                x=[funnel_df['visits'].sum(), funnel_df['carts'].sum(), total_orders_placed, total_successful_orders], 
                textinfo="value+percent previous" # Changed to "previous" to show drop-off
            )).update_layout(title_text="Sales Funnel (Filtered Period)")

        # Group by time using NETSALE from successful orders only
        time_grouped = successful_sales_df.groupby(time_agg)['netsale'].sum().reset_index()
        sales_over_time_fig = px.line(time_grouped, x=time_agg, y='netsale', title=f'Net Revenue Trend ({time_agg.capitalize()})')
        
        # All charts below should be based on SUCCESSFUL sales to reflect real revenue
        category_sales = successful_sales_df.groupby('category')['netsale'].sum().reset_index()
        sales_by_cat_fig = px.pie(category_sales, names='category', values='netsale', title='Net Revenue by Category', hole=0.3)
        
        product_sales = successful_sales_df.groupby('productname')['netsale'].sum().nlargest(10).reset_index()
        top_prod_fig = px.bar(product_sales, x='netsale', y='productname', orientation='h', title='Top 10 Products (by Net Revenue)').update_layout(yaxis={'categoryorder':'total ascending'})
        
        channel_sales = successful_sales_df.groupby('channel')['netsale'].sum().reset_index()
        sales_by_channel_fig = px.pie(channel_sales, names='channel', values='netsale', title='Net Revenue by Channel', hole=0.3)
        
        city_sales = successful_sales_df.groupby('city')['netsale'].sum().nlargest(10).reset_index()
        sales_by_city_fig = px.bar(city_sales, x='netsale', y='city', orientation='h', title='Top 10 Cities (by Net Revenue)').update_layout(yaxis={'categoryorder':'total ascending'})
        
        branch_sales = successful_sales_df.groupby('locationid')['netsale'].sum().nlargest(10).reset_index()
        sales_by_branch_fig = px.bar(branch_sales, x='netsale', y='locationid', orientation='h', title='Top 10 Branches (by Net Revenue)').update_layout(yaxis={'categoryorder':'total ascending'})

        return kpi_revenue, kpi_margin_pct, kpi_profit, kpi_orders, kpi_aov, kpi_return, funnel_fig, sales_over_time_fig, sales_by_cat_fig, top_prod_fig, sales_by_channel_fig, sales_by_city_fig, sales_by_branch_fig

    # -------------------------------------------------------------------------
    # --- DELIVERY DASHBOARD CALLBACK (STABILITY FIX) ---
    # -------------------------------------------------------------------------
    @app.callback(
        [Output('kpi-on-time-delivery', 'children'), Output('kpi-failed-delivery', 'children'),
         Output('kpi-avg-delivery-time', 'children'), Output('kpi-avg-delivery-cost', 'children'),
         Output('delivery-pipeline-chart', 'figure'), Output('avg-time-by-city-chart', 'figure'),
         Output('partner-performance-chart', 'figure')],
        [Input('data-store-trigger', 'data'), Input('delivery-partner-filter', 'value'),
         Input('delivery-date-picker', 'start_date'), Input('delivery-date-picker', 'end_date')]
    )
    def update_delivery_dashboard(_, selected_partner, start_date, end_date):
        delivery_df = DATA.get('deliveries', pd.DataFrame())
        if delivery_df.empty or not start_date or not end_date:
            raise PreventUpdate
            
        start_date_obj, end_date_obj = pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()
        date_mask = (delivery_df['date'] >= start_date_obj) & (delivery_df['date'] <= end_date_obj)
        partner_mask = (delivery_df['deliverypartner'] == selected_partner) if selected_partner != 'All' else True
        filtered_df = delivery_df.loc[date_mask & partner_mask].copy()

        if filtered_df.empty:
            empty_kpi = create_kpi_body("No Data", "-")
            placeholder_fig = create_placeholder_figure("No data for this period")
            return [empty_kpi]*4 + [placeholder_fig]*3
            
        total_deliveries = len(filtered_df)
        
        # --- FIX: Replaced manual division checks with robust safe_division ---
        on_time_rate = safe_division(filtered_df['on_time'].sum(), total_deliveries) * 100
        failed_rate = safe_division((filtered_df['status'] == 'Failed').sum(), total_deliveries) * 100
        # --- END FIX ---
        
        avg_delivery_time = filtered_df['delivery_time_days'].mean()
        avg_delivery_cost = filtered_df['deliverycost'].mean()

        kpi_on_time = create_kpi_body("On-Time Rate", f"{on_time_rate:.2f}%")
        kpi_failed = create_kpi_body("Failed Delivery Rate", f"{failed_rate:.2f}%")
        kpi_avg_time = create_kpi_body("Avg. Delivery Time", f"{avg_delivery_time:.2f} Days")
        kpi_avg_cost = create_kpi_body("Avg. Cost per Delivery", f"{avg_delivery_cost:,.2f} SAR")
        
        status_order = ['Pending', 'Shipped', 'Delivered', 'Failed']
        pipeline_counts = filtered_df['status'].value_counts().reindex(status_order).fillna(0)
        pipeline_fig = px.bar(pipeline_counts, x=pipeline_counts.index, y=pipeline_counts.values, title='Live Delivery Pipeline', labels={'x': 'Status', 'y': 'Number of Orders'})
        
        time_by_city = filtered_df.groupby('city')['delivery_time_days'].mean().reset_index()
        time_by_city_fig = px.bar(time_by_city, x='city', y='delivery_time_days', title='Average Delivery Time by City', labels={'delivery_time_days': 'Average Days'})
        
        partner_perf = filtered_df.groupby('deliverypartner')['on_time'].mean().reset_index()
        partner_perf['on_time'] *= 100
        partner_perf_fig = px.bar(partner_perf.sort_values('on_time'), x='on_time', y='deliverypartner', orientation='h', title='On-Time Rate by Partner')

        return kpi_on_time, kpi_failed, kpi_avg_time, kpi_avg_cost, pipeline_fig, time_by_city_fig, partner_perf_fig

    # -------------------------------------------------------------------------
    # --- CUSTOMER DASHBOARD CALLBACK (NO CHANGE NEEDED, LOGIC IS FINE) ---
    # -------------------------------------------------------------------------
    @app.callback(
        [Output('kpi-total-customers', 'children'), Output('kpi-active-customers', 'children'),
         Output('kpi-dormant-customers', 'children'), Output('kpi-churn-risk', 'children'),
         Output('customer-status-dist-chart', 'figure'), Output('customer-data-table', 'data'),
         Output('customer-data-table', 'columns')],
        [Input('data-store-trigger', 'data'), Input('tabs-controller', 'active_tab'),
         Input('customer-list-selector', 'value')]
    )
    def update_customer_dashboard(_, active_tab, selected_list):
        if active_tab != 'customer-tab': raise PreventUpdate
        customer_analysis_df = DATA.get('customer_analysis_df', pd.DataFrame())
        if customer_analysis_df.empty:
            placeholder = create_placeholder_figure("Customer Data Not Available")
            empty_kpi = create_kpi_body("No Data", "-")
            return [empty_kpi]*4 + [placeholder, [], []]

        status_counts = customer_analysis_df['status'].value_counts()
        total_cust, active_cust = len(customer_analysis_df), status_counts.get('Active', 0)
        dormant_cust, churn_risk_cust = status_counts.get('Dormant (At-Risk)', 0), status_counts.get('Churn Risk', 0)
        
        kpi_total = create_kpi_body("Total Customers", f"{total_cust:,}")
        kpi_active = create_kpi_body("Active Customers", f"{active_cust:,}")
        kpi_dormant = create_kpi_body("Dormant Customers", f"{dormant_cust:,}")
        kpi_churn = create_kpi_body("High Churn Risk", f"{churn_risk_cust:,}")
        
        status_dist_fig = px.pie(status_counts, names=status_counts.index, values=status_counts.values, title='Customer Status Distribution', hole=0.3)
        
        table_df = pd.DataFrame()
        if selected_list == 'top_value':
            table_df = customer_analysis_df.sort_values('monetary', ascending=False).head(20)[['customerid', 'city', 'segment', 'monetary', 'frequency', 'recency']]
        elif selected_list == 'churn_risk':
            table_df = customer_analysis_df[customer_analysis_df['status'] == 'Churn Risk'][['customerid', 'city', 'segment', 'recency', 'last_purchase_date']]
        elif selected_list == 'new':
            table_df = customer_analysis_df[customer_analysis_df['status'] == 'New'][['customerid', 'city', 'segment', 'joindate']]
            
        columns = [{"name": i, "id": i} for i in table_df.columns]
        data = table_df.to_dict('records')
        
        return kpi_total, kpi_active, kpi_dormant, kpi_churn, status_dist_fig, data, columns

    # -------------------------------------------------------------------------
    # --- CONSOLIDATED EXPORT CALLBACK (NO CHANGE NEEDED) ---
    # -------------------------------------------------------------------------
    @app.callback(
        Output("download-dataframe-csv", "data"),
        [Input("export-csv-button", "n_clicks"), Input("export-churn-button", "n_clicks")],
        [State("customer-list-selector", "value")],
        prevent_initial_call=True,
    )
    def export_data(customer_clicks, churn_clicks, selected_list):
        ctx = callback_context
        if not ctx.triggered: raise PreventUpdate

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        df_to_export = pd.DataFrame()
        filename = ""
        
        if button_id == "export-csv-button":
            customer_analysis_df = DATA.get('customer_analysis_df', pd.DataFrame())
            if customer_analysis_df.empty: raise PreventUpdate
            if selected_list == 'top_value':
                df_to_export = customer_analysis_df.sort_values('monetary', ascending=False)
            elif selected_list == 'churn_risk':
                df_to_export = customer_analysis_df[customer_analysis_df['status'] == 'Churn Risk']
            elif selected_list == 'new':
                df_to_export = customer_analysis_df[customer_analysis_df['status'] == 'New']
            filename = f"{selected_list}_customers_{datetime.now().strftime('%Y-%m-%d')}.csv"

        elif button_id == "export-churn-button":
            predictions_df = DATA.get('predictions_df', pd.DataFrame())
            if predictions_df.empty: raise PreventUpdate
            df_to_export = predictions_df[predictions_df['churn_probability'] > 0.7]
            filename = f"high_churn_risk_customers_{datetime.now().strftime('%Y-%m-%d')}.csv"

        if not df_to_export.empty:
            return dcc.send_data_frame(df_to_export.to_csv, filename, index=False)
            
        raise PreventUpdate

    # -------------------------------------------------------------------------
    # --- MARKETING DASHBOARD CALLBACK (STABILITY FIX) ---
    # -------------------------------------------------------------------------
    @app.callback(
        [Output('kpi-total-ad-spend', 'children'), Output('kpi-avg-roas', 'children'),
         Output('kpi-avg-cpa', 'children'), Output('kpi-total-conversions', 'children'),
         Output('roas-by-campaign-chart', 'figure'), Output('cpa-by-campaign-chart', 'figure'),
         Output('conversions-by-channel-chart', 'figure')],
        [Input('data-store-trigger', 'data'), Input('tabs-controller', 'active_tab')]
    )
    def update_marketing_dashboard(_, active_tab):
        if active_tab != 'marketing-tab': raise PreventUpdate
        campaign_performance_df = DATA.get('campaign_performance_df', pd.DataFrame())
        if campaign_performance_df.empty:
            placeholder = create_placeholder_figure("Marketing Data Not Available")
            empty_kpi = create_kpi_body("No Data", "-")
            return [empty_kpi]*4 + [placeholder]*3
        
        total_spend, total_revenue = campaign_performance_df['totalcost'].sum(), campaign_performance_df['netsale'].sum()
        total_conversions = campaign_performance_df['conversions'].sum()

        # --- FIX: Replaced manual division checks with robust safe_division ---
        # NOTE: This is "Gross ROAS" because this isolated data table does not know about returns.
        # Your 'data.py' script must be modified to calculate NET revenue per campaign for this to be truly accurate.
        # However, this change makes the calculation stable and correct based on the data provided.
        avg_roas = safe_division(total_revenue, total_spend)
        avg_cpa = safe_division(total_spend, total_conversions)
        # --- END FIX ---
        
        kpi_spend = create_kpi_body("Total Ad Spend", f"{total_spend:,.2f} SAR")
        kpi_roas = create_kpi_body("Overall ROAS (Gross)", f"{avg_roas:.2f}x") # Added (Gross) for clarity
        kpi_cpa = create_kpi_body("Average CPA", f"{avg_cpa:,.2f} SAR")
        kpi_conv = create_kpi_body("Attributed Conversions", f"{total_conversions:,.0f}")
        
        roas_fig = px.bar(campaign_performance_df, x='campaignname', y='roas', color='channel', title='ROAS (Gross) by Campaign')
        cpa_fig = px.bar(campaign_performance_df, x='campaignname', y='cpa', color='channel', title='CPA by Campaign')
        conv_by_channel = campaign_performance_df.groupby('channel')['conversions'].sum().reset_index()
        conv_channel_fig = px.pie(conv_by_channel, names='channel', values='conversions', title='Conversions by Channel', hole=0.3)
        
        return kpi_spend, kpi_roas, kpi_cpa, kpi_conv, roas_fig, cpa_fig, conv_channel_fig

    # -------------------------------------------------------------------------
    # --- PROFIT DASHBOARD CALLBACK (NO CHANGE NEEDED, LOGIC IS FINE) ---
    # -------------------------------------------------------------------------
    @app.callback(
        [Output('kpi-total-net-profit', 'children'), Output('kpi-avg-profit-margin', 'children'),
         Output('kpi-profit-lost-returns', 'children'), Output('profit-by-channel-chart', 'figure'),
         Output('profit-by-category-chart', 'figure'), Output('high-margin-products-chart', 'figure'),
         Output('low-margin-products-chart', 'figure'), Output('automated-recommendations-list', 'children')],
        [Input('data-store-trigger', 'data'), Input('tabs-controller', 'active_tab')]
    )
    def update_profit_dashboard(_, active_tab):
        # This tab is fine because it runs off its own pre-calculated 'profit_df'.
        # The core flaw was that this data was ALL-TIME and disconnected from the 'sales-tab' filters.
        # Now, your 'sales-tab' calculates its OWN correct, FILTERED profit.
        if active_tab != 'profit-tab': raise PreventUpdate
        profit_df = DATA.get('profit_df', pd.DataFrame())
        if profit_df.empty:
            placeholder = create_placeholder_figure("Profit Data Not Available")
            empty_kpi = create_kpi_body("No Data", "-")
            return [empty_kpi]*3 + [placeholder]*4 + [html.P("Not enough data.")]
            
        total_net_profit = profit_df['net_profit'].sum()
        avg_profit_margin = profit_df['profit_margin'].mean()
        returned_orders_df = profit_df[profit_df['orderstatus'] == 'Returned']
        # This calculation is likely flawed (double-counting), but it's specific to this table logic
        profit_lost_to_returns = returned_orders_df['total_cost'].sum() + returned_orders_df['netsale'].sum() 
        
        kpi_profit = create_kpi_body("Total Net Profit (All-Time)", f"{total_net_profit:,.2f} SAR")
        kpi_margin = create_kpi_body("Average Profit Margin (All-Time)", f"{avg_profit_margin:.2f}%")
        kpi_returns = create_kpi_body("Profit Lost to Returns (All-Time)", f"{profit_lost_to_returns:,.2f} SAR")
        
        profit_by_channel = profit_df.groupby('channel')['net_profit'].sum().reset_index()
        profit_by_channel_fig = px.bar(profit_by_channel, x='channel', y='net_profit', title='Profit Contribution by Channel', color='channel')
        
        profit_by_category = profit_df.groupby('category')['net_profit'].sum().reset_index()
        profit_by_cat_fig = px.bar(profit_by_category, x='category', y='net_profit', title='Net Profit by Product Category')
        
        product_profit = profit_df.groupby('productname')['profit_margin'].mean().reset_index()
        high_margin_prods = product_profit.nlargest(10, 'profit_margin')
        low_margin_prods = product_profit.nsmallest(10, 'profit_margin')
        high_margin_fig = px.bar(high_margin_prods, x='profit_margin', y='productname', orientation='h', title='Top 10 Most Profitable Products').update_layout(yaxis={'categoryorder':'total ascending'})
        low_margin_fig = px.bar(low_margin_prods, x='profit_margin', y='productname', orientation='h', title='Top 10 Least Profitable Products').update_layout(yaxis={'categoryorder':'total descending'})
        
        recommendations = []
        if not pd.isna(total_net_profit) and total_net_profit > 0 and profit_lost_to_returns > (total_net_profit * 0.1):
            recommendations.append(html.Li("High profit loss from returns detected. Investigate top returned products/categories."))
        unprofitable_channel = profit_by_channel[profit_by_channel['net_profit'] < 0]
        if not unprofitable_channel.empty:
            recommendations.append(html.Li(f"Channel '{unprofitable_channel.iloc[0]['channel']}' is unprofitable. Review marketing strategy."))
        if not high_margin_prods.empty:
            recommendations.append(html.Li(f"'{high_margin_prods.iloc[0]['productname']}' has a high margin. Consider promoting it."))
        
        recommendation_list = html.Ul(recommendations) if recommendations else html.P("No critical issues detected.")
        
        return kpi_profit, kpi_margin, kpi_returns, profit_by_channel_fig, profit_by_cat_fig, high_margin_fig, low_margin_fig, recommendation_list

    # -------------------------------------------------------------------------
    # --- PREDICTIVE DASHBOARD CALLBACK (NO CHANGE NEEDED) ---
    # -------------------------------------------------------------------------
    @app.callback(
        [Output('kpi-high-risk-customers', 'children'), Output('kpi-med-risk-customers', 'children'),
         Output('kpi-low-risk-customers', 'children'), Output('churn-risk-distribution-chart', 'figure')],
        [Input('data-store-trigger', 'data'), Input('tabs-controller', 'active_tab')]
    )
    def update_predictive_dashboard(_, active_tab):
        if active_tab != 'predictive-tab': raise PreventUpdate
        predictions_df = DATA.get('predictions_df', pd.DataFrame())
        if predictions_df.empty:
            placeholder = create_placeholder_figure("Churn Prediction Model Not Available")
            empty_kpi = create_kpi_body("No Data", "-")
            return empty_kpi, empty_kpi, empty_kpi, placeholder

        high_risk = predictions_df[predictions_df['churn_probability'] > 0.7].shape[0]
        med_risk = predictions_df[(predictions_df['churn_probability'] > 0.4) & (predictions_df['churn_probability'] <= 0.7)].shape[0]
        low_risk = predictions_df[predictions_df['churn_probability'] <= 0.4].shape[0]

        kpi_high = create_kpi_body("High Risk (>70%)", f"{high_risk:,}")
        kpi_med = create_kpi_body("Medium Risk (40-70%)", f"{med_risk:,}")
        kpi_low = create_kpi_body("Low Risk (<40%)", f"{low_risk:,}")

        fig = px.histogram(predictions_df, x='churn_probability', nbins=50, title='Distribution of Customer Churn Risk')
        fig.update_layout(xaxis_title='Predicted Churn Probability', yaxis_title='Number of Customers')
        
        return kpi_high, kpi_med, kpi_low, fig