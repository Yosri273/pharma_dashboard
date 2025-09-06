# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# E-commerce Data Analyst App - V21.1 (UI Upgrade)
#
# This is the main entry point for the application. It has been updated to
# include the Bootstrap Icons library for a more modern UI.
# -----------------------------------------------------------------------------

import dash
import logging
import sys
import dash_bootstrap_components as dbc

# Import the necessary components from our new modules
from layouts import create_main_layout
from callbacks import register_callbacks
from data import initialize_data
from database import get_engine

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- 1. INITIALIZE APP AND DATA ---
logger.info("--- Starting Pharma Analytics Hub v21.1 ---")
# --- FIX: Added Bootstrap Icons to the external stylesheets ---
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True
)
server = app.server
app.title = "Pharma Analytics Hub"

engine = get_engine()
initialize_data(engine)

# --- 2. DEFINE APP LAYOUT & REGISTER CALLBACKS ---
app.layout = create_main_layout()
register_callbacks(app)
logger.info("Application ready.")

# --- 3. RUN THE APP ---
if __name__ == '__main__':
    app.run(debug=True, port=8052)

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
from utils import create_kpi_body, create_placeholder_figure

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
            "customer-tab": create_customer_layout, "competitor-tab": create_competitor_layout,
            "marketing-tab": create_marketing_layout, "profit-tab": create_profit_layout,
            "predictive-tab": create_predictive_layout
        }
        return layouts.get(active_tab, lambda: html.H4("Tab not found."))()

    # --- SALES DASHBOARD CALLBACK ---
    @app.callback(
        [Output('kpi-total-revenue', 'children'), Output('kpi-gross-margin', 'children'),
         Output('kpi-net-profit', 'children'), Output('kpi-total-orders', 'children'),
         Output('kpi-aov', 'children'), Output('kpi-return-rate', 'children'),
         Output('sales-funnel-chart', 'figure'), Output('sales-over-time-chart', 'figure'),
         Output('sales-by-category-chart', 'figure'), Output('top-products-chart', 'figure'),
         Output('sales-by-channel-chart', 'figure'), Output('sales-by-city-chart', 'figure'),
         Output('sales-by-branch-chart', 'figure')],
        [Input('data-store-trigger', 'data'), Input('channel-filter-dropdown', 'value'),
         Input('sales-date-picker', 'start_date'), Input('sales-date-picker', 'end_date'),
         Input('time-agg-selector', 'value')]
    )
    def update_sales_dashboard(_, selected_channel, start_date, end_date, time_agg):
        sales_df = DATA.get('sales', pd.DataFrame())
        funnel_df = DATA.get('sales_funnel', pd.DataFrame())
        if sales_df.empty or not start_date or not end_date:
            raise PreventUpdate

        start_date_obj, end_date_obj = pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()
        date_mask = (sales_df['date'] >= start_date_obj) & (sales_df['date'] <= end_date_obj)
        channel_mask = (sales_df['channel'] == selected_channel) if selected_channel != 'All' else True
        filtered_sales = sales_df.loc[date_mask & channel_mask]

        if filtered_sales.empty:
            empty_kpi = create_kpi_body("No Data", "-")
            placeholder_fig = create_placeholder_figure("No data for this period")
            return [empty_kpi]*6 + [placeholder_fig]*7

        total_revenue = filtered_sales['netsale'].sum()
        total_cogs = filtered_sales['costofgoodssold'].sum()
        net_profit = total_revenue - total_cogs
        gross_margin = (net_profit / total_revenue * 100) if total_revenue > 0 else 0
        total_orders = filtered_sales['orderid'].nunique()
        aov = total_revenue / total_orders if total_orders > 0 else 0
        returned_orders = filtered_sales[filtered_sales['orderstatus'] == 'Returned']['orderid'].nunique()
        return_rate = (returned_orders / total_orders * 100) if total_orders > 0 else 0

        kpi_revenue = create_kpi_body("Total Revenue", f"{total_revenue:,.2f} SAR")
        kpi_margin = create_kpi_body("Gross Margin", f"{gross_margin:.2f}%")
        kpi_profit = create_kpi_body("Net Profit", f"{net_profit:,.2f} SAR")
        kpi_orders = create_kpi_body("Total Orders", f"{total_orders:,}")
        kpi_aov = create_kpi_body("Avg Order Value", f"{aov:,.2f} SAR")
        kpi_return = create_kpi_body("Return Rate", f"{return_rate:.2f}%")

        funnel_fig = create_placeholder_figure("Funnel Data Not Available")
        if not funnel_df.empty:
            completed = filtered_sales[filtered_sales['orderstatus'] == 'Completed']['orderid'].nunique()
            funnel_fig = go.Figure(go.Funnel(y=["Visits", "Carts", "Orders", "Fulfilled"], x=[funnel_df['visits'].sum(), funnel_df['carts'].sum(), funnel_df['orders'].sum(), completed], textinfo="value+percent initial")).update_layout(title_text="Sales Funnel")

        time_grouped = filtered_sales.groupby(time_agg)['netsale'].sum().reset_index()
        sales_over_time_fig = px.line(time_grouped, x=time_agg, y='netsale', title=f'Net Sales Trend ({time_agg.capitalize()})')
        
        category_sales = filtered_sales.groupby('category')['netsale'].sum().reset_index()
        sales_by_cat_fig = px.pie(category_sales, names='category', values='netsale', title='Sales by Category', hole=0.3)
        
        product_sales = filtered_sales.groupby('productname')['netsale'].sum().nlargest(10).reset_index()
        top_prod_fig = px.bar(product_sales, x='netsale', y='productname', orientation='h', title='Top 10 Products').update_layout(yaxis={'categoryorder':'total ascending'})
        
        channel_sales = filtered_sales.groupby('channel')['netsale'].sum().reset_index()
        sales_by_channel_fig = px.pie(channel_sales, names='channel', values='netsale', title='Sales by Channel', hole=0.3)
        
        city_sales = filtered_sales.groupby('city')['netsale'].sum().nlargest(10).reset_index()
        sales_by_city_fig = px.bar(city_sales, x='netsale', y='city', orientation='h', title='Top 10 Cities by Sales').update_layout(yaxis={'categoryorder':'total ascending'})
        
        branch_sales = filtered_sales.groupby('locationid')['netsale'].sum().nlargest(10).reset_index()
        sales_by_branch_fig = px.bar(branch_sales, x='netsale', y='locationid', orientation='h', title='Top 10 Pharmacy Branches by Sales').update_layout(yaxis={'categoryorder':'total ascending'})

        return kpi_revenue, kpi_margin, kpi_profit, kpi_orders, kpi_aov, kpi_return, funnel_fig, sales_over_time_fig, sales_by_cat_fig, top_prod_fig, sales_by_channel_fig, sales_by_city_fig, sales_by_branch_fig

    # --- DELIVERY DASHBOARD CALLBACK ---
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
        on_time_rate = (filtered_df['on_time'].sum() / total_deliveries * 100) if total_deliveries > 0 else 0
        failed_rate = ((filtered_df['status'] == 'Failed').sum() / total_deliveries * 100) if total_deliveries > 0 else 0
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

    # --- CUSTOMER DASHBOARD CALLBACK ---
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

    # --- CONSOLIDATED EXPORT CALLBACK ---
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

    # --- OTHER DASHBOARD CALLBACKS ---
    @app.callback(
        [Output('kpi-price-advantage', 'children'), Output('kpi-price-disadvantage', 'children'),
         Output('kpi-promo-frequency', 'children'), Output('price-comparison-scatter-chart', 'figure'),
         Output('promo-analysis-chart', 'figure'), Output('assortment-overlap-chart', 'figure')],
        [Input('data-store-trigger', 'data'), Input('tabs-controller', 'active_tab')]
    )
    def update_competitor_dashboard(_, active_tab):
        if active_tab != 'competitor-tab': raise PreventUpdate
        price_comparison_df = DATA.get('price_comparison_df', pd.DataFrame())
        competitor_df = DATA.get('competitors', pd.DataFrame())
        sales_df = DATA.get('sales', pd.DataFrame())
        if price_comparison_df.empty or competitor_df.empty or sales_df.empty:
            placeholder = create_placeholder_figure("Data Not Available")
            empty_kpi = create_kpi_body("No Data", "-")
            return empty_kpi, empty_kpi, empty_kpi, placeholder, placeholder, placeholder

        products_cheaper = price_comparison_df[price_comparison_df['price_difference'] < 0].shape[0]
        products_pricier = price_comparison_df[price_comparison_df['price_difference'] > 0].shape[0]
        promo_rate = (competitor_df['onpromotion'].sum() / len(competitor_df) * 100)
        
        kpi_advantage = create_kpi_body("Products We Undercut", f"{products_cheaper}")
        kpi_disadvantage = create_kpi_body("Products More Expensive", f"{products_pricier}")
        kpi_promo = create_kpi_body("Avg. Competitor Promo Rate", f"{promo_rate:.2f}%")
        
        price_comp_fig = px.scatter(price_comparison_df, x='our_price', y='avg_competitor_price', hover_name='productname', text='productname', size='price_difference', title='Our Price vs. Average Competitor Price')
        price_comp_fig.add_shape(type='line', x0=0, y0=0, x1=price_comparison_df['our_price'].max(), y1=price_comparison_df['our_price'].max(), line=dict(color='red', dash='dash'))
        
        promo_freq = competitor_df.groupby('competitor')['onpromotion'].mean().reset_index()
        promo_freq['onpromotion'] *= 100
        promo_fig = px.bar(promo_freq, x='competitor', y='onpromotion', title='Promotion Frequency by Competitor')
        
        our_products = set(sales_df['productname'].unique())
        nahdi_products = set(competitor_df[competitor_df['competitor'] == 'Nahdi']['productname'].unique())
        dawaa_products = set(competitor_df[competitor_df['competitor'] == 'Al-Dawaa']['productname'].unique())
        
        venn_data = pd.DataFrame([
            {'sets': ['Ours Only'], 'size': len(our_products - nahdi_products - dawaa_products)},
            {'sets': ['Nahdi Only'], 'size': len(nahdi_products - our_products - dawaa_products)},
            {'sets': ['Al-Dawaa Only'], 'size': len(dawaa_products - our_products - nahdi_products)},
            {'sets': ['Ours & Nahdi'], 'size': len(our_products & nahdi_products - dawaa_products)},
            {'sets': ['Ours & Al-Dawaa'], 'size': len(our_products & dawaa_products - nahdi_products)},
            {'sets': ['Nahdi & Al-Dawaa'], 'size': len(nahdi_products & dawaa_products - our_products)},
            {'sets': ['All Three'], 'size': len(our_products & nahdi_products & dawaa_products)},
        ])
        assortment_fig = px.bar(venn_data, x='size', y='sets', orientation='h', title='Product Assortment Overlap')
        
        return kpi_advantage, kpi_disadvantage, kpi_promo, price_comp_fig, promo_fig, assortment_fig

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
        avg_roas = total_revenue / total_spend if total_spend > 0 else 0
        total_conversions = campaign_performance_df['conversions'].sum()
        avg_cpa = total_spend / total_conversions if total_conversions > 0 else 0
        
        kpi_spend = create_kpi_body("Total Ad Spend", f"{total_spend:,.2f} SAR")
        kpi_roas = create_kpi_body("Overall ROAS", f"{avg_roas:.2f}x")
        kpi_cpa = create_kpi_body("Average CPA", f"{avg_cpa:,.2f} SAR")
        kpi_conv = create_kpi_body("Attributed Conversions", f"{total_conversions:,.0f}")
        
        roas_fig = px.bar(campaign_performance_df, x='campaignname', y='roas', color='channel', title='ROAS by Campaign')
        cpa_fig = px.bar(campaign_performance_df, x='campaignname', y='cpa', color='channel', title='CPA by Campaign')
        conv_by_channel = campaign_performance_df.groupby('channel')['conversions'].sum().reset_index()
        conv_channel_fig = px.pie(conv_by_channel, names='channel', values='conversions', title='Conversions by Channel', hole=0.3)
        
        return kpi_spend, kpi_roas, kpi_cpa, kpi_conv, roas_fig, cpa_fig, conv_channel_fig

    @app.callback(
        [Output('kpi-total-net-profit', 'children'), Output('kpi-avg-profit-margin', 'children'),
         Output('kpi-profit-lost-returns', 'children'), Output('profit-by-channel-chart', 'figure'),
         Output('profit-by-category-chart', 'figure'), Output('high-margin-products-chart', 'figure'),
         Output('low-margin-products-chart', 'figure'), Output('automated-recommendations-list', 'children')],
        [Input('data-store-trigger', 'data'), Input('tabs-controller', 'active_tab')]
    )
    def update_profit_dashboard(_, active_tab):
        if active_tab != 'profit-tab': raise PreventUpdate
        profit_df = DATA.get('profit_df', pd.DataFrame())
        if profit_df.empty:
            placeholder = create_placeholder_figure("Profit Data Not Available")
            empty_kpi = create_kpi_body("No Data", "-")
            return [empty_kpi]*3 + [placeholder]*4 + [html.P("Not enough data.")]
            
        total_net_profit = profit_df['net_profit'].sum()
        avg_profit_margin = profit_df['profit_margin'].mean()
        returned_orders_df = profit_df[profit_df['orderstatus'] == 'Returned']
        profit_lost_to_returns = returned_orders_df['total_cost'].sum() + returned_orders_df['netsale'].sum()
        
        kpi_profit = create_kpi_body("Total Net Profit", f"{total_net_profit:,.2f} SAR")
        kpi_margin = create_kpi_body("Average Profit Margin", f"{avg_profit_margin:.2f}%")
        kpi_returns = create_kpi_body("Profit Lost to Returns", f"{profit_lost_to_returns:,.2f} SAR")
        
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

# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Central Configuration Module - V21.0 (Final Master)
#
# This module is the single source of truth for all application settings,
# database connection logic, and data schemas. It uses pydantic-settings
# for robust, type-safe configuration management.
# -----------------------------------------------------------------------------

import os
import re
import logging
import sys
from pydantic_settings import BaseSettings
from typing import Dict, List, Any

# --- 1. Pydantic Settings Class ---
class Settings(BaseSettings):
    """Defines all application settings and their types via environment variables."""
    DB_USER: str = "mohamedyousri"
    DB_PASSWORD: str = ""
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "pharma_db"
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

    @property
    def DATABASE_URL(self) -> str:
        """Constructs the final database URL, applying necessary fixes."""
        if 'DATABASE_URL' in os.environ:
            db_url = os.environ['DATABASE_URL']
            # Safely normalize postgres/postgresql prefixes
            db_url = re.sub(r'^postgres(?!\w)', 'postgresql', db_url)
            if "render.com" in db_url and "?sslmode=require" not in db_url:
                db_url += "?sslmode=require"
            return db_url
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

settings = Settings()

# --- 2. Centralized Data Schemas ---
SALES_SCHEMA_NORM: Dict[str, List[str]] = {
    'orderid': ['OrderID', 'order_id'], 'timestamp': ['Timestamp', 'order_date'], 
    'productid': ['ProductID', 'product_sku'], 'productname': ['ProductName'],
    'category': ['Category'], 'quantity': ['Quantity', 'qty'], 'grossvalue': ['GrossValue'], 
    'discountvalue': ['DiscountValue'], 'costofgoodssold': ['CostOfGoodsSold', 'cogs'], 
    'customerid': ['CustomerID', 'user_id'], 'city': ['City', 'store_city'], 
    'locationid': ['LocationID', 'store_id'], 'channel': ['Channel'], 'orderstatus': ['OrderStatus', 'status']
}
DELIVERY_SCHEMA_NORM: Dict[str, List[str]] = {
    'deliveryid': ['DeliveryID'], 'orderid': ['OrderID'], 'orderdate': ['OrderDate'], 
    'promiseddate': ['PromisedDate'], 'actualdeliverydate': ['ActualDeliveryDate'], 
    'status': ['Status'], 'deliverypartner': ['DeliveryPartner'], 'city': ['City'], 
    'deliverycost': ['DeliveryCost']
}
CUSTOMER_SCHEMA_NORM: Dict[str, List[str]] = { 'customerid': ['CustomerID'], 'joindate': ['JoinDate'], 'city': ['City'], 'segment': ['Segment'] }
FUNNEL_SCHEMA_NORM: Dict[str, List[str]] = { 'week': ['Week'], 'visits': ['Visits'], 'carts': ['Carts'], 'orders': ['Orders'] }
COMPETITOR_SCHEMA_NORM: Dict[str, List[str]] = { 'date': ['Date'], 'competitor': ['Competitor'], 'productid': ['ProductID'], 'productname': ['ProductName'], 'price': ['Price'], 'onpromotion': ['OnPromotion'] }
CAMPAIGN_SCHEMA_NORM: Dict[str, List[str]] = {
    'campaignid': ['CampaignID'], 'campaignname': ['CampaignName'], 'channel': ['Channel'], 
    'startdate': ['StartDate'], 'enddate': ['EndDate'], 'totalcost': ['TotalCost'], 
    'impressions': ['Impressions'], 'clicks': ['Clicks']
}
ATTRIBUTION_SCHEMA_NORM: Dict[str, List[str]] = { 'orderid': ['OrderID'], 'campaignid': ['CampaignID'] }

TABLE_CONFIG: Dict[str, Dict[str, Any]] = {
    "sales": {"schema_norm": SALES_SCHEMA_NORM, "filename": "sales_data.csv", "file_prefix": "sales_"},
    "deliveries": {"schema_norm": DELIVERY_SCHEMA_NORM, "filename": "delivery_data.csv", "file_prefix": "delivery_"},
    "customers": {"schema_norm": CUSTOMER_SCHEMA_NORM, "filename": "customer_data.csv", "file_prefix": "customer_"},
    "competitors": {"schema_norm": COMPETITOR_SCHEMA_NORM, "filename": "competitor_data.csv", "file_prefix": "competitor_"},
    "sales_funnel": {"schema_norm": FUNNEL_SCHEMA_NORM, "filename": "funnel_data.csv", "file_prefix": "funnel_"},
    "marketing_campaigns": {"schema_norm": CAMPAIGN_SCHEMA_NORM, "filename": "marketing_campaigns.csv", "file_prefix": "campaigns_"},
    "marketing_attribution": {"schema_norm": ATTRIBUTION_SCHEMA_NORM, "filename": "marketing_attribution.csv", "file_prefix": "attribution_"}
}

# Configure professional logging
# This ensures that any module importing this config file will have logging available.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.info(f"Configuration loaded. DB Target: {settings.DATABASE_URL.split('@')[-1]}")

# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Data Processing Engine - V21.0 (Final Master)
#
# This module is the central engine for all data loading, preparation, and
# enrichment. It is designed to be modular, resilient, and scalable.
# -----------------------------------------------------------------------------

import logging
import pandas as pd
from sqlalchemy.engine import Engine
from typing import Dict
from datetime import datetime, timedelta

# Import from our central modules
from database import refresh_all_data
from utils import safe_division

logger = logging.getLogger(__name__)

# This global dictionary will act as an in-memory data store for the app.
DATA: Dict[str, pd.DataFrame] = {}

# --- HELPER FUNCTIONS FOR DATA ENRICHMENT ---

def _enrich_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enriches the raw sales data with calculated columns for analysis."""
    logger.info("Enriching sales data...")
    if df.empty:
        return df
    
    # Use .get() for safer column access to prevent KeyErrors
    df['netsale'] = df.get('grossvalue', 0) - df.get('discountvalue', 0)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['week'] = df['timestamp'].dt.to_period('W').astype(str)
    df['month'] = df['timestamp'].dt.to_period('M').astype(str)
    return df

def _calculate_customer_segments(customers_df: pd.DataFrame, sales_df: pd.DataFrame) -> pd.DataFrame:
    """Performs RFM analysis and dynamically segments customers."""
    logger.info("Calculating customer segments...")
    if customers_df.empty or sales_df.empty:
        return pd.DataFrame()

    customers_df['joindate'] = pd.to_datetime(customers_df['joindate'])
    
    # FIX (F): Added missing .reset_index() to the RFM calculation
    rfm_df = sales_df.groupby('customerid').agg(
        last_purchase_date=('timestamp', 'max'),
        frequency=('orderid', 'nunique'),
        monetary=('netsale', 'sum')
    ).reset_index()
    
    current_date = datetime.now()
    rfm_df['recency'] = (current_date - rfm_df['last_purchase_date']).dt.days
    
    analysis_df = pd.merge(customers_df, rfm_df, on='customerid', how='left')

    def get_status(row):
        join_recency = (current_date - row['joindate']).days
        if join_recency <= 90: return 'New'
        if pd.isna(row['recency']): return 'Never Purchased'
        if row['recency'] <= 90: return 'Active'
        if 90 < row['recency'] <= 180: return 'Dormant (At-Risk)'
        return 'Churn Risk'

    analysis_df['status'] = analysis_df.apply(get_status, axis=1)
    return analysis_df

# --- MAIN INITIALIZATION FUNCTION ---

def initialize_data(engine: Engine) -> None:
    """
    Main orchestrator to load all raw data from the database and then call the
    various enrichment and transformation functions. Results are stored in the
    global DATA dictionary.
    """
    global DATA
    DATA = refresh_all_data(engine)

    # Sequentially enrich and create analysis dataframes
    if 'sales' in DATA:
        DATA['sales'] = _enrich_sales_data(DATA.get('sales', pd.DataFrame()))
    
    if 'customers' in DATA and 'sales' in DATA:
        DATA['customer_analysis_df'] = _calculate_customer_segments(
            DATA.get('customers', pd.DataFrame()),
            DATA.get('sales', pd.DataFrame())
        )
    
    logger.info("Data enrichment complete.")

# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Secure Data Access Layer - V21.0 (Final Master)
#
# This module handles all direct database interactions. It uses the central
# config and SQLAlchemy's Inspector for safe, robust, and secure data access.
# -----------------------------------------------------------------------------

import logging
import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from typing import Dict

# Import from our central, single source of truth
from config import settings, TABLE_CONFIG

logger = logging.getLogger(__name__)

# This global variable will hold our single, cached engine instance
# to ensure efficient connection pooling.
_engine = None

def get_engine() -> Engine:
    """
    Creates and returns a single, cached SQLAlchemy engine instance.
    This prevents creating new connections for every request.
    """
    global _engine
    if _engine is None:
        db_url = settings.DATABASE_URL
        logger.info(f"Creating new database engine for host: {db_url.split('@')[-1]}")
        _engine = create_engine(db_url)
        # Test the connection on creation
        try:
            with _engine.connect() as connection:
                logger.info("Database engine created and connection successful.")
        except Exception as e:
            logger.critical(f"Database connection failed on initial creation: {e}", exc_info=True)
            raise
    return _engine

def safe_table_exists(engine: Engine, table_name: str) -> bool:
    """Securely checks if a table exists using the SQLAlchemy Inspector."""
    try:
        inspector = inspect(engine)
        return table_name in inspector.get_table_names()
    except Exception as e:
        logger.error(f"Failed to inspect table existence for '{table_name}': {e}", exc_info=True)
        return False

def load_data_safely(table_name: str, engine: Engine) -> pd.DataFrame:
    """
    Securely loads a full table into a pandas DataFrame. It validates the
    table name against a predefined list to prevent SQL injection.
    """
    # FIX (C): Validate table_name against an allowed list
    if table_name not in TABLE_CONFIG:
        logger.error(f"[SECURITY] Attempted to load non-whitelisted table: '{table_name}'")
        return pd.DataFrame()

    try:
        if not safe_table_exists(engine, table_name):
            logger.warning(f"Table '{table_name}' not found. Returning empty DataFrame.")
            return pd.DataFrame()

        # Using read_sql_table is safer than f-strings
        df = pd.read_sql_table(table_name, engine)
        df.columns = [col.lower() for col in df.columns]
        logger.info(f"Successfully loaded {len(df)} rows from table '{table_name}'.")
        return df
    except Exception as e:
        logger.error(f"Could not load table '{table_name}'. Error: {e}", exc_info=True)
        return pd.DataFrame()

def refresh_all_data(engine: Engine) -> Dict[str, pd.DataFrame]:
    """Loads all tables defined in the config into a dictionary of DataFrames."""
    logger.info("--- Refreshing all data from database ---")
    dataframes = {}
    for table_name in TABLE_CONFIG.keys():
        dataframes[table_name] = load_data_safely(table_name, engine)
    logger.info("--- Data refresh complete ---")
    return dataframes

# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# UI Layouts Module - V22.0 (Modern UI Redesign)
#
# This module defines the visual structure of the application, focusing on a
# professional, modern, and clean aesthetic using a new theme and better
# component organization. This is the full, unabbreviated version.
# -----------------------------------------------------------------------------

import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
import pandas as pd

# Import the global data store to build dynamic components like dropdowns
from data import DATA

# --- 1. REUSABLE UI COMPONENTS ---

def create_kpi_card(title: str, kpi_id: str, color: str, width: int = 4, md_width: int = 6) -> dbc.Col:
    """Creates a Bootstrap Column containing a KPI Card."""
    return dbc.Col(dbc.Card(id=kpi_id, color=color, inverse=True), lg=width, md=md_width, sm=12, class_name="mb-4")

def create_graph_card(graph_id: str, width: int = 6) -> dbc.Col:
    """Creates a Bootstrap Column containing a Graph component in a Card."""
    return dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id=graph_id))), md=width, class_name="mb-4")

# --- 2. DASHBOARD LAYOUT FUNCTIONS ---

def create_sales_layout() -> dbc.Container:
    """Creates the layout for the Sales Command Center."""
    sales_df = DATA.get('sales', pd.DataFrame())
    if sales_df.empty:
        return dbc.Container(html.H4("Sales Data Not Available", className="text-center mt-5"))
    
    return dbc.Container([
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col(dcc.Dropdown(id='channel-filter-dropdown', options=[{'label': 'All Channels', 'value': 'All'}] + [{'label': ch, 'value': ch} for ch in sorted(sales_df['channel'].unique())], value='All', clearable=False), md=5),
                dbc.Col(dcc.DatePickerRange(id='sales-date-picker', min_date_allowed=sales_df['date'].min(), max_date_allowed=sales_df['date'].max(), start_date=sales_df['date'].min(), end_date=sales_df['date'].max()), md=5),
                dbc.Col(dcc.RadioItems(id='time-agg-selector', options=[{'label': 'Daily', 'value': 'date'}, {'label': 'Weekly', 'value': 'week'}, {'label': 'Monthly', 'value': 'month'}], value='date', inline=True), md=2),
            ], align="center"),
        ]), className="mb-4"),
        dbc.Row([
            create_kpi_card("Total Revenue", "kpi-total-revenue", "primary"),
            create_kpi_card("Gross Margin", "kpi-gross-margin", "success"),
            create_kpi_card("Net Profit", "kpi-net-profit", "dark"),
        ]),
        dbc.Row([
            create_kpi_card("Total Orders", "kpi-total-orders", "info"),
            create_kpi_card("Avg Order Value", "kpi-aov", "secondary"),
            create_kpi_card("Return Rate", "kpi-return-rate", "danger"),
        ]),
        dbc.Row([create_graph_card('sales-funnel-chart', width=12)]),
        dbc.Row([create_graph_card('sales-over-time-chart', width=12)]),
        dbc.Row([create_graph_card('sales-by-category-chart'), create_graph_card('top-products-chart')]),
        dbc.Row([create_graph_card('sales-by-channel-chart'), create_graph_card('sales-by-city-chart')]),
        dbc.Row([create_graph_card('sales-by-branch-chart', width=12)]),
    ], fluid=True)

def create_delivery_layout() -> dbc.Container:
    """Creates the layout for the Logistics Command Center."""
    delivery_df = DATA.get('deliveries', pd.DataFrame())
    if delivery_df.empty:
        return dbc.Container(html.H4("Delivery Data Not Available", className="text-center mt-5"))
    return dbc.Container([
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col(dcc.Dropdown(id='delivery-partner-filter', options=[{'label': 'All Partners', 'value': 'All'}] + [{'label': p, 'value': p} for p in sorted(delivery_df['deliverypartner'].unique())], value='All', clearable=False), md=6),
                dbc.Col(dcc.DatePickerRange(id='delivery-date-picker', min_date_allowed=delivery_df['date'].min(), max_date_allowed=delivery_df['date'].max(), start_date=delivery_df['date'].min(), end_date=delivery_df['date'].max()), md=6),
            ]),
        ]), className="mb-4"),
        dbc.Row([
            create_kpi_card("On-Time Rate", "kpi-on-time-delivery", "success", width=3, md_width=6),
            create_kpi_card("Failed Delivery Rate", "kpi-failed-delivery", "danger", width=3, md_width=6),
            create_kpi_card("Avg. Delivery Time", "kpi-avg-delivery-time", "warning", width=3, md_width=6),
            create_kpi_card("Avg. Cost per Delivery", "kpi-avg-delivery-cost", "info", width=3, md_width=6),
        ]),
        dbc.Row([create_graph_card('delivery-pipeline-chart', width=12)]),
        dbc.Row([create_graph_card('avg-time-by-city-chart'), create_graph_card('partner-performance-chart')]),
    ], fluid=True)

def create_customer_layout() -> dbc.Container:
    """Creates the layout for the Customer Action Center."""
    customer_analysis_df = DATA.get('customer_analysis_df', pd.DataFrame())
    if customer_analysis_df.empty:
        return dbc.Container(html.H4("Customer or Sales Data Not Available", className="text-center mt-5"))
    return dbc.Container([
        dbc.Row([
            create_kpi_card("Total Customers", "kpi-total-customers", "primary", width=3, md_width=6),
            create_kpi_card("Active Customers", "kpi-active-customers", "success", width=3, md_width=6),
            create_kpi_card("Dormant Customers", "kpi-dormant-customers", "warning", width=3, md_width=6),
            create_kpi_card("High Churn Risk", "kpi-churn-risk", "danger", width=3, md_width=6),
        ]),
        dbc.Row([create_graph_card('customer-status-dist-chart', width=12)]),
        html.Hr(className="my-4"),
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col(html.H4("Actionable Customer Lists"), md=6),
                dbc.Col(dcc.RadioItems(id='customer-list-selector', options=[{'label': 'Top-Value Customers', 'value': 'top_value'}, {'label': 'High Churn Risk', 'value': 'churn_risk'}, {'label': 'New Customers', 'value': 'new'}], value='top_value', inline=True, labelClassName="me-3"), md=6),
            ], align="center"),
            dbc.Row([
                dbc.Col(dash_table.DataTable(id='customer-data-table', style_cell={'textAlign': 'left'}, style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'}, page_size=10), width=10, className="mt-3"),
                dbc.Col(dbc.Button(["Export ", html.I(className="bi bi-download")], id="export-csv-button", color="primary", className="mt-3 w-100"), width=2),
            ]),
        ])),
    ], fluid=True)

def create_competitor_layout() -> dbc.Container:
    """Creates the layout for the Market Intelligence dashboard."""
    competitor_df = DATA.get('competitors', pd.DataFrame())
    price_comparison_df = DATA.get('price_comparison_df', pd.DataFrame())
    if competitor_df.empty or price_comparison_df.empty:
        return dbc.Container(html.H4("Competitor or Sales Data Not Available", className="text-center mt-5"))
    return dbc.Container([
        dbc.Row([
            create_kpi_card("Products We Undercut", "kpi-price-advantage", "success"),
            create_kpi_card("Products More Expensive", "kpi-price-disadvantage", "danger"),
            create_kpi_card("Avg. Competitor Promo Rate", "kpi-promo-frequency", "info"),
        ]),
        dbc.Row([create_graph_card('price-comparison-scatter-chart', width=12)]),
        dbc.Row([create_graph_card('promo-analysis-chart'), create_graph_card('assortment-overlap-chart')]),
    ], fluid=True)

def create_marketing_layout() -> dbc.Container:
    """Creates the layout for the Marketing Effectiveness dashboard."""
    campaign_performance_df = DATA.get('campaign_performance_df', pd.DataFrame())
    if campaign_performance_df.empty:
        return dbc.Container(html.H4("Marketing Campaign Data Not Available", className="text-center mt-5"))
    return dbc.Container([
        dbc.Row([
            create_kpi_card("Total Ad Spend", "kpi-total-ad-spend", "info", width=3, md_width=6),
            create_kpi_card("Overall ROAS", "kpi-avg-roas", "success", width=3, md_width=6),
            create_kpi_card("Average CPA", "kpi-avg-cpa", "warning", width=3, md_width=6),
            create_kpi_card("Attributed Conversions", "kpi-total-conversions", "primary", width=3, md_width=6),
        ]),
        dbc.Row([create_graph_card('roas-by-campaign-chart', width=12)]),
        dbc.Row([create_graph_card('cpa-by-campaign-chart'), create_graph_card('conversions-by-channel-chart')]),
    ], fluid=True)

def create_profit_layout() -> dbc.Container:
    """Creates the layout for the Profit Optimization dashboard."""
    profit_df = DATA.get('profit_df', pd.DataFrame())
    if profit_df.empty:
        return dbc.Container(html.H4("Profit calculation requires Sales, Delivery, and Marketing data.", className="text-center mt-5"))
    return dbc.Container([
        dbc.Row([
            create_kpi_card("Total Net Profit", "kpi-total-net-profit", "success"),
            create_kpi_card("Average Profit Margin", "kpi-avg-profit-margin", "primary"),
            create_kpi_card("Profit Lost to Returns", "kpi-profit-lost-returns", "danger"),
        ]),
        dbc.Row([
            dbc.Col([html.H4("Key Profit Drivers", className="mb-3"), create_graph_card('profit-by-channel-chart', width=12), create_graph_card('profit-by-category-chart', width=12)], md=6),
            dbc.Col([html.H4("Actionable Recommendations", className="mb-3"), dbc.Card(dbc.CardBody(id='automated-recommendations-list'), style={"height": "95%"})], md=6),
        ]),
        html.Hr(className="my-4"),
        dbc.Row([create_graph_card('high-margin-products-chart'), create_graph_card('low-margin-products-chart')]),
    ], fluid=True)

def create_predictive_layout() -> dbc.Container:
    """Creates the layout for the Predictive Insights dashboard."""
    predictions_df = DATA.get('predictions_df', pd.DataFrame())
    if predictions_df.empty:
        return dbc.Container([
            html.H4("Churn Prediction Model Not Available", className="text-center mt-5"),
            html.P("Please run the 'model_trainer.py' script to enable this feature.", className="text-center")
        ])
    return dbc.Container([
        dbc.Row([
            create_kpi_card("High Risk (>70%)", "kpi-high-risk-customers", "danger"),
            create_kpi_card("Medium Risk (40-70%)", "kpi-med-risk-customers", "warning"),
            create_kpi_card("Low Risk (<40%)", "kpi-low-risk-customers", "success"),
        ]),
        dbc.Row([create_graph_card('churn-risk-distribution-chart', width=12)]),
        html.Hr(className="my-4"),
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col(html.H4("High-Risk Customer List for Retention Campaign"), width=10),
                dbc.Col(dbc.Button(["Export List ", html.I(className="bi bi-download")], id="export-churn-button", color="secondary"), width=2),
            ]),
            dbc.Row([
                dbc.Col(dash_table.DataTable(id='churn-data-table', style_cell={'textAlign': 'left'}, style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'}, page_size=10, sort_action='native'), className="mt-3")
            ]),
        ])),
    ], fluid=True)

# --- 3. MAIN APPLICATION LAYOUT ---
def create_main_layout() -> html.Div:
    """Creates the main application layout, including the new Navbar and tabs."""
    navbar = dbc.NavbarSimple(
        children=[
            dbc.Button(
                ["Refresh Data ", html.I(className="bi bi-arrow-clockwise")],
                id="refresh-data-button", color="secondary", className="ms-auto"
            ),
        ],
        brand="Pharma Analytics Hub", brand_href="#", color="primary", dark=True, className="mb-4"
    )
    return html.Div([
        dcc.Store(id='data-store-trigger'),
        dcc.Download(id="download-dataframe-csv"),
        navbar,
        dbc.Container([
            dbc.Tabs(id="tabs-controller", active_tab="sales-tab", children=[
                dbc.Tab(label="Sales", tab_id="sales-tab"),
                dbc.Tab(label="Logistics", tab_id="delivery-tab"),
                dbc.Tab(label="Customers", tab_id="customer-tab"),
                dbc.Tab(label="Market Intel", tab_id="competitor-tab"),
                dbc.Tab(label="Marketing", tab_id="marketing-tab"),
                dbc.Tab(label="Profit Optimization", tab_id="profit-tab"),
                dbc.Tab(label="Predictive Insights", tab_id="predictive-tab"),
            ]),
            html.Div(id='tab-content', className="mt-4")
        ], fluid=True)
    ])

# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Utilities Toolbox - V21.0 (Final Master)
#
# This module provides a set of clean, well-documented, and reusable utility
# functions for the application (e.g., math, UI helpers).
# -----------------------------------------------------------------------------

import logging
from typing import Dict, Any, Union
import dash_bootstrap_components as dbc
from dash import html

logger = logging.getLogger(__name__)

def safe_division(numerator: Union[int, float], denominator: Union[int, float]) -> float:
    """
    Performs division safely, returning 0.0 if the denominator is zero.
    This prevents "division by zero" errors in KPI calculations.

    Args:
        numerator: The number to be divided.
        denominator: The number to divide by.

    Returns:
        The result of the division, or 0.0 if an error occurs.
    """
    if denominator is None or denominator == 0 or numerator is None:
        return 0.0
    try:
        return float(numerator / denominator)
    except (ValueError, TypeError):
        return 0.0


def create_placeholder_figure(message: str = "No data available") -> Dict[str, Any]:
    """
    Creates a blank Plotly figure with a custom message.
    Used as a placeholder when data for a chart is missing or empty.

    Args:
        message: The text to display on the empty chart.

    Returns:
        A dictionary representing a blank Plotly figure layout.
    """
    logger.debug(f"Creating placeholder figure with message: '{message}'")
    return {
        "layout": {
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "annotations": [{
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 16}
            }]
        }
    }


def create_kpi_body(title: str, value: str) -> dbc.CardBody:
    """
    Creates a consistent CardBody for a Key Performance Indicator (KPI).
    This reusable component ensures all KPIs have the same style and structure.

    Args:
        title: The title of the KPI (e.g., "Total Revenue").
        value: The formatted string value of the KPI (e.g., "1,234.56 SAR").

    Returns:
        A dash_bootstrap_components.CardBody object.
    """
    return dbc.CardBody([
        html.H4(title, className="card-title"),
        html.P(value, className="card-text fs-3")
    ])

# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# E-commerce Data Loader - V21.0 (Final Master)
#
# This script is the master setup tool for the database. It reads all local
# master CSV files and completely rebuilds the database tables. It also
# provides a function for the scheduler to process incremental files.
# -----------------------------------------------------------------------------

import pandas as pd
import sys
import os
import logging

# Import from our central, single-source-of-truth modules
from config import TABLE_CONFIG
from database import get_engine

# Configure professional logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

def normalize_headers(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """
    Case-insensitively renames DataFrame columns based on the schema.
    This makes the loader resilient to minor changes in CSV header casing.
    """
    header_map = {}
    cols_lower = {c.lower(): c for c in df.columns}
    for clean_name, possible_names in schema.items():
        for pname in possible_names:
            p_low = pname.lower()
            if p_low in cols_lower:
                header_map[cols_lower[p_low]] = clean_name
                break
    
    df = df.rename(columns=header_map)
    # This debug log is helpful for verifying header mapping
    logging.debug(f"Normalized headers for {list(schema.keys())[0]}: {df.columns.tolist()}")
    return df

def bootstrap_database(engine):
    """
    Loads all master CSV files from the main directory, completely replacing
    all tables in the database. This is for initial setup or a full reset.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for table_name, config in TABLE_CONFIG.items():
        logger.info(f"--- Bootstrapping table: {table_name} ---")
        file_path = os.path.join(base_dir, config['filename'])
        try:
            df = pd.read_csv(file_path)
            df = normalize_headers(df, config['schema_norm'])
            
            # Add calculated columns after cleaning headers
            if 'grossvalue' in df.columns and 'discountvalue' in df.columns:
                df['netsale'] = df['grossvalue'] - df['discountvalue']
            
            # if_exists='replace' will drop the table first if it exists
            df.to_sql(table_name, engine, if_exists='replace', index=False)
            logger.info(f"  [SUCCESS] Table '{table_name}' created with {len(df)} rows.")
        except FileNotFoundError:
            logger.error(f"Master file not found: {config['filename']}. Skipping.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while processing {config['filename']}: {e}", exc_info=True)
            raise  # Stop the entire bootstrap process if any file fails

def process_incoming_file_and_append(filepath: str, engine) -> bool:
    """
    Processes a single incoming file and appends it to the database.
    Returns True on success and False on failure.
    """
    logger.info(f"--- Processing incoming file: {filepath} ---")
    filename = os.path.basename(filepath).lower()
    table_name = None
    
    # Determine which table this file belongs to based on its prefix
    for name, config in TABLE_CONFIG.items():
        if filename.startswith(config.get('file_prefix', '')):
            table_name = name
            break
            
    if not table_name:
        logger.warning(f"Unrecognized file prefix for '{filename}'. Skipping.")
        return False

    try:
        df = pd.read_csv(filepath)
        df = normalize_headers(df, TABLE_CONFIG[table_name]['schema_norm'])
        
        if 'grossvalue' in df.columns and 'discountvalue' in df.columns:
            df['netsale'] = df['grossvalue'] - df['discountvalue']
        
        # Use if_exists='append' to add new data without deleting old data
        df.to_sql(table_name, engine, if_exists='append', index=False)
        logger.info(f"  [SUCCESS] Appended {len(df)} rows to '{table_name}'.")
        return True # Return True on success for archiving
    except Exception as e:
        logger.error(f"Failed to process and append file '{filepath}'. Error: {e}", exc_info=True)
        return False # Return False on failure

if __name__ == "__main__":
    logger.info("--- Running Database Bootstrap Tool v21.0 ---")
    try:
        engine = get_engine()
        # The get_engine function already tests the connection
        bootstrap_database(engine)
        logger.info("\n--- Database bootstrap process finished successfully ---")
    except Exception as e:
        logger.critical(f"\n--- Bootstrap failed. Error: {e}", exc_info=True)
        sys.exit(1)

# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Automated Data Pipeline Scheduler - V21.0 (Final Master)
#
# This script is the automated "factory" that runs in the background,
# processing new data files from the 'incoming_data' folder on a schedule.
# It is now fully integrated with the new modular project structure.
# -----------------------------------------------------------------------------

import logging
import os
import shutil
import sys
import signal
from apscheduler.schedulers.blocking import BlockingScheduler

# Import from our central, single-source-of-truth modules
from load_data import process_incoming_file_and_append
from database import get_engine

# Configure professional logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
INCOMING_DIR = "incoming_data"
ARCHIVE_DIR = "archive"

def process_new_files_job():
    """
    The main job that the scheduler will run. It finds CSV files in the
    incoming directory, processes them, and archives them on success.
    """
    logger.info("--- Scheduler waking up, checking for new files... ---")
    try:
        engine = get_engine()
        # Ensure directories exist
        if not os.path.exists(INCOMING_DIR):
            logger.warning(f"'{INCOMING_DIR}' not found. Creating it.")
            os.makedirs(INCOMING_DIR)
        if not os.path.exists(ARCHIVE_DIR):
            logger.warning(f"'{ARCHIVE_DIR}' not found. Creating it.")
            os.makedirs(ARCHIVE_DIR)

        files_to_process = [f for f in os.listdir(INCOMING_DIR) if f.endswith('.csv')]

        if not files_to_process:
            logger.info("No new files found.")
            return

        logger.info(f"Found {len(files_to_process)} new file(s): {files_to_process}")
        for filename in files_to_process:
            filepath = os.path.join(INCOMING_DIR, filename)
            # Call the processing function from our central ETL engine
            success = process_incoming_file_and_append(filepath, engine)
            
            if success:
                archive_path = os.path.join(ARCHIVE_DIR, filename)
                shutil.move(filepath, archive_path)
                logger.info(f"  [ARCHIVED] Moved '{filename}' to '{ARCHIVE_DIR}'.")
            else:
                logger.error(f"  [ERROR] Failed to process '{filename}'. It will remain in the incoming folder for review.")

    except Exception as e:
        logger.error(f"  [FATAL] The scheduler job failed unexpectedly. Error: {e}", exc_info=True)

if __name__ == "__main__":
    logger.info("--- Starting Automated Data Pipeline Scheduler v21.0 ---")
    
    scheduler = BlockingScheduler()
    scheduler.add_job(process_new_files_job, 'interval', minutes=1, id='process_incoming_files')
    
    # --- Graceful Shutdown Handling ---
    def shutdown(signum, frame):
        logger.warning(f"Shutdown signal {signum} received. Shutting down scheduler...")
        scheduler.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info(f"Scheduler is now watching '{INCOMING_DIR}'. Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped by user.")

