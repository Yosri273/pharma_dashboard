from app.comprehensive_analysis.layout import get_comprehensive_layout
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
import pandas as pd
from etl.transforms import DATA
from app.utils.ui_helpers import create_filter_options, create_multi_filter_options, create_kpi_card, create_graph_card, create_datatable_card

# --- Dashboard Layouts ---
# ... (create_sales_layout is unchanged) ...
def create_sales_layout() -> dbc.Container:
    sales_df = DATA.get('sales', pd.DataFrame())
    if sales_df.empty: return dbc.Container(html.H4("Sales Data Not Available", className="text-center mt-5"))
    channel_opts, category_opts, region_opts = sales_df['channel'].unique(), sales_df['category'].unique(), sales_df['city'].unique()
    # For product selector, limit to top N unique names to avoid huge option lists in the DOM
    TOP_PRODUCT_OPTIONS = 200
    if 'productname' in sales_df.columns:
        unique_products = list(sales_df['productname'].unique())
        product_opts = unique_products[:TOP_PRODUCT_OPTIONS]
    else:
        product_opts = []
    branch_opts = sales_df['LocationID'].unique() if 'LocationID' in sales_df.columns else (sales_df['locationid'].unique() if 'locationid' in sales_df.columns else [])
    return dbc.Container([
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([html.Label("Date Range:"), dcc.DatePickerRange(id='sales-date-picker', min_date_allowed=sales_df['date'].min(), max_date_allowed=sales_df['date'].max(), start_date=sales_df['date'].min(), end_date=sales_df['date'].max(), className="d-block")], width=12, lg=3, className="mb-2"),
                dbc.Col([html.Label("Region (City):"), dcc.Dropdown(id='sales-region-filter', options=create_multi_filter_options(region_opts), value=['All'], multi=True, clearable=False)], width=12, lg=2, className="mb-2"),
                dbc.Col([html.Label("Category:"), dcc.Dropdown(id='sales-category-filter', options=create_multi_filter_options(category_opts), value=['All'], multi=True, clearable=False)], width=12, lg=2, className="mb-2"),
                dbc.Col([html.Label("Channel:"), dcc.Dropdown(id='channel-filter-dropdown', options=create_filter_options(channel_opts), value='All', clearable=False)], width=12, lg=2, className="mb-2"),
                dbc.Col([html.Label("Aggregate:"), dcc.RadioItems(id='time-agg-selector', options=[{'label': 'Daily', 'value': 'date'}, {'label': 'Weekly', 'value': 'week'}, {'label': 'Monthly', 'value': 'month'}], value='date', inline=True)], width=12, lg=1, className="mb-2 align-self-center"),
                dbc.Col([html.Label("Actions", style={'visibility': 'hidden'}), dbc.ButtonGroup([dbc.Button("Apply", id="sales-apply-btn", color="primary"), dbc.Button("Export PDF", id="sales-export-btn", color="secondary")], className="w-100")], width=12, lg=2, className="mb-2 align-self-end"),
            ], align="bottom"),
            # Additional filters row: Product & Branch (kept separate to avoid crowding controls)
            dbc.Row([
                dbc.Col([html.Label("Product:"), dcc.Dropdown(id='sales-product-filter', options=create_multi_filter_options(product_opts), value=['All'], multi=True, clearable=False, searchable=True)], width=12, lg=6, className="mb-2"),
                dbc.Col([html.Label("Branch:"), dcc.Dropdown(id='sales-branch-filter', options=create_multi_filter_options(branch_opts), value=['All'], multi=True, clearable=False)], width=12, lg=6, className="mb-2"),
            ], align="bottom"),
        ]), className="mb-4"),
    dbc.Row([create_kpi_card("Total Revenue", "kpi_total_revenue", "primary"), create_kpi_card("Gross Margin", "kpi_gross_margin", "success"), create_kpi_card("Net Profit", "kpi_net_profit", "dark")]),
    dbc.Row([create_kpi_card("Total Orders", "kpi_total_orders", "info"), create_kpi_card("Avg Order Value", "kpi_aov", "secondary"), create_kpi_card("Return Rate", "kpi_return_rate", "danger")]),
    # Recommendations placed immediately below KPI cards for faster action
    dbc.Row([dbc.Col([
        html.H4("Actionable Recommendations", className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Dropdown(id='sales-rec-severity-filter', options=[{'label':'All','value':'all'},{'label':'Critical','value':'critical'},{'label':'Warning','value':'warning'},{'label':'Info','value':'info'}], value='all', clearable=False, style={'width':'220px','marginBottom':'8px'}), width=12, lg=3),
            dbc.Col(dbc.ButtonGroup([
                dbc.Button("Edit Thresholds", id="sales-edit-thresholds-btn", color="light"),
                dbc.Button("Refresh Recommendations", id="sales-refresh-recs-btn", color="primary", outline=True),
            ]), width=12, lg=5)
        ]),
    dbc.Card(dbc.CardBody(id='sales-recommendations-list'), style={"minHeight": "120px"}),
    dcc.Store(id='sales-thresholds-saved-signal', data=0),
    dbc.Toast(id='sales-toast', header='Sales', is_open=False, duration=3000, icon='success', children='Saved thresholds.', style={'position':'fixed','top':70,'right':20,'zIndex':1050}),
        # Sales thresholds modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Sales Thresholds")),
            dbc.ModalBody(
                dbc.Row([
                    dbc.Col([
                        html.H6('Common thresholds'),
                        dbc.Label('Conversion warning'), dcc.Input(id='sales_conversion_rate_warning_input', type='number', step='0.001', placeholder='e.g. 0.05', style={'width':'100%', 'color':'#111'}),
                        dbc.Label('Conversion critical'), dcc.Input(id='sales_conversion_rate_critical_input', type='number', step='0.001', placeholder='e.g. 0.02', style={'width':'100%', 'color':'#111'}),
                        dbc.Label('Gross margin warning'), dcc.Input(id='sales_gross_margin_warning_input', type='number', step='0.01', placeholder='e.g. 0.20', style={'width':'100%', 'color':'#111'}),
                        dbc.Label('Gross margin critical'), dcc.Input(id='sales_gross_margin_critical_input', type='number', step='0.01', placeholder='e.g. 0.10', style={'width':'100%', 'color':'#111'}),
                        dbc.Label('AOV warning'), dcc.Input(id='sales_aov_warning_input', type='number', step='0.1', placeholder='e.g. 35.0', style={'width':'100%', 'color':'#111'}),
                        dbc.Label('AOV info'), dcc.Input(id='sales_aov_info_input', type='number', step='0.1', placeholder='e.g. 20.0', style={'width':'100%', 'color':'#111'}),
                        dbc.Label('Return rate warning'), dcc.Input(id='sales_return_rate_warning_input', type='number', step='0.01', placeholder='e.g. 0.08', style={'width':'100%', 'color':'#111'}),
                    ], md=5),
                    dbc.Col([
                        html.H6('Advanced / Raw JSON'),
                        dcc.Textarea(id='sales-thresholds-json-textarea', style={'width':'100%','minHeight':'200px','fontFamily':'monospace', 'color':'#111'}),
                        html.Hr(),
                        html.H6('All KPI Thresholds (editable table)'),
                        dash_table.DataTable(id='sales-thresholds-datatable', columns=[{'name': 'kpi','id':'kpi','type':'text','editable':False},{'name': 'info','id':'info','type':'numeric','editable':True},{'name': 'warning','id':'warning','type':'numeric','editable':True},{'name': 'critical','id':'critical','type':'numeric','editable':True}], data=[], editable=True, style_table={'height':'200px','overflowY':'auto'}, style_cell={'textAlign':'left','minWidth':'120px','width':'160px','maxWidth':'260px','color':'#111'}, style_header={'fontWeight':'bold','color':'#111','backgroundColor':'#f7f7f7'})
                    ], md=7)
                ])
            ),
            dbc.ModalFooter([
                dbc.Button("Auto-fill", id='sales-thresholds-autofill-btn', color='secondary', class_name='me-2'),
                dbc.Button("Load", id='sales-thresholds-load-btn', color='secondary', class_name='me-2'),
                dbc.Button("Save", id='sales-thresholds-save-btn', color='primary', class_name='me-2'),
                dbc.Button("Close", id='sales-thresholds-close-btn', color='light'),
                html.Span(id='sales-thresholds-save-feedback', style={'marginLeft':'12px'})
            ])
        ], id='sales-thresholds-modal', size='lg')
    ], width=12)]),
    dbc.Row([create_graph_card('sales-funnel-chart', width=12)]),
        dbc.Row([create_graph_card('sales-over-time-chart', width=12)]),
        html.Hr(className="my-4"), dbc.Row([html.H4("Advanced Sales Analytics", className="mb-3 text-center")]), dbc.Row([create_graph_card('period-growth-chart', title="Period-over-Period Growth"), create_graph_card('price-volume-chart', title="Price vs. Volume Analysis")]), html.Hr(className="my-4"),
    dbc.Row([create_graph_card('sales-by-category-chart'), create_graph_card('top-products-chart')]), dbc.Row([create_graph_card('sales-by-channel-chart'), create_graph_card('sales-by-city-chart')]), dbc.Row([create_graph_card('sales-by-branch-chart', width=12)]),
    
    ], fluid=True)


# --- MODIFIED: LOGISTICS TAB LAYOUT ---
def create_delivery_layout() -> dbc.Container:
    """Creates the layout for the Logistics Command Center (Internal Fleet)."""
    delivery_df = DATA.get('deliveries', pd.DataFrame())
    if delivery_df.empty:
        return dbc.Container(html.H4("Delivery Data Not Available", className="text-center mt-5"))
        
    # Get dynamic options for new filters from the dataframe
    driver_opts = delivery_df['driverid'].unique() if 'driverid' in delivery_df.columns else []
    vehicle_opts = delivery_df['vehicletype'].unique() if 'vehicletype' in delivery_df.columns else []
    region_opts = delivery_df['city'].unique()
    
    return dbc.Container([
        dbc.Card(dbc.CardBody([
            dbc.Row([
                 dbc.Col([
                    html.Label("Date Range:"),
                    dcc.DatePickerRange(
                        id='delivery-date-picker',
                        min_date_allowed=delivery_df['date'].min(), max_date_allowed=delivery_df['date'].max(),
                        start_date=delivery_df['date'].min(), end_date=delivery_df['date'].max(), className="d-block"
                    )
                ], width=12, lg=3, className="mb-2"),
                dbc.Col([
                    html.Label("Region (City):"),
                    dcc.Dropdown(id='delivery-region-filter', options=create_multi_filter_options(region_opts), value=['All'], multi=True, clearable=False)
                ], width=12, lg=2, className="mb-2"),
                 dbc.Col([
                    html.Label("Driver:"),
                     dcc.Dropdown(id='driver-filter', options=create_filter_options(driver_opts), value='All', clearable=False) # NEW
                ], width=12, lg=2, className="mb-2"),
                dbc.Col([
                    html.Label("Vehicle Type:"),
                     dcc.Dropdown(id='vehicle-type-filter', options=create_filter_options(vehicle_opts), value='All', clearable=False) # NEW
                ], width=12, lg=2, className="mb-2"),
                dbc.Col([
                    html.Label("Actions", style={'visibility': 'hidden'}),
                    dbc.ButtonGroup([
                        dbc.Button("Apply", id="delivery-apply-btn", color="primary"),
                        dbc.Button("Export PDF", id="delivery-export-btn", color="secondary")
                    ], className="w-100")
                ], width=12, lg=3, className="mb-2 align-self-end"),
            ], align="bottom"),
        ]), className="mb-4"),
        
        dbc.Row([
            create_kpi_card("On-Time Rate", "kpi-on-time-delivery", "success", width=3, md_width=6),
            create_kpi_card("Failed Delivery Rate", "kpi-failed-delivery", "danger", width=3, md_width=6),
            create_kpi_card("Avg. Delivery Time", "kpi-avg-delivery-time", "warning", width=3, md_width=6),
            create_kpi_card("Avg. Cost per Delivery", "kpi-avg-delivery-cost", "info", width=3, md_width=6),
        ]),

        # Recommendations directly under Delivery KPIs
        dbc.Row([dbc.Col([
            html.H4("Actionable Recommendations", className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Dropdown(id='delivery-rec-severity-filter', options=[{'label':'All','value':'all'},{'label':'Critical','value':'critical'},{'label':'Warning','value':'warning'},{'label':'Info','value':'info'}], value='all', clearable=False, style={'width':'220px','marginBottom':'8px'}), width=12, lg=3),
                dbc.Col(dbc.ButtonGroup([
                    dbc.Button("Edit Thresholds", id="delivery-edit-thresholds-btn", color="light"),
                    dbc.Button("Refresh Recommendations", id="delivery-refresh-recs-btn", color="primary", outline=True),
                ]), width=12, lg=5)
            ]),
            dbc.Card(dbc.CardBody(id='delivery-recommendations-list'), style={"minHeight": "120px"}),
            dcc.Store(id='delivery-thresholds-saved-signal', data=0),
            dbc.Toast(id='delivery-toast', header='Logistics', is_open=False, duration=3000, icon='success', children='Saved thresholds.', style={'position':'fixed','top':70,'right':20,'zIndex':1050}),
            # Delivery thresholds modal
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle("Logistics Thresholds")),
                dbc.ModalBody(
                    dbc.Row([
                        dbc.Col([
                            html.H6('Common thresholds'),
                            dbc.Label('On-time delivery warning'), dcc.Input(id='delivery_on_time_delivery_warning_input', type='number', step='0.01', placeholder='e.g. 0.90', style={'width':'100%','color':'#111'}),
                            dbc.Label('Avg delivery time warn (days)'), dcc.Input(id='delivery_avg_delivery_time_warning_days_input', type='number', step='0.1', placeholder='e.g. 5.0', style={'width':'100%','color':'#111'}),
                            dbc.Label('Return rate warning'), dcc.Input(id='delivery_return_rate_warning_input', type='number', step='0.01', placeholder='e.g. 0.08', style={'width':'100%','color':'#111'}),
                        ], md=5),
                        dbc.Col([
                            html.H6('Advanced / Raw JSON'),
                            dcc.Textarea(id='delivery-thresholds-json-textarea', style={'width':'100%','minHeight':'200px','fontFamily':'monospace','color':'#111'}),
                            html.Hr(), html.H6('All KPI Thresholds (editable table)'),
                            dash_table.DataTable(id='delivery-thresholds-datatable', columns=[{'name':'kpi','id':'kpi','type':'text','editable':False},{'name':'info','id':'info','type':'numeric','editable':True},{'name':'warning','id':'warning','type':'numeric','editable':True},{'name':'critical','id':'critical','type':'numeric','editable':True}], data=[], editable=True, style_table={'height':'200px','overflowY':'auto'}, style_cell={'textAlign':'left','minWidth':'120px','width':'160px','maxWidth':'260px','color':'#111'}, style_header={'fontWeight':'bold','color':'#111','backgroundColor':'#f7f7f7'})
                        ], md=7)
                    ])
                ),
                dbc.ModalFooter([
                    dbc.Button("Auto-fill", id='delivery-thresholds-autofill-btn', color='secondary', class_name='me-2'),
                    dbc.Button("Load", id='delivery-thresholds-load-btn', color='secondary', class_name='me-2'),
                    dbc.Button("Save", id='delivery-thresholds-save-btn', color='primary', class_name='me-2'),
                    dbc.Button("Close", id='delivery-thresholds-close-btn', color='light'),
                    html.Span(id='delivery-thresholds-save-feedback', style={'marginLeft':'12px'})
                ])
            ], id='delivery-thresholds-modal', size='lg')
        ], width=12)]),

        dbc.Row([create_graph_card('delivery-pipeline-chart', width=12)]),
        
        # --- NEW: Replaced partner charts with internal fleet charts ---
        html.Hr(className="my-4"),
        dbc.Row([html.H4("Internal Fleet Analytics", className="mb-3 text-center")]),
        dbc.Row([
            create_graph_card('driver-leaderboard-chart', title="Driver Performance Leaderboard"),
            create_graph_card('vehicle-efficiency-chart', title="Vehicle Type Efficiency")
        ]),
        html.Hr(className="my-4"),
        # --- END MODIFICATION ---

        dbc.Row([create_graph_card('avg-time-by-city-chart', title="Average Delivery Time by City", width=12)]),
        
    ], fluid=True)

# ... (All other layout functions: create_customer_layout, create_marketing_layout, etc. remain unchanged) ...
def create_customer_layout() -> dbc.Container:
    customer_analysis_df = DATA.get('customer_analysis_df', pd.DataFrame())
    if customer_analysis_df.empty: return dbc.Container(html.H4("Customer or Sales Data Not Available", className="text-center mt-5"))
    region_opts, segment_opts = customer_analysis_df['city'].unique(), customer_analysis_df['segment'].unique()
    return dbc.Container([
        dbc.Card(dbc.CardBody([
             dbc.Row([
                 dbc.Col([html.Label("Customer Join Date Range:"), dcc.DatePickerRange(id='customer-date-picker', min_date_allowed=customer_analysis_df['joindate'].min(), max_date_allowed=customer_analysis_df['joindate'].max(), start_date=customer_analysis_df['joindate'].min(), end_date=customer_analysis_df['joindate'].max(), className="d-block")], width=12, lg=4, className="mb-2"),
                 dbc.Col([html.Label("Region (City):"), dcc.Dropdown(id='customer-region-filter', options=create_multi_filter_options(region_opts), value=['All'], multi=True, clearable=False)], width=12, lg=3, className="mb-2"),
                 dbc.Col([html.Label("Customer Segment:"), dcc.Dropdown(id='customer-segment-filter', options=create_multi_filter_options(segment_opts), value=['All'], multi=True, clearable=False)], width=12, lg=3, className="mb-2"),
                 dbc.Col([html.Label("Actions", style={'visibility': 'hidden'}), dbc.ButtonGroup([dbc.Button("Apply", id="customer-apply-btn", color="primary"), dbc.Button("Export PDF", id="customer-export-btn", color="secondary")], className="w-100")], width=12, lg=2, className="mb-2 align-self-end"),
             ], align="bottom"),
        ]), className="mb-4"),
    dbc.Row([create_kpi_card("Total Customers", "kpi-total-customers", "primary", width=3, md_width=6), create_kpi_card("Active Customers", "kpi-active-customers", "info", width=3, md_width=6), create_kpi_card("Retention Rate", "kpi-retention-rate", "success", width=3, md_width=6), create_kpi_card("Repeat Purchase Rate", "kpi-repeat-purchase-rate", "success", width=3, md_width=6)]),
    dbc.Row([create_kpi_card("Dormant Customers", "kpi-dormant-customers", "warning", width=6, md_width=6), create_kpi_card("High Churn Risk", "kpi-churn-risk", "danger", width=6, md_width=6)]),

    # Recommendations directly under Customer KPIs
    dbc.Row([dbc.Col([
        html.H4("Actionable Recommendations", className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Dropdown(id='customer-rec-severity-filter', options=[{'label':'All','value':'all'},{'label':'Critical','value':'critical'},{'label':'Warning','value':'warning'},{'label':'Info','value':'info'}], value='all', clearable=False, style={'width':'220px','marginBottom':'8px'}), width=12, lg=3),
            dbc.Col(dbc.ButtonGroup([
                dbc.Button("Edit Thresholds", id="customer-edit-thresholds-btn", color="light"),
                dbc.Button("Refresh Recommendations", id="customer-refresh-recs-btn", color="primary", outline=True),
            ]), width=12, lg=5)
        ]),
    dbc.Card(dbc.CardBody(id='customer-recommendations-list'), style={"minHeight": "120px"}),
    dcc.Store(id='customer-thresholds-saved-signal', data=0),
    dbc.Toast(id='customer-toast', header='Customers', is_open=False, duration=3000, icon='success', children='Saved thresholds.', style={'position':'fixed','top':70,'right':20,'zIndex':1050}),
        # Customer thresholds modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Customer Thresholds")),
            dbc.ModalBody(
                dbc.Row([
                    dbc.Col([
                        html.H6('Common thresholds'),
                        dbc.Label('Churn warning'), dcc.Input(id='customer_churn_warning_input', type='number', step='0.01', placeholder='e.g. 0.05', style={'width':'100%','color':'#111'}),
                        dbc.Label('Repeat rate info'), dcc.Input(id='customer_repeat_rate_info_input', type='number', step='0.01', placeholder='e.g. 0.25', style={'width':'100%','color':'#111'}),
                    ], md=5),
                    dbc.Col([
                        html.H6('Advanced / Raw JSON'),
                        dcc.Textarea(id='customer-thresholds-json-textarea', style={'width':'100%','minHeight':'200px','fontFamily':'monospace','color':'#111'}),
                        html.Hr(), html.H6('All KPI Thresholds (editable table)'),
                        dash_table.DataTable(id='customer-thresholds-datatable', columns=[{'name':'kpi','id':'kpi','type':'text','editable':False},{'name':'info','id':'info','type':'numeric','editable':True},{'name':'warning','id':'warning','type':'numeric','editable':True},{'name':'critical','id':'critical','type':'numeric','editable':True}], data=[], editable=True, style_table={'height':'200px','overflowY':'auto'}, style_cell={'textAlign':'left','minWidth':'120px','width':'160px','maxWidth':'260px','color':'#111'}, style_header={'fontWeight':'bold','color':'#111','backgroundColor':'#f7f7f7'})
                    ], md=7)
                ])
            ),
            dbc.ModalFooter([
                dbc.Button("Auto-fill", id='customer-thresholds-autofill-btn', color='secondary', class_name='me-2'),
                dbc.Button("Load", id='customer-thresholds-load-btn', color='secondary', class_name='me-2'),
                dbc.Button("Save", id='customer-thresholds-save-btn', color='primary', class_name='me-2'),
                dbc.Button("Close", id='customer-thresholds-close-btn', color='light'),
                html.Span(id='customer-thresholds-save-feedback', style={'marginLeft':'12px'})
            ])
        ], id='customer-thresholds-modal', size='lg')
    ], width=12)]),

    dbc.Row([create_graph_card('customer-status-dist-chart', width=12)]),
        html.Hr(className="my-4"), dbc.Row([create_graph_card('rfm-bubble-chart', title="RFM Segment Analysis", width=12)]), html.Hr(className="my-4"),
        dbc.Card(dbc.CardBody([
            dbc.Row([dbc.Col(html.H4("Actionable Customer Lists"), md=6), dbc.Col(dcc.RadioItems(id='customer-list-selector', options=[{'label': 'Top-Value Customers', 'value': 'top_value'}, {'label': 'High Churn Risk', 'value': 'churn_risk'}, {'label': 'New Customers', 'value': 'new'}], value='top_value', inline=True, labelClassName="me-3"), md=6)], align="center"),
            dbc.Row([create_datatable_card(table_id='customer-data-table', title="", width=10), dbc.Col(dbc.Button(["Export ", html.I(className="bi bi-download")], id="export-csv-button", color="primary", className="mt-3 w-100"), lg=2, md=12, sm=12)]),
        ])),
    
    ], fluid=True)
def create_marketing_layout() -> dbc.Container:
    campaign_performance_df = DATA.get('campaign_performance_df', pd.DataFrame())
    if campaign_performance_df.empty: return dbc.Container(html.H4("Marketing Campaign Data Not Available", className="text-center mt-5"))
    channel_opts = campaign_performance_df['channel'].unique()
    product_opts = campaign_performance_df['productname'].unique() if 'productname' in campaign_performance_df.columns else []
    branch_opts = campaign_performance_df['LocationID'].unique() if 'LocationID' in campaign_performance_df.columns else (campaign_performance_df['locationid'].unique() if 'locationid' in campaign_performance_df.columns else [])
    return dbc.Container([
        dbc.Card(dbc.CardBody([
            dbc.Row([
                 dbc.Col([html.Label("Campaign Date Range (Overlap):"), dcc.DatePickerRange(id='marketing-date-picker', min_date_allowed=pd.to_datetime(campaign_performance_df['startdate']).min().date(), max_date_allowed=pd.to_datetime(campaign_performance_df['enddate']).max().date(), start_date=pd.to_datetime(campaign_performance_df['startdate']).min().date(), end_date=pd.to_datetime(campaign_performance_df['enddate']).max().date(), className="d-block")], width=12, lg=5, className="mb-2"),
                 dbc.Col([html.Label("Channel:"), dcc.Dropdown(id='marketing-channel-filter', options=create_filter_options(channel_opts), value='All', clearable=False)], width=12, lg=4, className="mb-2"),
                 dbc.Col([html.Label("Product:"), dcc.Dropdown(id='marketing-product-filter', options=create_multi_filter_options(product_opts), value=['All'], multi=True, clearable=False)], width=12, lg=3, className="mb-2"),
                 dbc.Col([html.Label("Actions", style={'visibility': 'hidden'}), dbc.ButtonGroup([dbc.Button("Apply", id="marketing-apply-btn", color="primary"), dbc.Button("Export PDF", id="marketing-export-btn", color="secondary")], className="w-100")], width=12, lg=3, className="mb-2 align-self-end"),
            ], align="bottom"),
        ]), className="mb-4"),
    dbc.Row([create_kpi_card("Total Ad Spend", "kpi-total-ad-spend", "info", width=3, md_width=6), create_kpi_card("Overall ROAS", "kpi-avg-roas", "primary", width=3, md_width=6), create_kpi_card("Average CPA (CAC)", "kpi-avg-cpa", "warning", width=3, md_width=6), create_kpi_card("CLV to CAC Ratio", "kpi-clv-cac-ratio", "success", width=3, md_width=6)]),
    dbc.Row([create_kpi_card("Attributed Conversions", "kpi-total-conversions", "dark", width=12)]),

    # Recommendations directly under Marketing KPIs for faster action
    dbc.Row([dbc.Col([
        html.H4("Actionable Recommendations", className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Dropdown(id='marketing-rec-severity-filter', options=[{'label':'All','value':'all'},{'label':'Critical','value':'critical'},{'label':'Warning','value':'warning'},{'label':'Info','value':'info'}], value='all', clearable=False, style={'width':'220px','marginBottom':'8px'}), width=12, lg=3),
            dbc.Col(dbc.ButtonGroup([
                dbc.Button("Edit Thresholds", id="marketing-edit-thresholds-btn", color="light"),
                dbc.Button("Refresh Recommendations", id="marketing-refresh-recs-btn", color="primary", outline=True),
            ]), width=12, lg=5)
        ]),
    dbc.Card(dbc.CardBody(id='marketing-recommendations-list'), style={"minHeight": "120px"}),
    dcc.Store(id='marketing-thresholds-saved-signal', data=0),
    dbc.Toast(id='marketing-toast', header='Marketing', is_open=False, duration=3000, icon='success', children='Saved thresholds.', style={'position':'fixed','top':70,'right':20,'zIndex':1050}),
        # Marketing thresholds modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Marketing Thresholds")),
            dbc.ModalBody(
                dbc.Row([
                    dbc.Col([
                        html.H6('Common thresholds'),
                        dbc.Label('ROAS warning'), dcc.Input(id='marketing_roas_warning_input', type='number', step='0.1', placeholder='e.g. 1.5', style={'width':'100%','color':'#111'}),
                        dbc.Label('ROAS critical'), dcc.Input(id='marketing_roas_critical_input', type='number', step='0.1', placeholder='e.g. 1.0', style={'width':'100%','color':'#111'}),
                        dbc.Label('CPA warning'), dcc.Input(id='marketing_cpa_warning_input', type='number', step='0.1', placeholder='e.g. 200', style={'width':'100%','color':'#111'}),
                        dbc.Label('Low spend threshold'), dcc.Input(id='marketing_low_spend_threshold_input', type='number', step='1', placeholder='e.g. 5000', style={'width':'100%','color':'#111'}),
                    ], md=5),
                    dbc.Col([
                        html.H6('Advanced / Raw JSON'),
                        dcc.Textarea(id='marketing-thresholds-json-textarea', style={'width':'100%','minHeight':'200px','fontFamily':'monospace','color':'#111'}),
                        html.Hr(), html.H6('All KPI Thresholds (editable table)'),
                        dash_table.DataTable(id='marketing-thresholds-datatable', columns=[{'name':'kpi','id':'kpi','type':'text','editable':False},{'name':'info','id':'info','type':'numeric','editable':True},{'name':'warning','id':'warning','type':'numeric','editable':True},{'name':'critical','id':'critical','type':'numeric','editable':True}], data=[], editable=True, style_table={'height':'200px','overflowY':'auto'}, style_cell={'textAlign':'left','minWidth':'120px','width':'160px','maxWidth':'260px','color':'#111'}, style_header={'fontWeight':'bold','color':'#111','backgroundColor':'#f7f7f7'})
                    ], md=7)
                ])
            ),
            dbc.ModalFooter([
                dbc.Button("Auto-fill", id='marketing-thresholds-autofill-btn', color='secondary', class_name='me-2'),
                dbc.Button("Load", id='marketing-thresholds-load-btn', color='secondary', class_name='me-2'),
                dbc.Button("Save", id='marketing-thresholds-save-btn', color='primary', class_name='me-2'),
                dbc.Button("Close", id='marketing-thresholds-close-btn', color='light'),
                html.Span(id='marketing-thresholds-save-feedback', style={'marginLeft':'12px'})
            ])
        ], id='marketing-thresholds-modal', size='lg')
    ], width=12)]),

    html.Hr(className="my-4"), dbc.Row([create_graph_card('clv-by-channel-chart', title="Average Customer Lifetime Value by Acquisition Channel", width=12)]), html.Hr(className="my-4"),
        dbc.Row([create_graph_card('roas-by-campaign-chart', width=12)]), dbc.Row([create_graph_card('cpa-by-campaign-chart'), create_graph_card('conversions-by-channel-chart')]),
    ], fluid=True)
def create_profit_layout() -> dbc.Container:
    profit_df = DATA.get('profit_df', pd.DataFrame())
    if profit_df.empty: return dbc.Container(html.H4("Profit calculation requires Sales, Delivery, and Marketing data.", className="text-center mt-5"))
    region_opts, category_opts = profit_df['city'].unique(), profit_df['category'].unique()
    product_opts = profit_df['productname'].unique() if 'productname' in profit_df.columns else []
    branch_opts = profit_df['LocationID'].unique() if 'LocationID' in profit_df.columns else (profit_df['locationid'].unique() if 'locationid' in profit_df.columns else [])
    return dbc.Container([
        dbc.Card(dbc.CardBody([
            dbc.Row([
                 dbc.Col([html.Label("Date Range:"), dcc.DatePickerRange(id='profit-date-picker', min_date_allowed=profit_df['date'].min(), max_date_allowed=profit_df['date'].max(), start_date=profit_df['date'].min(), end_date=profit_df['date'].max(), className="d-block")], width=12, lg=4, className="mb-2"),
                 dbc.Col([html.Label("Region (City):"), dcc.Dropdown(id='profit-region-filter', options=create_multi_filter_options(region_opts), value=['All'], multi=True, clearable=False)], width=12, lg=3, className="mb-2"),
                 dbc.Col([html.Label("Category:"), dcc.Dropdown(id='profit-category-filter', options=create_multi_filter_options(category_opts), value=['All'], multi=True, clearable=False)], width=12, lg=2, className="mb-2"),
                 dbc.Col([html.Label("Product:"), dcc.Dropdown(id='profit-product-filter', options=create_multi_filter_options(product_opts), value=['All'], multi=True, clearable=False)], width=12, lg=2, className="mb-2"),
                 dbc.Col([html.Label("Branch:"), dcc.Dropdown(id='profit-branch-filter', options=create_multi_filter_options(branch_opts), value=['All'], multi=True, clearable=False)], width=12, lg=2, className="mb-2"),
                dbc.Col([html.Label("Actions", style={'visibility': 'hidden'}), dbc.ButtonGroup([
                    dbc.Button("Apply", id="profit-apply-btn", color="primary"),
                    dbc.Button("Export PDF", id="profit-export-btn", color="secondary"),
                ], className="w-100")], width=12, lg=2, className="mb-2 align-self-end"),
            ], align="bottom"),
        ]), className="mb-4"),
    dbc.Row([create_kpi_card("Total Net Profit", "kpi-total-net-profit", "success"), create_kpi_card("Average Profit Margin", "kpi-avg-profit-margin", "primary"), create_kpi_card("Profit Lost to Returns", "kpi-profit-lost-returns", "danger")]),

    # Recommendations directly under Profit KPIs
    dbc.Row([dbc.Col([
        html.H4("Actionable Recommendations", className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Dropdown(id='profit-rec-severity-filter', options=[{'label':'All','value':'all'},{'label':'Critical','value':'critical'},{'label':'Warning','value':'warning'},{'label':'Info','value':'info'}], value='all', clearable=False, style={'width':'220px','marginBottom':'8px'}), width=12, lg=3),
            dbc.Col(dbc.ButtonGroup([
                dbc.Button("Edit Thresholds", id="profit-edit-thresholds-btn", color="light"),
                dbc.Button("Refresh Recommendations", id="profit-refresh-recs-btn", color="primary", outline=True),
            ]), width=12, lg=5)
        ]),
    dbc.Card(dbc.CardBody(id='automated-recommendations-list'), style={"minHeight": "120px"}),
    dcc.Store(id='profit-thresholds-saved-signal', data=0),
    dbc.Toast(id='profit-toast', header='Profit', is_open=False, duration=3000, icon='success', children='Saved thresholds.', style={'position':'fixed','top':70,'right':20,'zIndex':1050})
        ,
        # Profit thresholds modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Profit Thresholds")),
            dbc.ModalBody(
                dbc.Row([
                    dbc.Col([
                        html.H6('Common thresholds'),
                        dbc.Label('Gross margin warning'), dcc.Input(id='profit_gross_margin_warning_input', type='number', step='0.01', placeholder='e.g. 0.60', style={'width':'100%','color':'#111'}),
                        dbc.Label('Gross margin critical'), dcc.Input(id='profit_gross_margin_critical_input', type='number', step='0.01', placeholder='e.g. 0.45', style={'width':'100%','color':'#111'}),
                        dbc.Label('Return rate warning'), dcc.Input(id='profit_return_rate_warning_input', type='number', step='0.01', placeholder='e.g. 0.04', style={'width':'100%','color':'#111'}),
                        dbc.Label('Return rate critical'), dcc.Input(id='profit_return_rate_critical_input', type='number', step='0.01', placeholder='e.g. 0.07', style={'width':'100%','color':'#111'}),
                        dbc.Label('AOV warning'), dcc.Input(id='profit_aov_warning_input', type='number', step='0.1', placeholder='e.g. 35.0', style={'width':'100%','color':'#111'}),
                        dbc.Label('AOV info'), dcc.Input(id='profit_aov_info_input', type='number', step='0.1', placeholder='e.g. 20.0', style={'width':'100%','color':'#111'}),
                    ], md=5),
                    dbc.Col([
                        html.H6('Advanced / Raw JSON'),
                        dcc.Textarea(id='profit-thresholds-json-textarea', style={'width':'100%','minHeight':'200px','fontFamily':'monospace','color':'#111'}),
                        html.Hr(), html.H6('All KPI Thresholds (editable table)'),
                        dash_table.DataTable(id='profit-thresholds-datatable', columns=[{'name':'kpi','id':'kpi','type':'text','editable':False},{'name':'info','id':'info','type':'numeric','editable':True},{'name':'warning','id':'warning','type':'numeric','editable':True},{'name':'critical','id':'critical','type':'numeric','editable':True}], data=[], editable=True, style_table={'height':'200px','overflowY':'auto'}, style_cell={'textAlign':'left','minWidth':'120px','width':'160px','maxWidth':'260px','color':'#111'}, style_header={'fontWeight':'bold','color':'#111','backgroundColor':'#f7f7f7'})
                    ], md=7)
                ])
            ),
            dbc.ModalFooter([
                dbc.Button("Auto-fill", id='profit-thresholds-autofill-btn', color='secondary', class_name='me-2'),
                dbc.Button("Load", id='profit-thresholds-load-btn', color='secondary', class_name='me-2'),
                dbc.Button("Save", id='profit-thresholds-save-btn', color='primary', class_name='me-2'),
                dbc.Button("Close", id='profit-thresholds-close-btn', color='light'),
                html.Span(id='profit-thresholds-save-feedback', style={'marginLeft':'12px'})
            ])
        ], id='profit-thresholds-modal', size='lg')
    ], width=12)]),

    html.Hr(className="my-4"),
        dbc.Row([create_graph_card('profit-waterfall-chart', title="Profitability Waterfall Analysis", width=12)]),
        html.Hr(className="my-4"),
        # Added: Channel and Category profit contribution charts to align with callbacks
        dbc.Row([create_graph_card('profit-by-channel-chart', title="Profit Contribution by Channel"),
                 create_graph_card('profit-by-category-chart', title="Net Profit by Product Category")]),
        html.Hr(className="my-4"),
        dbc.Row([create_graph_card('high-margin-products-chart'), create_graph_card('low-margin-products-chart')]),
    ], fluid=True)
def _create_forecast_tab() -> dbc.Tab:
    return dbc.Tab(label="Demand Forecasting & Promotion Simulation", children=[dbc.Row([dbc.Col(dbc.Card([dbc.CardBody([html.H5("Simulation Controls"), dbc.Label("Forecast Horizon (Days):"), dcc.Slider(id="forecast-slider-days", min=30, max=180, step=30, value=90, marks={30:'30', 90:'90', 180:'180'}, tooltip={"placement": "bottom", "always_visible": True}), html.Hr(), dbc.Label("Simulate Promotion Uplift (% Increase):"), dcc.Slider(id="forecast-slider-promo", min=0, max=50, step=5, value=0, marks={i: f"{i}%" for i in range(0, 51, 10)}, tooltip={"placement": "bottom", "always_visible": True}), dbc.Button("Run Simulation", id="forecast-run-button", color="primary", className="mt-4 w-100")])]), lg=3, md=12, className="mb-4"), dbc.Col([dbc.Row([create_kpi_card("Forecasted Revenue (Baseline)", "pred-kpi-forecast-rev", "primary", width=6, md_width=6), create_kpi_card("Simulated Revenue Lift (Promo)", "pred-kpi-sim-lift", "success", width=6, md_width=6)]), dbc.Row([create_graph_card(graph_id="forecast-simulation-chart", title="Demand Forecast & Promotion Simulation", width=12)])], lg=9, md=12)], className="mt-3")])
def _create_churn_tab() -> dbc.Tab:
    return dbc.Tab(label="Customer Churn & LTV", children=[dcc.Loading(id="loading-churn-content", type="default", children=html.Div(id="churn-tab-content-wrapper", className="mt-3"))])
def create_predictive_layout() -> dbc.Container:
    """Redesigned Predictive page:
    - Left column: Forecast & Simulation controls and chart
    - Middle column: Churn KPIs, drivers and at-risk table
    - Right column: Models registry + Jobs panel (for MLOps visibility)
    """
    return dbc.Container([
        dbc.Row([dbc.Col(html.H4("Predictive Analytics & Simulations"), width=12, className="mb-3")]),
        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody([html.H5("Forecast Controls"), dbc.Label("Forecast Horizon (Days):"), dcc.Slider(id="forecast-slider-days", min=30, max=180, step=30, value=90, marks={30:'30', 90:'90', 180:'180'}), html.Hr(), dbc.Label("Simulate Promotion Uplift (%Increase):"), dcc.Slider(id="forecast-slider-promo", min=0, max=50, step=5, value=0, marks={i: f"{i}%" for i in range(0, 51, 10)}), dbc.Button("Run Simulation", id="forecast-run-button", color="primary", className="mt-2 w-100"), dbc.Button("Train Forecast Model", id="train-forecast-btn", color="secondary", className="mt-2 w-100")]), className="mb-4"),
                dbc.Card(dbc.CardBody([create_graph_card(graph_id="forecast-simulation-chart", title="Demand Forecast & Promotion Simulation", width=12)])),
            ], lg=4, md=12),

            dbc.Col([
                dbc.Card(dbc.CardBody([html.H5("Customer Churn & LTV"), dcc.Loading(id="loading-churn-content", type="default", children=html.Div(id="churn-tab-content-wrapper"))])),
                # Dataset readiness will be rendered inline by the churn tab content callback
                # (previous approach caused callback registration timing issues).
            ], lg=5, md=12),

            dbc.Col([
                dbc.Card(dbc.CardBody([html.H5("Models & Jobs"), html.P("Registered models and recent training jobs."),
                    dbc.Button([html.I(className="bi bi-rocket-takeoff me-2"), "Train Churn Model"], id='train-churn-btn', color='primary', className='mb-2 w-100'),
                    dash_table.DataTable(id='models-registry-table', columns=[{'name':'Model Name','id':'model_name'},{'name':'Trained At','id':'trained_at'},{'name':'Artifact Path','id':'artifact_path'},{'name':'Metrics','id':'metrics'}], page_size=5, style_table={'overflowX': 'auto'}), html.Hr(),
                    # Progress indicator for the most recent running job
                    html.Div(id='job-progress-container', children=[
                        html.Div(id='job-progress-label', children='No active training jobs.'),
                        dbc.Progress(id='job-progress-bar', value=0, striped=False, animated=False, style={'height': '18px', 'marginTop': '6px'})
                    ]),
                    html.Hr(), dash_table.DataTable(id='jobs-status-table', columns=[{'name':'Job ID','id':'job_id'},{'name':'Status','id':'status'},{'name':'Updated','id':'updated_at'},{'name':'Details','id':'details'}], page_size=5, style_table={'overflowX': 'auto'}), dcc.Interval(id='jobs-poll-interval', interval=15*1000, n_intervals=0)])),
                html.Hr(),
                html.H6('Model Explainability'),
                dcc.Dropdown(id='model-selector-dropdown', options=[], placeholder='Select a model', clearable=True),
                html.Div(id='shap-summary-container', children=[
                    dash_table.DataTable(
                        id='shap-summary-table',
                        columns=[{'name':'Feature','id':'Feature'},{'name':'MeanAbsSHAP','id':'MeanAbsSHAP'}],
                        page_size=10,
                        style_table={'overflowX': 'auto', 'borderRadius': '8px'},
                        style_header={'backgroundColor': 'rgba(255,255,255,0.06)', 'color': '#e8f0ff', 'fontWeight': '700', 'border': 'none'},
                        style_cell={'backgroundColor': 'rgba(255,255,255,0.02)', 'color': '#f5f8ff', 'textAlign': 'left', 'padding': '0.4rem 0.55rem', 'border': 'none'}
                    ),
                    dcc.Graph(id='shap-summary-chart')
                ]),
            ], lg=3, md=12)
        ], className="mt-3"),
    # Hidden stores for training signal and model lists (global store defined in main layout)
    # NOTE: do not define `model-training-signal-store` here to avoid duplicate component ids.
    ], fluid=True)
def create_main_layout() -> html.Div:
    navbar = dbc.Navbar(
        dbc.Container([
            html.A(
                dbc.Row([
                    dbc.Col(html.I(className="bi bi-bar-chart-line-fill", style={"fontSize": "2rem", "color": "#fff"}), width="auto"),
                    dbc.Col(dbc.NavbarBrand("Yosri Analytics Hub", className="ms-2"), width="auto")
                ], align="center", className="g-0"),
                href="#", style={"textDecoration": "none"}
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            dbc.Collapse(
                dbc.Nav([
                    dbc.Button([
                        html.I(className="bi bi-arrow-clockwise me-2"), "Refresh Data "
                    ], id="refresh-data-button", color="secondary", class_name="shadow-sm")
                ], className="ms-auto p-2", navbar=True),
                id="navbar-collapse", is_open=False, navbar=True
            )
    ]),
        color="primary", dark=True, class_name="mb-4 shadow-sm"
    )
    return html.Div([
        dcc.Interval(id='alert-poll-interval', interval=60 * 1000, n_intervals=0),
        html.Div(id='active-alert-banner-container', style={'padding': '10px'}),
        dcc.Store(id='data-store-trigger'),
    dcc.Download(id="download-dataframe-csv"),
    # Dedicated download component for Predictive tab exports to avoid duplicate outputs
    dcc.Download(id="pred-download-dataframe-csv"),
        dcc.Download(id="download-dashboard-pdf"),
        dcc.Store(id='store-forecast-model'),
        dcc.Store(id='store-churn-model'),
    dcc.Store(id='model-training-signal-store', data=0),
    # Store that carries the dataset readiness summary (warnings, counts) so callbacks can gate training.
    dcc.Store(id='dataset-readiness-store', data={}),
    # Hidden button used by predictive callbacks to trigger manual training from the predictive tab.
    dbc.Button(id='run-manual-churn-train-btn', style={'display':'none'}),
    # Confirmation modal that appears when dataset readiness warnings exist and the user attempts to train.
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Confirm Training")),
        dbc.ModalBody(html.Div("Dataset readiness warnings were detected. Training now may produce unreliable models. Do you want to proceed?")),
        dbc.ModalFooter([
            dbc.Button("Confirm", id='confirm-train-yes', color='primary', className='me-2'),
            dbc.Button("Cancel", id='confirm-train-no', color='secondary')
        ])
    ], id='confirm-train-modal', is_open=False),
    # Confirmation modal specifically for Forecast training (reuses messaging)
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Confirm Forecast Training")),
        dbc.ModalBody(html.Div("Dataset readiness warnings were detected for forecast training. Proceed to train the Demand Forecaster anyway?")),
        dbc.ModalFooter([
            dbc.Button("Confirm", id='confirm-forecast-yes', color='primary', className='me-2'),
            dbc.Button("Cancel", id='confirm-forecast-no', color='secondary')
        ])
    ], id='confirm-forecast-modal', is_open=False),
        navbar,
        dbc.Container([
            dbc.Tabs(
                id="tabs-controller",
                active_tab="welcome-tab",
                class_name="mb-4 shadow-sm",
                children=[
                    dbc.Tab(
                        label="Welcome",
                        tab_id="welcome-tab",
                        class_name="fw-bold",
                        children=[
                            dbc.Container(
                                fluid=True,
                                style={
                                    'background': 'linear-gradient(180deg, #0f1724 0%, #0b1220 100%)',
                                    'padding': '40px',
                                    'borderRadius': '8px'
                                },
                                children=[
                                    dbc.Row(
                                        dbc.Col(
                                            html.H1("Welcome to Yosri Analytics Hub", className="display-5 fw-bold text-light text-center"),
                                            width=12
                                        ),
                                        className='mb-2',
                                        style={'paddingTop': '10px'}
                                    ),
                                    dbc.Row(
                                        dbc.Col(
                                            html.P(
                                                "Our enterprise analytics engine, designed to drive intelligent growth through data.",
                                                className="lead text-light text-center mx-auto",
                                                style={'maxWidth': '900px'}
                                            ),
                                            width=12
                                        ),
                                        className='mb-3'
                                    ),
                                    html.Hr(style={'borderColor': 'rgba(255,255,255,0.08)'}),

                                    # 2x2 feature cards
                                    dbc.Row([
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody([
                                                    html.H5("Unified Data, Total Clarity.", className='fw-bold text-light'),
                                                    html.P("A fully automated ETL pipeline delivers a single source of truth for consistent, reliable reporting across the business.", className='text-light')
                                                ])
                                            , color='dark', inverse=True, class_name='h-100 shadow-sm'),
                                            md=6,
                                            class_name='mb-3'
                                        ),
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody([
                                                    html.H5("From Insight to Action.", className='fw-bold text-light'),
                                                    html.P("Go beyond dashboards. Receive clear, actionable recommendations that translate directly into strategic execution.", className='text-light')
                                                ])
                                            , color='dark', inverse=True, class_name='h-100 shadow-sm'),
                                            md=6,
                                            class_name='mb-3'
                                        )
                                    ], className='mb-3'),

                                    dbc.Row([
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody([
                                                    html.H5("Predictive Power with Machine Learning.", className='fw-bold text-light'),
                                                    html.P("Leverage our integrated churn model to proactively retain customers and transform potential risks into growth opportunities.", className='text-light')
                                                ])
                                            , color='dark', inverse=True, class_name='h-100 shadow-sm'),
                                            md=6,
                                            class_name='mb-3'
                                        ),
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody([
                                                    html.H5("Engineered for Enterprise.", className='fw-bold text-light'),
                                                    html.P("Built on a robust, containerized architecture with a dedicated model registry, ensuring the reliability and stability our operations depend on.", className='text-light')
                                                ])
                                            , color='dark', inverse=True, class_name='h-100 shadow-sm'),
                                            md=6,
                                            class_name='mb-3'
                                        )
                                    ], className='mb-4'),

                                    dbc.Row(
                                        dbc.Col(
                                            html.P(
                                                html.Em("Our Goal: To harness intelligent automation to build a more profitable, data-driven future."),
                                                className='text-light text-center'
                                            ),
                                            width=12
                                        ),
                                        className='mb-3'
                                    ),

                                    dbc.Row(
                                        dbc.Col(
                                            dbc.Button("Get started", id='welcome-get-started', color='primary', class_name='d-block mx-auto'),
                                            width=12
                                        )
                                    )
                                ]
                            )
                        ]
                    ),
                    dbc.Tab(label="Comprehensive Analysis", tab_id="comprehensive-tab", class_name="fw-bold", children=[get_comprehensive_layout()]),
                    dbc.Tab(label="Sales", tab_id="sales-tab", class_name="fw-bold"),
                    dbc.Tab(label="Logistics", tab_id="delivery-tab", class_name="fw-bold"),
                    dbc.Tab(label="Customers", tab_id="customer-tab", class_name="fw-bold"),
                    dbc.Tab(label="Marketing", tab_id="marketing-tab", class_name="fw-bold"),
                    dbc.Tab(label="Profit Optimization", tab_id="profit-tab", class_name="fw-bold"),
                    dbc.Tab(label="Predictive Insights", tab_id="predictive-tab", class_name="fw-bold")
                ]
            ),
            html.Div(id='tab-content', className="mt-4"),
            # Debugging aid: show current active_tab value when tabs are clicked
            html.Div(id='tab-debug', style={'display': 'none'})
        ], fluid=True)
    ])