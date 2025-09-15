from dash import Input, Output, State, callback
import dash
import json
import os


def _is_number_like(v):
    """Return True if v is None or can be converted to float."""
    # treat None or empty-string as "no value" (allowed)
    if v is None or v == '':
        return True
    try:
        # allow numeric strings too
        float(v)
        return True
    except Exception:
        return False


def _coerce_number(v):
    """Coerce numeric-like strings to int/float, return None for empty values, otherwise return original."""
    if v is None or v == '':
        return None
    try:
        f = float(v)
        # return int when it represents an integer value
        if f.is_integer():
            return int(f)
        return f
    except Exception:
        return v


def validate_threshold_inputs(table_data, conv_warn, conv_crit, gm_warn, gm_crit, aov_warn, aov_info, ret_warn):
    """Validate that numeric-friendly inputs and datatable numeric cells are numeric or empty.
    Returns: (ok: bool, message: str)
    """
    # Friendly inputs
    friendly = {
        'conversion_rate_warning': conv_warn,
        'conversion_rate_critical': conv_crit,
        'gross_margin_warning': gm_warn,
        'gross_margin_critical': gm_crit,
        'aov_warning': aov_warn,
        'aov_info': aov_info,
        'return_rate_warning': ret_warn,
    }
    for k, v in friendly.items():
        if not _is_number_like(v):
            return False, f"Friendly input '{k}' must be numeric or empty. Got: {v}"

    # Table rows
    try:
        if table_data:
            for r in table_data:
                for col in ('info', 'warning', 'critical'):
                    if col in r and not _is_number_like(r.get(col)):
                        return False, f"Table cell for KPI '{r.get('kpi')}' column '{col}' must be numeric or empty. Got: {r.get(col)}"
    except Exception as e:
        return False, f"Invalid table data: {e}"
    return True, ''


def merge_thresholds_from_inputs(json_text, table_data, conv_warn, conv_crit, gm_warn, gm_crit, aov_warn, aov_info, ret_warn):
    """Merge base JSON with friendly inputs and datatable rows and return the merged dict (does not write file)."""
    try:
        base = json.loads(json_text or '{}')
    except Exception:
        base = {}
    # Overwrite with friendly fields where provided
    # Only write friendly values if they are not empty-string or None
    if conv_warn not in (None, ''): base['conversion_rate_warning'] = _coerce_number(conv_warn)
    if conv_crit not in (None, ''): base['conversion_rate_critical'] = _coerce_number(conv_crit)
    if gm_warn not in (None, ''): base['gross_margin_warning'] = _coerce_number(gm_warn)
    if gm_crit not in (None, ''): base['gross_margin_critical'] = _coerce_number(gm_crit)
    if aov_warn not in (None, ''): base['aov_warning'] = _coerce_number(aov_warn)
    if aov_info not in (None, ''): base['aov_info'] = _coerce_number(aov_info)
    if ret_warn not in (None, ''): base['return_rate_warning'] = _coerce_number(ret_warn)
    # Merge datatable rows if present
    try:
        if table_data:
            for row in table_data:
                kpi = row.get('kpi')
                if not kpi: continue
                if 'info' in row and row['info'] is not None: base[f"{kpi}_info"] = _coerce_number(row['info'])
                if 'warning' in row and row['warning'] is not None: base[f"{kpi}_warning"] = _coerce_number(row['warning'])
                if 'critical' in row and row['critical'] is not None: base[f"{kpi}_critical"] = _coerce_number(row['critical'])
    except Exception:
        pass
    return base


# Helpers for per-tab controllers

def _tab_thresholds_path(tab: str) -> str:
    out_dir = os.path.join(os.getcwd(), 'model_store')
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"thresholds_{tab}.json")


def _build_table_rows_from_dict(data: dict) -> list:
    rows_map = {}
    for k, v in (data or {}).items():
        if k.endswith('_info'):
            base = k[:-5]
            rows_map.setdefault(base, {})['info'] = v
        elif k.endswith('_warning'):
            base = k[:-8]
            rows_map.setdefault(base, {})['warning'] = v
        elif k.endswith('_critical'):
            base = k[:-9]
            rows_map.setdefault(base, {})['critical'] = v
    table_rows = []
    for base, vals in rows_map.items():
        table_rows.append({'kpi': base, 'info': vals.get('info'), 'warning': vals.get('warning'), 'critical': vals.get('critical')})
    return sorted(table_rows, key=lambda r: r['kpi'])


def _filter_kpis_for_tab(tab: str, kpis: dict) -> dict:
    tab_keys = {
        'sales': {'conversion_rate', 'gross_margin', 'aov', 'return_rate', 'gmv', 'net_sales', 'num_orders'},
        'delivery': {'on_time_delivery_rate', 'avg_delivery_time', 'return_rate'},
        'marketing': {'roas', 'cpa', 'total_ad_spend', 'conversion_rate'},
        'profit': {'gross_margin', 'return_rate', 'aov', 'net_profit'},
        'customer': {'churn_rate', 'repeat_purchase_rate', 'retention_rate', 'clv_cac_ratio'},
    }.get(tab, set())
    out = {}
    for key, val in (kpis or {}).items():
        try:
            float(val)
        except Exception:
            continue
        k = str(key).lower()
        if k.startswith('kpi_'):
            k = k[4:]
        k = k.replace(' ', '_')
        if k in tab_keys:
            out[k] = float(val)
    return out


def _autofill_thresholds_from_data(tab: str, k: dict) -> tuple[dict, dict]:
    """Return (thresholds_dict, friendly_inputs_dict) using heuristics per tab."""
    filtered = _filter_kpis_for_tab(tab, k)
    thresholds = {}
    for key, num in filtered.items():
        if num == 0:
            thresholds[f"{key}_info"] = num
            thresholds[f"{key}_warning"] = 0
            thresholds[f"{key}_critical"] = 0
        else:
            if 0 < abs(num) <= 1:
                thresholds[f"{key}_info"] = num
                thresholds[f"{key}_warning"] = round(num * 0.95, 4)
                thresholds[f"{key}_critical"] = round(num * 0.9, 4)
            else:
                thresholds[f"{key}_info"] = num
                thresholds[f"{key}_warning"] = round(num * 0.9, 4)
                thresholds[f"{key}_critical"] = round(num * 0.75, 4)

    friendly = {}
    if tab == 'profit':
        friendly.update({'gross_margin_warning': 0.60, 'gross_margin_critical': 0.45, 'return_rate_warning': 0.04, 'return_rate_critical': 0.07, 'aov_warning': 35.0, 'aov_info': 20.0})
    elif tab == 'marketing':
        friendly.update({'roas_warning': 1.5, 'roas_critical': 1.0, 'cpa_warning': 200.0, 'low_spend_threshold': 5000.0})
    elif tab == 'delivery':
        friendly.update({'on_time_delivery_warning': 0.90, 'avg_delivery_time_warning_days': 5.0, 'return_rate_warning': 0.08})
    elif tab == 'sales':
        friendly.update({'conversion_rate_warning': 0.05, 'conversion_rate_critical': 0.02, 'gross_margin_warning': 0.20, 'gross_margin_critical': 0.10, 'aov_warning': 35.0, 'aov_info': 20.0, 'return_rate_warning': 0.08})
    elif tab == 'customer':
        friendly.update({'churn_warning': 0.05, 'repeat_rate_info': 0.25})
    return thresholds, friendly


# Per-tab controllers

def _load_threshold_json(tab: str) -> dict:
    path = _tab_thresholds_path(tab)
    sample = os.path.join(os.getcwd(), 'model_store', 'recommendation_thresholds.sample.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    if os.path.exists(sample):
        with open(sample, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    return {}


@callback(
    Output('profit-thresholds-modal', 'is_open'),
    Output('profit-thresholds-saved-signal', 'data', allow_duplicate=True),
    Output('profit-toast', 'is_open', allow_duplicate=True),
    Output('profit-thresholds-json-textarea', 'value'),
    Output('profit-thresholds-datatable', 'data'),
    Output('profit_gross_margin_warning_input', 'value'),
    Output('profit_gross_margin_critical_input', 'value'),
    Output('profit_aov_warning_input', 'value'),
    Output('profit_aov_info_input', 'value'),
    Output('profit_return_rate_warning_input', 'value'),
    Output('profit_return_rate_critical_input', 'value'),
    Output('profit-thresholds-save-feedback', 'children'),
    Input('profit-edit-thresholds-btn', 'n_clicks'),
    Input('profit-thresholds-load-btn', 'n_clicks'),
    Input('profit-thresholds-autofill-btn', 'n_clicks'),
    Input('profit-thresholds-save-btn', 'n_clicks'),
    Input('profit-thresholds-close-btn', 'n_clicks'),
    State('profit-thresholds-json-textarea', 'value'),
    State('profit-thresholds-modal', 'is_open'),
    State('profit-thresholds-datatable', 'data'),
    State('profit_gross_margin_warning_input', 'value'),
    State('profit_gross_margin_critical_input', 'value'),
    State('profit_aov_warning_input', 'value'),
    State('profit_aov_info_input', 'value'),
    State('profit_return_rate_warning_input', 'value'),
    State('profit_return_rate_critical_input', 'value'),
    prevent_initial_call=True
)
def profit_thresholds_controller(edit_click, load_click, autofill_click, save_click, close_click, json_text, is_open, table_data,
                                 gm_warn, gm_crit, aov_warn, aov_info, rr_warn, rr_crit):
    ctx = dash.callback_context
    if not ctx.triggered:
        # profit has 12 outputs; keep full arity
        return (is_open, dash.no_update, False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, '')
    trig = ctx.triggered[0]['prop_id'].split('.')[0]
    tab = 'profit'

    if trig == 'profit-edit-thresholds-btn':
        # Open and load current saved thresholds (or sample) into the modal
        try:
            data = _load_threshold_json(tab)
            table_rows = _build_table_rows_from_dict(data)
            return True, dash.no_update, False, json.dumps(data, indent=2), table_rows, data.get('gross_margin_warning'), data.get('gross_margin_critical'), data.get('aov_warning'), data.get('aov_info'), data.get('return_rate_warning'), data.get('return_rate_critical'), ''
        except Exception:
            return True, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, ''

    if trig == 'profit-thresholds-load-btn':
        try:
            data = _load_threshold_json(tab)
            table_rows = _build_table_rows_from_dict(data)
            return (is_open, dash.no_update, True, json.dumps(data, indent=2), table_rows, data.get('gross_margin_warning'), data.get('gross_margin_critical'), data.get('aov_warning'), data.get('aov_info'), data.get('return_rate_warning'), data.get('return_rate_critical'), '')
        except Exception as e:
                return (is_open, f"{{\n  \"error\": \"{str(e)}\"\n}}", dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update)

    if trig == 'profit-thresholds-autofill-btn':
        try:
            from etl.transforms import DATA as TRANSFORMS_DATA
            k = TRANSFORMS_DATA.get('kpis', {}) or {}
            thresholds, friendly = _autofill_thresholds_from_data(tab, k)
            table_rows = _build_table_rows_from_dict(thresholds)
            return is_open, dash.no_update, True, json.dumps(thresholds, indent=2), table_rows, friendly.get('gross_margin_warning'), friendly.get('gross_margin_critical'), friendly.get('aov_warning'), friendly.get('aov_info'), friendly.get('return_rate_warning'), friendly.get('return_rate_critical'), ''
        except Exception as e:
            return (is_open, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, f'Autofill failed: {e}')

    if trig == 'profit-thresholds-save-btn':
        try:
            # Merge into JSON
            try:
                base = json.loads(json_text or '{}')
            except Exception:
                base = {}
            for row in (table_data or []):
                kpi = row.get('kpi');
                if not kpi: continue
                if row.get('info') is not None: base[f"{kpi}_info"] = _coerce_number(row.get('info'))
                if row.get('warning') is not None: base[f"{kpi}_warning"] = _coerce_number(row.get('warning'))
                if row.get('critical') is not None: base[f"{kpi}_critical"] = _coerce_number(row.get('critical'))
            # friendly
            if gm_warn not in (None, ''): base['gross_margin_warning'] = _coerce_number(gm_warn)
            if gm_crit not in (None, ''): base['gross_margin_critical'] = _coerce_number(gm_crit)
            if aov_warn not in (None, ''): base['aov_warning'] = _coerce_number(aov_warn)
            if aov_info not in (None, ''): base['aov_info'] = _coerce_number(aov_info)
            if rr_warn not in (None, ''): base['return_rate_warning'] = _coerce_number(rr_warn)
            if rr_crit not in (None, ''): base['return_rate_critical'] = _coerce_number(rr_crit)
            # write
            path = _tab_thresholds_path(tab)
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(base, fh, indent=2)
            table_rows = _build_table_rows_from_dict(base)
            curr = dash.callback_context.states.get('profit-thresholds-saved-signal.data')
            try:
                curr = int(curr) if curr is not None else 0
            except Exception:
                curr = 0
                return False, curr + 1, True, json.dumps(base, indent=2), table_rows, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, 'Saved'
        except Exception as e:
            return False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, f'Error saving: {e}'

    # close explicitly
    if trig == 'profit-thresholds-close-btn':
        return False, dash.no_update, False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, ''
    return dash.no_update, dash.no_update, False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, ''


# Sales thresholds controller
@callback(
    Output('sales-thresholds-modal', 'is_open'),
    Output('sales-thresholds-saved-signal', 'data', allow_duplicate=True),
    Output('sales-toast', 'is_open', allow_duplicate=True),
    Output('sales-thresholds-json-textarea', 'value'),
    Output('sales-thresholds-datatable', 'data'),
    Output('sales_conversion_rate_warning_input', 'value'),
    Output('sales_conversion_rate_critical_input', 'value'),
    Output('sales_gross_margin_warning_input', 'value'),
    Output('sales_gross_margin_critical_input', 'value'),
    Output('sales_aov_warning_input', 'value'),
    Output('sales_aov_info_input', 'value'),
    Output('sales_return_rate_warning_input', 'value'),
    Output('sales-thresholds-save-feedback', 'children'),
    Input('sales-edit-thresholds-btn', 'n_clicks'),
    Input('sales-thresholds-load-btn', 'n_clicks'),
    Input('sales-thresholds-autofill-btn', 'n_clicks'),
    Input('sales-thresholds-save-btn', 'n_clicks'),
    Input('sales-thresholds-close-btn', 'n_clicks'),
    State('sales-thresholds-json-textarea', 'value'),
    State('sales-thresholds-modal', 'is_open'),
    State('sales-thresholds-datatable', 'data'),
    State('sales_conversion_rate_warning_input', 'value'),
    State('sales_conversion_rate_critical_input', 'value'),
    State('sales_gross_margin_warning_input', 'value'),
    State('sales_gross_margin_critical_input', 'value'),
    State('sales_aov_warning_input', 'value'),
    State('sales_aov_info_input', 'value'),
    State('sales_return_rate_warning_input', 'value'),
    prevent_initial_call=True
)
def sales_thresholds_controller(edit_click, load_click, autofill_click, save_click, close_click, json_text, is_open, table_data,
                                conv_warn, conv_crit, gm_warn, gm_crit, aov_warn, aov_info, ret_warn):
    ctx = dash.callback_context
    if not ctx.triggered:
        return (is_open, dash.no_update, False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, '')
    trig = ctx.triggered[0]['prop_id'].split('.')[0]
    tab = 'sales'
    if trig == 'sales-edit-thresholds-btn':
        try:
            data = _load_threshold_json(tab)
            table_rows = _build_table_rows_from_dict(data)
            return True, dash.no_update, False, json.dumps(data, indent=2), table_rows, data.get('conversion_rate_warning'), data.get('conversion_rate_critical'), data.get('gross_margin_warning'), data.get('gross_margin_critical'), data.get('aov_warning'), data.get('aov_info'), data.get('return_rate_warning'), ''
        except Exception:
            # ensure we return the full tuple of 13 outputs (use no_update for most fields)
            return True, dash.no_update, False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, ''
    if trig == 'sales-thresholds-load-btn':
        try:
            data = _load_threshold_json(tab)
            table_rows = _build_table_rows_from_dict(data)
            return (is_open, dash.no_update, True, json.dumps(data, indent=2), table_rows, data.get('conversion_rate_warning'), data.get('conversion_rate_critical'), data.get('gross_margin_warning'), data.get('gross_margin_critical'), data.get('aov_warning'), data.get('aov_info'), data.get('return_rate_warning'), '')
        except Exception as e:
            # return full arity: bubble error into save-feedback (last output)
            return (is_open, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, str(e))
    if trig == 'sales-thresholds-autofill-btn':
        try:
            from etl.transforms import DATA as TRANSFORMS_DATA
            k = TRANSFORMS_DATA.get('kpis', {}) or {}
            thresholds, friendly = _autofill_thresholds_from_data(tab, k)
            table_rows = _build_table_rows_from_dict(thresholds)
            return is_open, dash.no_update, True, json.dumps(thresholds, indent=2), table_rows, friendly.get('conversion_rate_warning'), friendly.get('conversion_rate_critical'), friendly.get('gross_margin_warning'), friendly.get('gross_margin_critical'), friendly.get('aov_warning'), friendly.get('aov_info'), friendly.get('return_rate_warning'), ''
        except Exception as e:
            return (is_open, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, f'Autofill failed: {e}')
    if trig == 'sales-thresholds-save-btn':
        try:
            try:
                base = json.loads(json_text or '{}')
            except Exception:
                base = {}
            for row in (table_data or []):
                kpi = row.get('kpi');
                if not kpi: continue
                if row.get('info') is not None: base[f"{kpi}_info"] = _coerce_number(row.get('info'))
                if row.get('warning') is not None: base[f"{kpi}_warning"] = _coerce_number(row.get('warning'))
                if row.get('critical') is not None: base[f"{kpi}_critical"] = _coerce_number(row.get('critical'))
            if conv_warn not in (None, ''): base['conversion_rate_warning'] = _coerce_number(conv_warn)
            if conv_crit not in (None, ''): base['conversion_rate_critical'] = _coerce_number(conv_crit)
            if gm_warn not in (None, ''): base['gross_margin_warning'] = _coerce_number(gm_warn)
            if gm_crit not in (None, ''): base['gross_margin_critical'] = _coerce_number(gm_crit)
            if aov_warn not in (None, ''): base['aov_warning'] = _coerce_number(aov_warn)
            if aov_info not in (None, ''): base['aov_info'] = _coerce_number(aov_info)
            if ret_warn not in (None, ''): base['return_rate_warning'] = _coerce_number(ret_warn)
            path = _tab_thresholds_path(tab)
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(base, fh, indent=2)
            table_rows = _build_table_rows_from_dict(base)
            curr = dash.callback_context.states.get('sales-thresholds-saved-signal.data')
            try:
                curr = int(curr) if curr is not None else 0
            except Exception:
                curr = 0
            # return values for all 13 Outputs (modal open, saved signal, toast open, json value, table data,
            # then the seven friendly input fields, then the save-feedback children)
            return False, curr + 1, True, json.dumps(base, indent=2), table_rows, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, 'Saved'
        except Exception as e:
            # ensure full arity on error
            return False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, f'Error saving: {e}'
    if trig == 'sales-thresholds-close-btn':
        return False, dash.no_update, False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, ''
    # default: return full arity with no updates
        return dash.no_update, dash.no_update, False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, ''


# Delivery thresholds controller
@callback(
    Output('delivery-thresholds-modal', 'is_open'),
    Output('delivery-thresholds-saved-signal', 'data', allow_duplicate=True),
    Output('delivery-toast', 'is_open', allow_duplicate=True),
    Output('delivery-thresholds-json-textarea', 'value'),
    Output('delivery-thresholds-datatable', 'data'),
    Output('delivery_on_time_delivery_warning_input', 'value'),
    Output('delivery_avg_delivery_time_warning_days_input', 'value'),
    Output('delivery_return_rate_warning_input', 'value'),
    Output('delivery-thresholds-save-feedback', 'children'),
    Input('delivery-edit-thresholds-btn', 'n_clicks'),
    Input('delivery-thresholds-load-btn', 'n_clicks'),
    Input('delivery-thresholds-autofill-btn', 'n_clicks'),
    Input('delivery-thresholds-save-btn', 'n_clicks'),
    Input('delivery-thresholds-close-btn', 'n_clicks'),
    State('delivery-thresholds-json-textarea', 'value'),
    State('delivery-thresholds-modal', 'is_open'),
    State('delivery-thresholds-datatable', 'data'),
    State('delivery_on_time_delivery_warning_input', 'value'),
    State('delivery_avg_delivery_time_warning_days_input', 'value'),
    State('delivery_return_rate_warning_input', 'value'),
    prevent_initial_call=True
)
def delivery_thresholds_controller(edit_click, load_click, autofill_click, save_click, close_click, json_text, is_open, table_data,
                                   ontime_warn, avg_days_warn, ret_warn):
    ctx = dash.callback_context
    if not ctx.triggered:
        return (is_open, dash.no_update, False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, '')
    trig = ctx.triggered[0]['prop_id'].split('.')[0]
    tab = 'delivery'
    if trig == 'delivery-edit-thresholds-btn':
        try:
            data = _load_threshold_json(tab)
            table_rows = _build_table_rows_from_dict(data)
            return True, dash.no_update, False, json.dumps(data, indent=2), table_rows, data.get('on_time_delivery_warning'), data.get('avg_delivery_time_warning_days'), data.get('return_rate_warning'), ''
        except Exception:
            return True, dash.no_update, dash.no_update, dash.no_update, dash.no_update, ''
    if trig == 'delivery-thresholds-load-btn':
        try:
            data = _load_threshold_json(tab)
            table_rows = _build_table_rows_from_dict(data)
            return (is_open, dash.no_update, True, json.dumps(data, indent=2), table_rows, data.get('on_time_delivery_warning'), data.get('avg_delivery_time_warning_days'), data.get('return_rate_warning'), '')
        except Exception as e:
            return (is_open, f"{{\n  \"error\": \"{str(e)}\"\n}}", dash.no_update, dash.no_update, dash.no_update, dash.no_update, str(e))
    if trig == 'delivery-thresholds-autofill-btn':
        try:
            from etl.transforms import DATA as TRANSFORMS_DATA
            k = TRANSFORMS_DATA.get('kpis', {}) or {}
            thresholds, friendly = _autofill_thresholds_from_data(tab, k)
            table_rows = _build_table_rows_from_dict(thresholds)
            return is_open, dash.no_update, True, json.dumps(thresholds, indent=2), table_rows, friendly.get('on_time_delivery_warning'), friendly.get('avg_delivery_time_warning_days'), friendly.get('return_rate_warning'), ''
        except Exception as e:
            return (is_open, dash.no_update, dash.no_update, dash.no_update, dash.no_update, f'Autofill failed: {e}')
    if trig == 'delivery-thresholds-save-btn':
        try:
            try:
                base = json.loads(json_text or '{}')
            except Exception:
                base = {}
            for row in (table_data or []):
                kpi = row.get('kpi');
                if not kpi: continue
                if row.get('info') is not None: base[f"{kpi}_info"] = _coerce_number(row.get('info'))
                if row.get('warning') is not None: base[f"{kpi}_warning"] = _coerce_number(row.get('warning'))
                if row.get('critical') is not None: base[f"{kpi}_critical"] = _coerce_number(row.get('critical'))
            if ontime_warn not in (None, ''): base['on_time_delivery_warning'] = _coerce_number(ontime_warn)
            if avg_days_warn not in (None, ''): base['avg_delivery_time_warning_days'] = _coerce_number(avg_days_warn)
            if ret_warn not in (None, ''): base['return_rate_warning'] = _coerce_number(ret_warn)
            path = _tab_thresholds_path(tab)
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(base, fh, indent=2)
            table_rows = _build_table_rows_from_dict(base)
            curr = dash.callback_context.states.get('delivery-thresholds-saved-signal.data')
            try:
                curr = int(curr) if curr is not None else 0
            except Exception:
                curr = 0
            return False, curr + 1, True, json.dumps(base, indent=2), table_rows, dash.no_update, dash.no_update, dash.no_update, 'Saved'
        except Exception as e:
            return False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, f'Error saving: {e}'
    if trig == 'delivery-thresholds-close-btn':
        return False, dash.no_update, False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, ''
    return dash.no_update, dash.no_update, False, dash.no_update, dash.no_update, dash.no_update, dash.no_update


# Marketing thresholds controller
@callback(
    Output('marketing-thresholds-modal', 'is_open'),
    Output('marketing-thresholds-saved-signal', 'data', allow_duplicate=True),
    Output('marketing-toast', 'is_open', allow_duplicate=True),
    Output('marketing-thresholds-json-textarea', 'value'),
    Output('marketing-thresholds-datatable', 'data'),
    Output('marketing_roas_warning_input', 'value'),
    Output('marketing_roas_critical_input', 'value'),
    Output('marketing_cpa_warning_input', 'value'),
    Output('marketing_low_spend_threshold_input', 'value'),
    Output('marketing-thresholds-save-feedback', 'children'),
    Input('marketing-edit-thresholds-btn', 'n_clicks'),
    Input('marketing-thresholds-load-btn', 'n_clicks'),
    Input('marketing-thresholds-autofill-btn', 'n_clicks'),
    Input('marketing-thresholds-save-btn', 'n_clicks'),
    Input('marketing-thresholds-close-btn', 'n_clicks'),
    State('marketing-thresholds-json-textarea', 'value'),
    State('marketing-thresholds-modal', 'is_open'),
    State('marketing-thresholds-datatable', 'data'),
    State('marketing_roas_warning_input', 'value'),
    State('marketing_roas_critical_input', 'value'),
    State('marketing_cpa_warning_input', 'value'),
    State('marketing_low_spend_threshold_input', 'value'),
    prevent_initial_call=True
)
def marketing_thresholds_controller(edit_click, load_click, autofill_click, save_click, close_click, json_text, is_open, table_data,
                                    roas_warn, roas_crit, cpa_warn, low_spend):
    ctx = dash.callback_context
    if not ctx.triggered:
        return (is_open, dash.no_update, False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, '')
    trig = ctx.triggered[0]['prop_id'].split('.')[0]
    tab = 'marketing'
    if trig == 'marketing-edit-thresholds-btn':
        try:
            data = _load_threshold_json(tab)
            table_rows = _build_table_rows_from_dict(data)
            return True, dash.no_update, False, json.dumps(data, indent=2), table_rows, data.get('roas_warning'), data.get('roas_critical'), data.get('cpa_warning'), data.get('low_spend_threshold'), ''
        except Exception:
            return True, dash.no_update, dash.no_update, dash.no_update, dash.no_update, ''
    if trig == 'marketing-thresholds-load-btn':
        try:
            data = _load_threshold_json(tab)
            table_rows = _build_table_rows_from_dict(data)
            return (is_open, dash.no_update, True, json.dumps(data, indent=2), table_rows, data.get('roas_warning'), data.get('roas_critical'), data.get('cpa_warning'), data.get('low_spend_threshold'), '')
        except Exception as e:
            return (is_open, f"{{\n  \"error\": \"{str(e)}\"\n}}", dash.no_update, dash.no_update, dash.no_update, dash.no_update, str(e))
    if trig == 'marketing-thresholds-autofill-btn':
        try:
            from etl.transforms import DATA as TRANSFORMS_DATA
            k = TRANSFORMS_DATA.get('kpis', {}) or {}
            thresholds, friendly = _autofill_thresholds_from_data(tab, k)
            table_rows = _build_table_rows_from_dict(thresholds)
            return is_open, dash.no_update, True, json.dumps(thresholds, indent=2), table_rows, friendly.get('roas_warning'), friendly.get('roas_critical'), friendly.get('cpa_warning'), friendly.get('low_spend_threshold'), ''
        except Exception as e:
            return (is_open, dash.no_update, dash.no_update, dash.no_update, dash.no_update, f'Autofill failed: {e}')
    if trig == 'marketing-thresholds-save-btn':
        try:
            try:
                base = json.loads(json_text or '{}')
            except Exception:
                base = {}
            for row in (table_data or []):
                kpi = row.get('kpi');
                if not kpi: continue
                if row.get('info') is not None: base[f"{kpi}_info"] = _coerce_number(row.get('info'))
                if row.get('warning') is not None: base[f"{kpi}_warning"] = _coerce_number(row.get('warning'))
                if row.get('critical') is not None: base[f"{kpi}_critical"] = _coerce_number(row.get('critical'))
            if roas_warn not in (None, ''): base['roas_warning'] = _coerce_number(roas_warn)
            if roas_crit not in (None, ''): base['roas_critical'] = _coerce_number(roas_crit)
            if cpa_warn not in (None, ''): base['cpa_warning'] = _coerce_number(cpa_warn)
            if low_spend not in (None, ''): base['low_spend_threshold'] = _coerce_number(low_spend)
            path = _tab_thresholds_path(tab)
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(base, fh, indent=2)
            table_rows = _build_table_rows_from_dict(base)
            curr = dash.callback_context.states.get('marketing-thresholds-saved-signal.data')
            try:
                curr = int(curr) if curr is not None else 0
            except Exception:
                curr = 0
            return False, curr + 1, True, json.dumps(base, indent=2), table_rows, dash.no_update, dash.no_update, dash.no_update, dash.no_update, 'Saved'
        except Exception as e:
            return False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, f'Error saving: {e}'
    if trig == 'marketing-thresholds-close-btn':
        return False, dash.no_update, False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, ''
    return dash.no_update, dash.no_update, False, dash.no_update, dash.no_update, dash.no_update, dash.no_update


# Customer thresholds controller
@callback(
    Output('customer-thresholds-modal', 'is_open'),
    Output('customer-thresholds-saved-signal', 'data', allow_duplicate=True),
    Output('customer-toast', 'is_open', allow_duplicate=True),
    Output('customer-thresholds-json-textarea', 'value'),
    Output('customer-thresholds-datatable', 'data'),
    Output('customer_churn_warning_input', 'value'),
    Output('customer_repeat_rate_info_input', 'value'),
    Output('customer-thresholds-save-feedback', 'children'),
    Input('customer-edit-thresholds-btn', 'n_clicks'),
    Input('customer-thresholds-load-btn', 'n_clicks'),
    Input('customer-thresholds-autofill-btn', 'n_clicks'),
    Input('customer-thresholds-save-btn', 'n_clicks'),
    Input('customer-thresholds-close-btn', 'n_clicks'),
    State('customer-thresholds-json-textarea', 'value'),
    State('customer-thresholds-modal', 'is_open'),
    State('customer-thresholds-datatable', 'data'),
    State('customer_churn_warning_input', 'value'),
    State('customer_repeat_rate_info_input', 'value'),
    prevent_initial_call=True
)
def customer_thresholds_controller(edit_click, load_click, autofill_click, save_click, close_click, json_text, is_open, table_data,
                                   churn_warn, repeat_info):
    ctx = dash.callback_context
    if not ctx.triggered:
        return (is_open, dash.no_update, False, dash.no_update, dash.no_update, dash.no_update, '')
    trig = ctx.triggered[0]['prop_id'].split('.')[0]
    tab = 'customer'
    if trig == 'customer-edit-thresholds-btn':
        try:
            data = _load_threshold_json(tab)
            table_rows = _build_table_rows_from_dict(data)
            return True, dash.no_update, False, json.dumps(data, indent=2), table_rows, data.get('churn_warning'), data.get('repeat_rate_info'), ''
        except Exception:
            return True, dash.no_update, dash.no_update, dash.no_update, ''
    if trig == 'customer-thresholds-load-btn':
        try:
            data = _load_threshold_json(tab)
            table_rows = _build_table_rows_from_dict(data)
            return (is_open, dash.no_update, True, json.dumps(data, indent=2), table_rows, data.get('churn_warning'), data.get('repeat_rate_info'), '')
        except Exception as e:
            return (is_open, f"{{\n  \"error\": \"{str(e)}\"\n}}", dash.no_update, dash.no_update, dash.no_update, str(e))
    if trig == 'customer-thresholds-autofill-btn':
        try:
            from etl.transforms import DATA as TRANSFORMS_DATA
            k = TRANSFORMS_DATA.get('kpis', {}) or {}
            thresholds, friendly = _autofill_thresholds_from_data(tab, k)
            table_rows = _build_table_rows_from_dict(thresholds)
            return is_open, json.dumps(thresholds, indent=2), table_rows, friendly.get('churn_warning'), friendly.get('repeat_rate_info'), ''
        except Exception as e:
            return (is_open, dash.no_update, dash.no_update, dash.no_update, f'Autofill failed: {e}')
    if trig == 'customer-thresholds-save-btn':
        try:
            try:
                base = json.loads(json_text or '{}')
            except Exception:
                base = {}
            for row in (table_data or []):
                kpi = row.get('kpi');
                if not kpi: continue
                if row.get('info') is not None: base[f"{kpi}_info"] = _coerce_number(row.get('info'))
                if row.get('warning') is not None: base[f"{kpi}_warning"] = _coerce_number(row.get('warning'))
                if row.get('critical') is not None: base[f"{kpi}_critical"] = _coerce_number(row.get('critical'))
            if churn_warn not in (None, ''): base['churn_warning'] = _coerce_number(churn_warn)
            if repeat_info not in (None, ''): base['repeat_rate_info'] = _coerce_number(repeat_info)
            path = _tab_thresholds_path(tab)
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(base, fh, indent=2)
            table_rows = _build_table_rows_from_dict(base)
            curr = dash.callback_context.states.get('customer-thresholds-saved-signal.data')
            try:
                curr = int(curr) if curr is not None else 0
            except Exception:
                curr = 0
            return False, curr + 1, True, json.dumps(base, indent=2), table_rows, dash.no_update, dash.no_update, 'Saved'
        except Exception as e:
            return False, dash.no_update, dash.no_update, dash.no_update, f'Error saving: {e}'
    if trig == 'customer-thresholds-close-btn':
        return False, dash.no_update, False, dash.no_update, dash.no_update, dash.no_update, ''
    return dash.no_update, dash.no_update, False, dash.no_update, dash.no_update, dash.no_update
