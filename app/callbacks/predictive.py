# -*- coding: utf-8 -*-
"""
Callbacks for the Predictive Insights Dashboard tab.
Handles demand forecasting and churn prediction models.
"""
import os
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, dcc, dash_table
import dash
import requests
import json
from dash.exceptions import PreventUpdate
from datetime import datetime, timedelta
import logging

from etl.transforms import DATA
from app.utils import create_kpi_body, create_placeholder_figure
from app.utils.analytics_helpers import run_churn_training_job, FORECAST_MODEL_PATH, CHURN_MODEL_PATH, CHURN_METRICS_PATH
from models.predictors import DemandForecaster, ChurnPredictor
from models.features import build_rfm_features
from app.layout import create_kpi_card, create_graph_card, create_datatable_card
from services.training import enqueue_training

logger = logging.getLogger(__name__)


def _enqueue_job_with_marker(target_fn):
    """Helper: enqueue target_fn and register a small on_complete that writes a marker file and clears in-progress flag."""
    def _on_complete(job_id, result):
        try:
            DATA['model_training_in_progress'] = False
        except Exception:
            pass
        try:
            trigger_path = os.path.join(os.path.dirname(__file__), '..', '..', 'model_store', f'{job_id}_complete.marker')
            with open(trigger_path, 'w') as fh:
                fh.write('done')
        except Exception:
            logger.exception('Failed to write job complete marker')

    try:
        DATA['model_training_in_progress'] = True
    except Exception:
        pass

    return enqueue_training(target_fn, on_complete=_on_complete)


def _get_db_backed_predictions(sales_df, customer_df):
    """Module-level helper: Return predictions DataFrame merged with RFM features.

    This is case-insensitive and will normalize common UI field names like
    'City' and 'Segment' so downstream UI code can rely on them.
    """
    try:
        preds = DATA.get('predictions_df', None)
        if preds is None or getattr(preds, 'empty', True):
            return None

        preds = preds.copy()
        # mapping lowercase -> original
        col_map = {str(c).lower(): c for c in preds.columns}
        lower_cols = [c.lower() for c in preds.columns]
        churn_col = None
        for lc in lower_cols:
            if 'churn' in lc and 'prob' in lc:
                churn_col = col_map[lc]
                break
        ltv_col = None
        for lc in lower_cols:
            if 'ltv' in lc or ('estimated' in lc and 'ltv' in lc):
                ltv_col = col_map[lc]
                break

        # Build features
        rfm = build_rfm_features(sales_df, customer_df, pd.to_datetime(datetime.now()))

        # detect customer id column in preds
        pred_customer_col = None
        for c in ['customerid', 'customer_id', 'id']:
            if c in preds.columns:
                pred_customer_col = c
                break
        if pred_customer_col is None:
            for lc, orig in col_map.items():
                if lc in ('customerid', 'customer_id', 'id'):
                    pred_customer_col = orig
                    break

        if pred_customer_col is None or 'customerid' not in rfm.columns:
            return None

        sel_cols = [pred_customer_col]
        if churn_col:
            sel_cols.append(churn_col)
        if ltv_col:
            sel_cols.append(ltv_col)

        preds_sel = preds[[c for c in sel_cols if c in preds.columns]].copy()
        if churn_col and churn_col in preds_sel.columns:
            preds_sel = preds_sel.rename(columns={churn_col: 'churn_probability'})
        if ltv_col and ltv_col in preds_sel.columns:
            preds_sel = preds_sel.rename(columns={ltv_col: 'estimated_ltv'})
        if pred_customer_col != 'customerid':
            preds_sel = preds_sel.rename(columns={pred_customer_col: 'customerid'})

        merged = rfm.merge(preds_sel, on='customerid', how='left')
        # normalize names for UI
        merged = merged.rename(columns={'churn_probability': 'ChurnProbability', 'estimated_ltv': 'Estimated_LTV'})

        # Ensure UI-friendly casing for City/Segment if present in lowercase
        if 'city' in merged.columns and 'City' not in merged.columns:
            merged = merged.rename(columns={'city': 'City'})
        if 'segment' in merged.columns and 'Segment' not in merged.columns:
            merged = merged.rename(columns={'segment': 'Segment'})

        # Ensure expected columns exist to avoid KeyErrors downstream
        for col in ['ChurnProbability', 'Estimated_LTV']:
            if col not in merged.columns:
                merged[col] = pd.NA

        return merged
    except Exception:
        logger.exception('Failed to build DB-backed predictions')
        return None

# Simple in-memory cache for loaded models to avoid repeated joblib loads
_MODEL_CACHE = {
    'forecast': {'obj': None, 'path': None, 'mtime': None},
    'churn': {'obj': None, 'path': None, 'mtime': None}
}

def _load_cached_model(cache_key: str, path: str):
    import os
    import joblib as _joblib
    entry = _MODEL_CACHE.get(cache_key)
    if entry is None:
        return None
    if not os.path.exists(path):
        entry.update({'obj': None, 'path': path, 'mtime': None})
        return None
    mtime = os.path.getmtime(path)
    # Reload if path changed or file updated
    if entry['obj'] is None or entry.get('path') != path or entry.get('mtime') != mtime:
        try:
            obj = _joblib.load(path)
            entry.update({'obj': obj, 'path': path, 'mtime': mtime})
        except Exception as e:
            logger.error(f"Failed to load model at {path}: {e}", exc_info=True)
            entry.update({'obj': None, 'path': path, 'mtime': None})
    return entry['obj']

def register_predictive_callbacks(app):
    """Registers all callbacks for the predictive analytics dashboard."""

    def _get_db_backed_predictions(sales_df, customer_df):
        """Return a predictions DataFrame joined to RFM features when possible.

        Looks for DATA['predictions_df'] (loaded by ETL). If present and contains
        a churn probability column (case-insensitive), merge it with RFM features
        and return a DataFrame shaped similarly to churn_predictor.predict_churn_probability output.
        """
        try:
            preds = DATA.get('predictions_df', None)
            if preds is None or preds.empty:
                return None

            # normalize column names to lowercase for searching
            preds = preds.copy()
            # create a mapping of lowercase -> original to preserve original names
            col_map = {str(c).lower(): c for c in preds.columns}
            lower_cols = [c.lower() for c in preds.columns]
            # find churn prob column (case-insensitive)
            churn_col = None
            for lc in lower_cols:
                if 'churn' in lc and 'prob' in lc:
                    churn_col = col_map[lc]
                    break
            # find LTV or estimated LTV column
            ltv_col = None
            for lc in lower_cols:
                if 'ltv' in lc or ('estimated' in lc and 'ltv' in lc):
                    ltv_col = col_map[lc]
                    break

            if churn_col is None:
                return None

            # Build RFM features to provide Monetary/Recency/Segment columns
            rfm = build_rfm_features(sales_df, customer_df, pd.to_datetime(datetime.now()))
            # Ensure merge key exists in both
            pred_customer_col = None
            for c in ['customerid', 'customer_id', 'id']:
                if c in preds.columns:
                    pred_customer_col = c
                    break
            if pred_customer_col is None:
                # fall back to lower-case lookup using the mapping
                for lc, orig in col_map.items():
                    if lc in ('customerid', 'customer_id', 'id'):
                        pred_customer_col = orig
                        break

            if pred_customer_col is None or 'customerid' not in rfm.columns:
                return None

            # Build selection list safely, only include columns that exist
            sel_cols = [pred_customer_col]
            if churn_col:
                sel_cols.append(churn_col)
            if ltv_col:
                sel_cols.append(ltv_col)

            preds_sel = preds[[c for c in sel_cols if c in preds.columns]].copy()
            # normalize churn column name if present
            if churn_col and churn_col in preds_sel.columns:
                preds_sel = preds_sel.rename(columns={churn_col: 'churn_probability'})
            if ltv_col and ltv_col in preds_sel.columns:
                preds_sel = preds_sel.rename(columns={ltv_col: 'estimated_ltv'})

            # Ensure the customer id column is named 'customerid' for merge
            if pred_customer_col != 'customerid':
                preds_sel = preds_sel.rename(columns={pred_customer_col: 'customerid'})

            merged = rfm.merge(preds_sel, on='customerid', how='left')
            # normalize names to the UI expected casing
            merged = merged.rename(columns={'churn_probability': 'ChurnProbability', 'estimated_ltv': 'Estimated_LTV'})
            return merged
        except Exception as e:
            logger.exception(f"Failed to build DB-backed predictions: {e}")
            return None

    @app.callback(
        [
            Output('pred-kpi-forecast-rev', 'children'),
            Output('pred-kpi-sim-lift', 'children'),
            Output('forecast-simulation-chart', 'figure')
        ],
        Input('forecast-run-button', 'n_clicks'),
        [
            State('forecast-slider-days', 'value'),
            State('forecast-slider-promo', 'value')
        ],
        prevent_initial_call=True
    )
    def update_forecast_simulation(n, fd, pp):
        if n == 0 or n is None:
            raise PreventUpdate
        # If model artifact missing, provide a lightweight baseline so UI remains usable.
        if not os.path.exists(FORECAST_MODEL_PATH):
            try:
                # Build a daily timeseries from sales and compute a naive forecast:
                from models.features import get_daily_sales_timeseries
                sales_df = DATA.get('sales', pd.DataFrame())
                ts = get_daily_sales_timeseries(sales_df, category='all', channel='all')
                # If insufficient history, show placeholder
                if ts.empty or len(ts) < 14:
                    # Not enough data for baseline; present training CTA + placeholder
                    fig = create_placeholder_figure('Model Not Trained — insufficient historical data for baseline')
                    return create_kpi_body("Forecasted Revenue", "-"), create_kpi_body("Simulated Lift", "-"), fig

                # Simple baseline: rolling mean of last 7 days as forecast for each future day
                last_date = ts['ds'].max()
                recent_mean = ts['y'].tail(7).mean()
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=fd)
                baseline_vals = [recent_mean] * len(future_dates)

                baseline_df = pd.DataFrame({'ds': list(ts['ds']) + list(future_dates), 'y': list(ts['y']) + baseline_vals})
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ts['ds'], y=ts['y'], mode='lines', name='Actual Sales'))
                fig.add_trace(go.Scatter(x=future_dates, y=baseline_vals, mode='lines', name='Baseline (7d MA Projection)'))

                fbv = sum(baseline_vals)
                sl = 0.0
                krt = create_kpi_body("Forecasted Revenue", f"{fbv:,.0f} SAR")
                klt = create_kpi_body("Simulated Lift", f"{sl:,.0f} SAR")
                fig.update_layout(title=f"Baseline (7d MA) — No Model Artifact", hovermode="x unified")
                return krt, klt, fig
            except Exception:
                logger.exception('Failed to compute baseline forecast')
                return create_kpi_body("Error", "-"), create_kpi_body("Error", "-"), create_placeholder_figure("Model Not Trained")

        # Load persisted forecaster and run simulation
        forecaster: DemandForecaster = _load_cached_model('forecast', FORECAST_MODEL_PATH)
        if forecaster is None:
            return create_kpi_body("Error", "-"), create_kpi_body("Error", "-"), create_placeholder_figure("Model Failed to Load")

        forecast_df = forecaster.predict_simulation(fd, pp)
        fig = go.Figure()
        history_df = forecaster.model.history
        fig.add_trace(go.Scatter(x=history_df['ds'], y=history_df['y'], mode='lines', name='Actual Sales'))
        baseline_fc = forecast_df[forecast_df['forecast_type'] == 'Baseline']
        sim_fc = forecast_df[forecast_df['forecast_type'] == 'Simulation']
        fig.add_trace(go.Scatter(x=baseline_fc['ds'], y=baseline_fc['yhat'], mode='lines', name='Baseline Forecast'))
        fig.add_trace(go.Scatter(x=baseline_fc['ds'], y=baseline_fc['yhat_upper'], mode='lines', line=dict(width=0), fill=None, showlegend=False))
        fig.add_trace(go.Scatter(x=baseline_fc['ds'], y=baseline_fc['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', name='Confidence Interval'))

        if not sim_fc.empty and pp > 0:
            fig.add_trace(go.Scatter(x=sim_fc['ds'], y=sim_fc['yhat'], mode='lines', name=f'Simulation (+{pp}%)'))

        asd = pd.to_datetime(datetime.now().date())
        aed = pd.to_datetime(datetime.now().date()) + timedelta(days=fd)
        fbv = baseline_fc[(baseline_fc['ds'] >= asd) & (baseline_fc['ds'] <= aed)]['yhat'].sum()
        fsv = sim_fc[(sim_fc['ds'] >= asd) & (sim_fc['ds'] <= aed)]['yhat'].sum() if not sim_fc.empty else 0.0

        if pd.isna(fsv) or fsv == 0:
            fsv = fbv
        
        sl = fsv - fbv
        krt = create_kpi_body("Forecasted Revenue", f"{fbv:,.0f} SAR")
        klt = create_kpi_body("Simulated Lift", f"{sl:,.0f} SAR")
        fig.update_layout(title=f"Baseline vs. Simulation (+{pp}%)", hovermode="x unified")
        return krt, klt, fig


    @app.callback(
    Output('dataset-readiness-store', 'data'),
    Input('tabs-controller', 'active_tab')
    )
    def dataset_readiness(active_tab):
        """Compute simple dataset readiness metrics for churn training.

        Returns a small summary: total customers, rows used for training (if RFM build succeeds),
        positive/negative counts for churn target when available, and warnings for low sample
        size or extreme class imbalance.
        """
        try:
            if active_tab != 'predictive-tab':
                raise PreventUpdate

            sales_df = DATA.get('sales')
            customers_df = DATA.get('customers')

            if sales_df is None and customers_df is None:
                # write an empty payload when no data is available
                return {
                    'total_customers': 0,
                    'sample_rows': 0,
                    'labeled_pos': None,
                    'labeled_neg': None,
                    'warnings': ['No customer or sales data loaded.']
                }

            # Normalize to DataFrame objects for downstream logic
            sales_df = sales_df if sales_df is not None else pd.DataFrame()
            customers_df = customers_df if customers_df is not None else pd.DataFrame()

            # Attempt to build RFM features; fall back to simple counts if it fails
            rfm = None
            try:
                if not sales_df.empty and not customers_df.empty:
                    rfm = build_rfm_features(sales_df, customers_df, pd.to_datetime(datetime.now()))
            except Exception:
                logger.debug('RFM build failed inside readiness check; falling back to basic counts', exc_info=True)

            total_customers = 0
            if rfm is not None and 'customerid' in rfm.columns:
                total_customers = int(rfm['customerid'].nunique())
                sample_rows = len(rfm)
            else:
                # Fallback: use customers_df if available, otherwise sales-derived unique customers
                if 'customerid' in customers_df.columns:
                    total_customers = int(customers_df['customerid'].nunique())
                    sample_rows = len(customers_df)
                elif 'customerid' in sales_df.columns:
                    total_customers = int(sales_df['customerid'].nunique())
                    sample_rows = len(sales_df)
                else:
                    total_customers = 0
                    sample_rows = 0

            # Detect churn label if present in customers_df or rfm
            churn_label_col = None
            for df in (rfm, customers_df, DATA.get('predictions_df')):
                if df is None:
                    continue
                for c in df.columns:
                    if str(c).lower() in ('churned', 'churn', 'is_churn', 'churn_flag') or 'churn' in str(c).lower() and 'prob' not in str(c).lower():
                        churn_label_col = c
                        break
                if churn_label_col:
                    break

            pos = neg = None
            if churn_label_col is not None:
                # Count positives/negatives in whichever df has the label
                src = rfm if (rfm is not None and churn_label_col in (rfm.columns if rfm is not None else [])) else customers_df if churn_label_col in customers_df.columns else None
                if src is not None:
                    vals = src[churn_label_col].dropna()
                    # Map truthy values to positive
                    pos = int((vals.astype(str).str.lower().isin(['1','true','yes','y','t'])).sum())
                    neg = int(len(vals) - pos)

            # If no label but predictions exist, compute using churn probability
            if pos is None and DATA.get('predictions_df') is not None:
                preds = DATA.get('predictions_df')
                prob_col = next((c for c in preds.columns if 'churn' in str(c).lower() and 'prob' in str(c).lower()), None)
                if prob_col:
                    pr = preds[prob_col].dropna()
                    pos = int((pr > 0.5).sum())
                    neg = int((pr <= 0.5).sum())

            # Evaluate warnings
            warnings = []
            if sample_rows < 200:
                warnings.append('Low sample size: fewer than 200 rows available for training. Consider collecting more data for reliable models.')
            if pos is not None and neg is not None:
                total_lbl = pos + neg
                if total_lbl > 0:
                    ratio = min(pos, neg) / max(pos, neg) if max(pos, neg) > 0 else 0
                    if ratio < 0.1:
                        warnings.append('High class imbalance detected (minority class <10% of labeled samples). Consider resampling or using class weights.')

            # Compose UI
            rows = [html.Div(html.Strong(f"Total customers (estimated): {total_customers}"))]
            rows.append(html.Div(f"Sample rows available: {sample_rows}"))
            if pos is not None and neg is not None:
                rows.append(html.Div(f"Labeled positives: {pos} | negatives: {neg}"))
            else:
                rows.append(html.Div("No labeled churn target detected. The model will train on generated labels or predictions if available."))

            if warnings:
                rows.append(html.Hr())
                for w in warnings:
                    rows.append(dbc.Alert(w, color='warning'))

            # Store a compact readiness payload for UI gating
            readiness_payload = {
                'total_customers': total_customers,
                'sample_rows': sample_rows,
                'labeled_pos': pos,
                'labeled_neg': neg,
                'warnings': warnings
            }

            # Only persist the readiness payload in the dcc.Store; UI rendering occurs in render_churn_tab_content
            return readiness_payload
        except PreventUpdate:
            raise
        except Exception:
            logger.exception('Failed to compute dataset readiness')
            return {
                'total_customers': 0,
                'sample_rows': 0,
                'labeled_pos': None,
                'labeled_neg': None,
                'warnings': ['Unable to compute dataset readiness.']
            }

    @app.callback(
        Output('churn-tab-content-wrapper', 'children'),
        [
            Input('tabs-controller', 'active_tab'),
            Input('model-training-signal-store', 'data'),
            Input('dataset-readiness-store', 'data')
        ]
    )
    def render_churn_tab_content(at, ts, readiness_store):
        logger.debug('render_churn_tab_content called with active_tab=%s, signal=%s', at, ts)
        if at != 'predictive-tab':
            logger.debug('render_churn_tab_content: not predictive-tab, raising PreventUpdate')
            raise PreventUpdate

        models_exist = os.path.exists(CHURN_MODEL_PATH) and os.path.exists(CHURN_METRICS_PATH)
        logger.debug('render_churn_tab_content: models_exist=%s (model:%s metrics:%s)', models_exist, os.path.exists(CHURN_MODEL_PATH), os.path.exists(CHURN_METRICS_PATH))

        # If a training-run was recently enqueued, show an in-progress message
        training_in_progress = False
        try:
            # render an inline dataset readiness panel if the store has data
            readiness_div = None
            try:
                if readiness_store:
                    rows = [html.Div(html.Strong(f"Total customers (estimated): {readiness_store.get('total_customers', 0)}"))]
                    rows.append(html.Div(f"Sample rows available: {readiness_store.get('sample_rows', 0)}"))
                    lp = readiness_store.get('labeled_pos')
                    ln = readiness_store.get('labeled_neg')
                    if lp is not None and ln is not None:
                        rows.append(html.Div(f"Labeled positives: {lp} | negatives: {ln}"))
                    if readiness_store.get('warnings'):
                        rows.append(html.Hr())
                        for w in readiness_store.get('warnings'):
                            rows.append(dbc.Alert(w, color='warning'))
                    readiness_div = dbc.Card(dbc.CardBody([html.H6("Dataset Readiness")] + rows), class_name="mt-3 mb-3")
            except Exception:
                logger.debug('Failed to render readiness inline', exc_info=True)
            training_in_progress = DATA.get('model_training_in_progress', False)
        except Exception:
            training_in_progress = False
        if training_in_progress and not models_exist:
            logger.debug('render_churn_tab_content: training in progress and no models -> showing in-progress message')
            return dbc.Alert([html.H4("Model Training In Progress"), html.P("A training job is running. The predictive models will appear here once complete.")], color="info")
        if not models_exist:
            logger.debug('render_churn_tab_content: no model artifacts found -> showing not trained message')
            return dbc.Alert([
                html.H4("Model Not Trained"),
                html.P("The customer churn prediction model has not been trained yet."),
                html.Hr(),
                dbc.Button("Run Training Job", id="run-manual-churn-train-btn", color="primary")
            ], color="warning")

        try:
            churn_predictor: ChurnPredictor = _load_cached_model('churn', CHURN_MODEL_PATH)
            # metrics are small — keep loading directly for now (cached later if needed)
            try:
                metrics: dict = joblib.load(CHURN_METRICS_PATH) if os.path.exists(CHURN_METRICS_PATH) else {}
            except Exception:
                metrics = {}
            sales_df, customer_df = DATA.get('sales', pd.DataFrame()), DATA.get('customers', pd.DataFrame())

            if sales_df.empty or customer_df.empty:
                logger.debug('render_churn_tab_content: required DATA missing sales_rows=%s customer_rows=%s', getattr(sales_df, 'shape', None), getattr(customer_df, 'shape', None))
                return dbc.Alert(html.P("Model artifacts exist, but no data is loaded. Please refresh the data."), color="danger")

            # Prefer persisted predictions (loaded by ETL) to ensure dashboard uses canonical scoring
            predictions_df = _get_db_backed_predictions(sales_df, customer_df)
            if predictions_df is None:
                # fallback to on-the-fly prediction using the loaded model
                try:
                    predictions_df = churn_predictor.predict_churn_probability(build_rfm_features(sales_df, customer_df, pd.to_datetime(datetime.now())))
                except Exception as e:
                    logger.exception('On-the-fly prediction failed, returning friendly message')
                    logger.debug('render_churn_tab_content: predictions_df was None and on-the-fly prediction failed')
                    return dbc.Alert(html.P("Model artifacts exist but failed to produce predictions. Consider retraining."), color="danger")

            # Ensure LTV column exists with consistent casing to avoid KeyErrors downstream
            if 'Estimated_LTV' not in predictions_df.columns and 'estimated_ltv' in predictions_df.columns:
                predictions_df = predictions_df.rename(columns={'estimated_ltv': 'Estimated_LTV'})
            if 'Estimated_LTV' not in predictions_df.columns:
                predictions_df['Estimated_LTV'] = pd.NA

            likely_churn_mask = predictions_df['ChurnProbability'] > 0.5
            churn_rate_pct = (predictions_df[likely_churn_mask]['customerid'].nunique() / predictions_df['customerid'].nunique()) * 100 if predictions_df['customerid'].nunique() > 0 else 0
            at_risk_revenue = predictions_df[likely_churn_mask]['Monetary'].sum()
            # Compute active LTV safely when column may be missing entirely
            try:
                active_ltv = predictions_df[~likely_churn_mask]['Estimated_LTV'].mean()
            except Exception:
                active_ltv = None

            # Prefer canonical ETL KPIs if present
            try:
                from etl import transforms
                etl_kpis = transforms.DATA.get('kpis', {}) or {}
            except Exception:
                etl_kpis = {}

            churn_val = etl_kpis.get('churn_rate') or (churn_rate_pct / 100.0 if churn_rate_pct else None)
            ltv_val = etl_kpis.get('clv') or active_ltv

            kpi_churn_rate = create_kpi_body("Predicted Churn Rate", f"{(churn_val*100) if churn_val is not None else churn_rate_pct:.1f}%")
            kpi_auc = create_kpi_body("Model AUC Score", f"{metrics.get('auc', 0):.3f}")
            kpi_risk_rev = create_kpi_body("Total At-Risk Revenue", f"{at_risk_revenue:,.0f} SAR")
            kpi_ltv = create_kpi_body("Avg. LTV (Active)", f"{ltv_val:,.0f} SAR" if (ltv_val is not None and not pd.isna(ltv_val)) else "N/A")

            # Compute SHAP/key drivers gracefully: don't raise — fallback to placeholder when unavailable
            kd_df = None
            try:
                kd_df = churn_predictor.get_key_drivers_df()
            except Exception:
                logger.exception('Failed to obtain key drivers (SHAP) from model')

            if kd_df is None or getattr(kd_df, 'empty', True):
                logger.debug('Key drivers missing or empty; using placeholder figure')
                fig_drivers = create_placeholder_figure('Feature importance not available')
            else:
                try:
                    fig_drivers = px.bar(kd_df.head(10), y='Feature', x='FeatureImportance', orientation='h', title='Top 10 Churn Drivers').update_layout(yaxis={'categoryorder': 'total ascending'})
                except Exception:
                    logger.exception('Failed to plot key drivers (SHAP)')
                    fig_drivers = create_placeholder_figure('Feature importance not available')
            churn_hist_fig = px.histogram(predictions_df, x='ChurnProbability', nbins=50, title="Churn Probability Distribution")

            wanted_cols = ['customerid', 'City', 'Segment', 'Recency', 'Monetary', 'ChurnProbability', 'Estimated_LTV']
            present_cols = [c for c in wanted_cols if c in predictions_df.columns]
            at_risk_df = predictions_df[likely_churn_mask][present_cols].head(50)
            # Safe formatting: only format if the column exists and is numeric
            if 'ChurnProbability' in at_risk_df.columns:
                try:
                    at_risk_df['ChurnProbability'] = at_risk_df['ChurnProbability'].map(lambda v: '{:.1%}'.format(v) if v is not None else 'N/A')
                except Exception:
                    at_risk_df['ChurnProbability'] = at_risk_df['ChurnProbability'].astype(str)
            if 'Estimated_LTV' in at_risk_df.columns:
                try:
                    at_risk_df['Estimated_LTV'] = at_risk_df['Estimated_LTV'].map(lambda v: '{:,.0f} SAR'.format(v) if pd.notna(v) else 'N/A')
                except Exception:
                    at_risk_df['Estimated_LTV'] = at_risk_df['Estimated_LTV'].astype(str)
            if 'Monetary' in at_risk_df.columns:
                try:
                    at_risk_df['Monetary'] = at_risk_df['Monetary'].map(lambda v: '{:,.0f}'.format(v) if pd.notna(v) else '0')
                except Exception:
                    at_risk_df['Monetary'] = at_risk_df['Monetary'].astype(str)

            table_cols = [{"name": i.replace("_", " ").title(), "id": i} for i in at_risk_df.columns]
            table_data = at_risk_df.to_dict('records')
            
            logger.debug('render_churn_tab_content: successfully prepared churn tab content; returning layout')
            # Build a cleaner 2-column responsive layout:
            # - Top: KPI row
            # - Middle: Two columns (left: Key drivers + distribution; right: At-risk table)
            # - Readiness card (if available) is inserted at the very top
            content_children = []
            # KPIs: render as two rows of two cards for better horizontal layout on narrow screens
            kpi_row_top = dbc.Row([
                create_kpi_card(kpi_id="pred-kpi-churn-rate-card", title="Predicted Churn Rate", color="danger", width=6, children=kpi_churn_rate),
                create_kpi_card(kpi_id="pred-kpi-churn-auc-card", title="Model AUC Score", color="info", width=6, children=kpi_auc),
            ], className="mb-3")
            kpi_row_bottom = dbc.Row([
                create_kpi_card(kpi_id="pred-kpi-churn-revenue-card", title="Total At-Risk Revenue", color="warning", width=6, children=kpi_risk_rev),
                create_kpi_card(kpi_id="pred-kpi-ltv-card", title="Avg. LTV (Active)", color="success", width=6, children=kpi_ltv),
            ], className="mb-3")

            # Stacked full-width blocks: Key Drivers, Distribution, At-Risk table
            drivers_block = create_graph_card('churn-key-drivers-chart', title="Key Drivers of Churn", width=12, height=420, children=dcc.Graph(figure=fig_drivers, style={'height': '380px'}))
            distribution_block = create_graph_card('churn-distribution-chart', title="Churn Probability Distribution", width=12, height=360, children=dcc.Graph(figure=churn_hist_fig, style={'height': '320px'}))
            at_risk_block = create_datatable_card('churn-at-risk-table', title="Top Customers At-Risk of Churn", width=12, children=[dash_table.DataTable(
                    columns=table_cols,
                    data=table_data,
                    page_size=10,
                    sort_action='native',
                    style_table={'overflowX': 'auto', 'height': '320px', 'borderRadius': '8px'},
                    style_header={'backgroundColor': 'rgba(255,255,255,0.06)', 'color': '#e8f0ff', 'fontWeight': '700', 'border': 'none'},
                    style_cell={'backgroundColor': 'rgba(255,255,255,0.02)', 'color': '#f5f8ff', 'textAlign': 'left', 'padding': '0.4rem 0.55rem', 'border': 'none'},
                    style_data={'backgroundColor': 'rgba(255,255,255,0.02)'},
                )])

            # Actionable Customer Lists (Predictive tab integration, unique IDs)
            actionable_lists_block = dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.H4("Actionable Customer Lists"), md=6),
                        dbc.Col(
                            dcc.RadioItems(
                                id='pred-customer-list-selector',
                                options=[
                                    {'label': 'Top-Value Customers', 'value': 'top_value'},
                                    {'label': 'High Churn Risk', 'value': 'churn_risk'},
                                    {'label': 'New Customers', 'value': 'new'}
                                ],
                                value='top_value',
                                inline=True,
                                labelClassName="me-3"
                            ),
                            md=6
                        )
                    ], align="center"),
                    dbc.Row([
                        create_datatable_card(table_id='pred-customer-data-table', title="", width=10),
                        dbc.Col(
                            dbc.Button(["Export ", html.I(className="bi bi-download")], id="pred-export-csv-button", color="primary", className="mt-3 w-100"),
                            lg=2, md=12, sm=12
                        )
                    ])
                ])
            )

            content_children.extend([kpi_row_top, kpi_row_bottom, html.Hr(), drivers_block, html.Hr(), distribution_block, html.Hr(), at_risk_block, html.Hr(), actionable_lists_block])
            if readiness_div is not None:
                content_children.insert(0, readiness_div)
            return html.Div(content_children, className="predictive-churn-content")
        except Exception as e:
            # Don't delete model artifacts on any exception - this can cause
            # transient failures to become permanent data loss for users.
            logger.error(f"Failed to render churn dashboard: {e}", exc_info=True)
            logger.debug('render_churn_tab_content: caught exception while rendering churn tab')
            # Return a friendly message prompting retrain if artifacts are actually missing or corrupt
            return dbc.Alert(html.Div([
                html.H4("Unable to Load Predictive Insights"),
                html.P("An error occurred while preparing the churn dashboard. The model artifacts may be missing or incompatible. Please check the server logs and retrain if necessary."),
                html.Div(str(e))
            ]), color="danger")

    @app.callback(
        Output('model-training-signal-store', 'data'),
        [
            Input('alert-poll-interval', 'n_intervals')
        ],
        State('model-training-signal-store', 'data'),
        prevent_initial_call=False
    )
    def trigger_churn_model_training(n_intervals, current_signal):
        """Poll job status and update the signal when jobs complete or run."""
        try:
            # Avoid making an HTTP call to our own server from within a callback
            # (this can deadlock in the dev server). Call the status service directly.
            from services.status import list_jobs
            jobs = list_jobs()
            # If any job is running, set signal to 1; if any succeeded recently, increment
            running = any(v.get('status') == 'running' for v in (jobs or {}).values())
            success = any(v.get('status') == 'success' for v in (jobs or {}).values())
            if running:
                try:
                    DATA['model_training_in_progress'] = True
                except Exception:
                    pass
                return (current_signal or 0)
            if success:
                # clear in-progress and bump the signal so UI can refresh
                try:
                    DATA['model_training_in_progress'] = False
                except Exception:
                    pass
                return (current_signal or 0) + 1
        except Exception:
            # polling failure is non-fatal; keep current signal
            return (current_signal or 0)


    # --- Combined churn modal handler: open/confirm/cancel in one callback ---
    @app.callback(
        Output('confirm-train-modal', 'is_open'),
        [
            Input('run-manual-churn-train-btn', 'n_clicks'),
            Input('train-churn-btn', 'n_clicks'),
            Input('confirm-train-yes', 'n_clicks'),
            Input('confirm-train-no', 'n_clicks')
        ],
        [State('dataset-readiness-store', 'data')],
        prevent_initial_call=True
    )
    def handle_churn_modal(run_clicks, train_clicks, yes_clicks, no_clicks, readiness):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        triggered = ctx.triggered[0]['prop_id'].split('.')[0]

        # Cancel -> close modal
        if triggered == 'confirm-train-no':
            return False

        # Open request from manual train button or the new visible button
        if triggered in ('run-manual-churn-train-btn', 'train-churn-btn'):
            warnings = (readiness or {}).get('warnings') if readiness is not None else None
            if warnings:
                return True  # open modal
            # No warnings — enqueue immediately
            try:
                DATA['model_training_in_progress'] = True
            except Exception:
                pass
            job_id = _enqueue_job_with_marker(run_churn_training_job)
            logger.info(f'Enqueued churn training job: {job_id}')
            return False

        # Confirm -> enqueue and close modal
        if triggered == 'confirm-train-yes':
            try:
                DATA['model_training_in_progress'] = True
            except Exception:
                pass
            job_id = _enqueue_job_with_marker(run_churn_training_job)
            logger.info(f'Enqueued churn training job (confirmed): {job_id}')
            return False

        raise PreventUpdate


    # --- Combined forecast modal handler: open/confirm/cancel in one callback ---
    @app.callback(
        Output('confirm-forecast-modal', 'is_open'),
        [
            Input('train-forecast-btn', 'n_clicks'),
            Input('confirm-forecast-yes', 'n_clicks'),
            Input('confirm-forecast-no', 'n_clicks')
        ],
        [State('dataset-readiness-store', 'data')],
        prevent_initial_call=True
    )
    def handle_forecast_modal(train_clicks, yes_clicks, no_clicks, readiness):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        triggered = ctx.triggered[0]['prop_id'].split('.')[0]

        if triggered == 'confirm-forecast-no':
            return False

        try:
            from etl.schedules import run_forecast_training_pipeline
        except Exception:
            logger.exception('Failed to import forecast training pipeline')
            raise PreventUpdate

        if triggered == 'train-forecast-btn':
            warnings = (readiness or {}).get('warnings') if readiness is not None else None
            if warnings:
                return True
            try:
                DATA['model_training_in_progress'] = True
            except Exception:
                pass
            job_id = _enqueue_job_with_marker(run_forecast_training_pipeline)
            logger.info(f'Enqueued forecast training job: {job_id}')
            return False

        if triggered == 'confirm-forecast-yes':
            try:
                DATA['model_training_in_progress'] = True
            except Exception:
                pass
            job_id = _enqueue_job_with_marker(run_forecast_training_pipeline)
            logger.info(f'Enqueued forecast training job (confirmed): {job_id}')
            return False

        raise PreventUpdate


    # --- Model registry & job-list callbacks ---
    @app.callback(
        Output('models-registry-table', 'data'),
        Input('jobs-poll-interval', 'n_intervals')
    )
    def refresh_model_registry(n):
        try:
            from services.model_registry import list_models
            rows = list_models()
            # Convert trained_at to isoformat for display
            for r in rows:
                if 'trained_at' in r and r['trained_at'] is not None:
                    try:
                        r['trained_at'] = r['trained_at'].isoformat()
                    except Exception:
                        r['trained_at'] = str(r['trained_at'])
                if 'metrics' in r and isinstance(r['metrics'], str):
                    try:
                        r['metrics'] = r['metrics']
                    except Exception:
                        r['metrics'] = r.get('metrics', '')
            return rows
        except Exception:
            logger.exception('Failed to refresh model registry')
            return []


    @app.callback(
        Output('jobs-status-table', 'data'),
        Input('jobs-poll-interval', 'n_intervals')
    )
    def refresh_jobs_table(n):
        try:
            from services.status import list_jobs
            jobs = list_jobs()
            out = []
            for jid, payload in (jobs or {}).items():
                # Normalize updated_at to a string (DataTable expects primitives)
                updated = payload.get('updated_at')
                try:
                    # datetime-like -> isoformat
                    if hasattr(updated, 'isoformat'):
                        updated = updated.isoformat()
                    else:
                        updated = str(updated) if updated is not None else ''
                except Exception:
                    updated = str(updated)

                # Ensure details is a primitive (string/number/bool). JSON-encode dicts/lists.
                details = payload.get('details')
                try:
                    if isinstance(details, (dict, list)):
                        details = json.dumps(details)
                    else:
                        details = str(details) if details is not None else ''
                except Exception:
                    details = str(details)

                out.append({
                    'job_id': jid,
                    'status': payload.get('status'),
                    'updated_at': updated,
                    'details': details
                })
            return out
        except Exception:
            logger.exception('Failed to refresh jobs table')
            return []


    @app.callback(
        [Output('job-progress-label', 'children'), Output('job-progress-bar', 'value'), Output('job-progress-bar', 'animated')],
        Input('jobs-poll-interval', 'n_intervals')
    )
    def update_job_progress(n):
        try:
            from services.status import list_jobs
            jobs = list_jobs() or {}
            # Prefer a running job; otherwise take the most recent by updated_at
            running = {jid: p for jid, p in jobs.items() if p.get('status') == 'running'}
            target = None
            if running:
                # pick the most recently updated running job
                target = max(running.items(), key=lambda kv: kv[1].get('updated_at', 0))[1]
            else:
                # pick the most recently updated job overall
                if jobs:
                    target = max(jobs.items(), key=lambda kv: kv[1].get('updated_at', 0))[1]

            if not target:
                return 'No active training jobs.', 0, False

            status = target.get('status', 'unknown')
            details = target.get('details') or {}
            # details may be JSON-dumped strings in some versions; attempt to coerce
            if isinstance(details, str):
                try:
                    details = json.loads(details)
                except Exception:
                    details = {'message': details}

            percent = 0
            animated = False
            if isinstance(details, dict) and 'percent' in details:
                try:
                    percent = int(details.get('percent') or 0)
                    percent = max(0, min(100, percent))
                except Exception:
                    percent = 0
            else:
                # Map common phase strings to rough progress
                phase = (details.get('phase') if isinstance(details, dict) else None) or status
                phase_map = {
                    'initializing': 5,
                    'generating_synthetic_data': 10,
                    'building_features': 20,
                    'features_built': 40,
                    'training_model': 50,
                    'model_trained': 80,
                    'artifacts_saved': 90,
                    'success': 100,
                    'failed': 100
                }
                percent = phase_map.get(phase, 0)

            label = f"Job {target.get('job_id', '')} — {status.capitalize()} ({percent}%)"
            animated = status == 'running' and percent < 100
            return label, percent, animated
        except Exception:
            logger.exception('Failed to update job progress')
            return 'Progress unavailable', 0, False


    @app.callback(
        Output('models-registry-table', 'columns'),
        Input('jobs-poll-interval', 'n_intervals')
    )
    def refresh_models_columns(n):
        # Keep static columns but ensure existence for the DataTable
        return [{'name':'Model Name','id':'model_name'},{'name':'Trained At','id':'trained_at'},{'name':'Artifact Path','id':'artifact_path'},{'name':'Metrics','id':'metrics'}]


    # Note: model registry data is refreshed by `refresh_model_registry` above.
    # Removed duplicate `refresh_models_data` and noop callback to avoid Dash
    # Duplicate callback outputs error. If you need additional triggers for
    # the `model-training-signal-store`, implement them inside the
    # `trigger_churn_model_training` callback or use `allow_duplicate=True`.


    @app.callback(
        Output('model-selector-dropdown', 'options'),
        Input('jobs-poll-interval', 'n_intervals')
    )
    def populate_model_selector(n):
        try:
            from services.model_registry import list_models
            rows = list_models()
            return [{'label': f"{r.get('model_name')} ({r.get('trained_at')})", 'value': r.get('id')} for r in rows]
        except Exception:
            logger.exception('Failed to populate model selector')
            return []

    # --- Predictive Actionable Customer Lists: populate table ---
    @app.callback(
        [
            Output('pred-customer-data-table', 'data'),
            Output('pred-customer-data-table', 'columns')
        ],
        [
            Input('pred-customer-list-selector', 'value'),
            Input('model-training-signal-store', 'data')
        ]
    )
    def update_pred_actionable_lists(selected_list, _signal):
        try:
            predictions_df = DATA.get('predictions_df', None)
            if predictions_df is not None and not getattr(predictions_df, 'empty', True):
                preds = predictions_df.copy()
                # normalize names
                lmap = {c.lower(): c for c in preds.columns}
                if 'churn_probability' in lmap and 'ChurnProbability' not in preds.columns:
                    preds = preds.rename(columns={lmap['churn_probability']: 'ChurnProbability'})
                if 'estimated_ltv' in lmap and 'Estimated_LTV' not in preds.columns:
                    preds = preds.rename(columns={lmap['estimated_ltv']: 'Estimated_LTV'})
            else:
                preds = None

            customer_analysis_df = DATA.get('customer_analysis_df', pd.DataFrame())

            if selected_list == 'top_value':
                if not customer_analysis_df.empty:
                    df = customer_analysis_df.copy()
                    sort_col = 'monetary' if 'monetary' in df.columns else ('Monetary' if 'Monetary' in df.columns else None)
                    cols = [c for c in ['customerid', 'city', 'segment', 'monetary', 'Monetary', 'frequency', 'recency'] if c in df.columns]
                    df = df[cols]
                    if sort_col:
                        df = df.sort_values(sort_col, ascending=False)
                    df = df.head(50)
                elif preds is not None and not preds.empty and 'Monetary' in preds.columns:
                    cols = [c for c in ['customerid', 'City', 'Segment', 'Monetary', 'Recency'] if c in preds.columns]
                    df = preds[cols].sort_values('Monetary', ascending=False).head(50)
                else:
                    df = pd.DataFrame()
            elif selected_list == 'churn_risk':
                if preds is not None and not preds.empty and 'ChurnProbability' in preds.columns:
                    df = preds[preds['ChurnProbability'] > 0.5]
                    cols = [c for c in ['customerid', 'City', 'Segment', 'Monetary', 'Recency', 'ChurnProbability'] if c in df.columns]
                    df = df[cols].sort_values('ChurnProbability', ascending=False).head(50)
                elif not customer_analysis_df.empty and 'status' in customer_analysis_df.columns:
                    df = customer_analysis_df[customer_analysis_df['status'] == 'Churn Risk']
                    cols = [c for c in ['customerid', 'city', 'segment', 'monetary', 'frequency', 'recency', 'last_purchase_date'] if c in df.columns]
                    df = df[cols].head(50)
                else:
                    df = pd.DataFrame()
            elif selected_list == 'new':
                if not customer_analysis_df.empty and 'status' in customer_analysis_df.columns:
                    df = customer_analysis_df[customer_analysis_df['status'] == 'New']
                    cols = [c for c in ['customerid', 'city', 'segment', 'monetary', 'frequency', 'recency', 'joindate'] if c in df.columns]
                    df = df[cols].head(50)
                else:
                    df = pd.DataFrame()
            else:
                df = pd.DataFrame()

            if df is None or df.empty:
                return [], []
            return df.to_dict('records'), [{'name': c, 'id': c} for c in df.columns]
        except Exception:
            logger.exception('Failed to update predictive actionable customer lists')
            return [], []

    # --- Predictive Actionable Customer Lists: Export ---
    @app.callback(
        Output('pred-download-dataframe-csv', 'data'),
        Input('pred-export-csv-button', 'n_clicks'),
        State('pred-customer-list-selector', 'value'),
        prevent_initial_call=True
    )
    def export_pred_actionable_lists(n, selected_list):
        if not n:
            raise PreventUpdate
        data, cols = update_pred_actionable_lists(selected_list, DATA.get('model_training_signal', 0))
        if not data:
            raise PreventUpdate
        df = pd.DataFrame(data)
        return dcc.send_data_frame(df.to_csv, f"{selected_list}_customers_{datetime.now().strftime('%Y-%m-%d')}.csv", index=False)


    # --- Forecast training enqueue: allow users to trigger the Demand Forecaster training ---
    @app.callback(
        Output('train-forecast-btn', 'children'),
        [Input('train-forecast-btn', 'n_clicks')],
        [State('dataset-readiness-store', 'data'), State('model-training-signal-store', 'data')],
        prevent_initial_call=True
    )
    def handle_train_forecast_click(n_clicks, readiness, current_signal):
        """Enqueue the forecast training pipeline. Update the model-training-signal-store by returning a friendly label for the button."""
        if not n_clicks:
            raise PreventUpdate

        # If dataset readiness warns, prefer user to check; still allow enqueue
        warnings = (readiness or {}).get('warnings') if readiness is not None else None

        try:
            from etl.schedules import run_forecast_training_pipeline
        except Exception:
            logger.exception('Failed to import forecast training pipeline')
            raise PreventUpdate

        # Enqueue using the shared helper which writes completion marker and flips in-progress flag
        try:
            job_id = _enqueue_job_with_marker(run_forecast_training_pipeline)
            logger.info(f'Enqueued forecast training job: {job_id}')
            # bump the model-training signal so UI polls refresh; if current_signal is None start at 1
            try:
                new_signal = (current_signal or 0) + 1
            except Exception:
                new_signal = 1
            # Return a temporary label informing the user
            return f"Training Enqueued ({job_id})"
        except Exception:
            logger.exception('Failed to enqueue forecast training')
            return "Enqueue Failed"


    @app.callback(
        [Output('shap-summary-table', 'data'), Output('shap-summary-chart', 'figure')],
        Input('model-selector-dropdown', 'value')
    )
    def update_shap_display(model_id):
        if not model_id:
            return [], {}
        try:
            from services.model_registry import list_models
            from app.utils.analytics_helpers import get_shap_summary
            rows = list_models()
            model = next((r for r in rows if r.get('id') == model_id), None)
            if not model:
                return [], {}
            path = model.get('artifact_path')
            summary = get_shap_summary(path)
            # Create a simple bar chart
            try:
                import plotly.express as px
                fig = px.bar(summary, x='MeanAbsSHAP', y='Feature', orientation='h', title='Top SHAP Features')
            except Exception:
                fig = {}
            return summary, fig
        except Exception:
            logger.exception('Failed to update SHAP display')
            return [], {}
