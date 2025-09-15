"""Yosri Analytics Hub - Application Bootstrap

Creates the Dash app, loads data, registers layout and callbacks, and
exposes HTTP endpoints used by the UI and automation.

This module should be imported only by runtime entrypoints (dev server, WSGI),
so that importing subpackages like `app.utils.*` doesn't start the web app.
"""

import dash
import logging
import sys
import dash_bootstrap_components as dbc
import threading
import os

from app.layout import create_main_layout
from app.callbacks import register_all_callbacks
from etl.transforms import initialize_data, DATA as TRANSFORMS_DATA
from services.db import get_engine
from services.status import list_jobs, get_job
from app.utils.sessions_report import generate_sessions_report
from app.utils.metrics_validator import run_validator


# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.info("--- Starting Yosri Analytics Hub ---")


# --- App bootstrap ---
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1, maximum-scale=1"}],
)
server = app.server
app.title = "Yosri Analytics Hub"


# --- Data initialization ---
engine = get_engine()
initialize_data(engine)


def _generate_startup_reports():
    """Generate cached reports (runs in a background thread)."""
    try:
        TRANSFORMS_DATA['sessions_report'] = generate_sessions_report()
    except Exception:
        TRANSFORMS_DATA['sessions_report'] = {}
    try:
        TRANSFORMS_DATA['metrics_validation'] = run_validator()
    except Exception:
        TRANSFORMS_DATA['metrics_validation'] = {}


threading.Thread(target=_generate_startup_reports, daemon=True).start()


# --- Layout and callbacks ---
app.layout = create_main_layout()
register_all_callbacks(app)
logger.info("Application ready.")


# --- HTTP endpoints ---
@server.route('/api/jobs')
def _list_jobs():
    try:
        return list_jobs()
    except Exception:
        return {}


@server.route('/api/job/<job_id>')
def _get_job(job_id):
    try:
        return get_job(job_id)
    except Exception:
        return {}


@server.route('/api/sessions_report')
def _get_sessions_report():
    try:
        return TRANSFORMS_DATA.get('sessions_report', {})
    except Exception:
        return {}


@server.route('/api/metrics_validation')
def _get_metrics_validation():
    try:
        return TRANSFORMS_DATA.get('metrics_validation', {})
    except Exception:
        return {}


@server.route('/api/kpis')
def _get_kpis():
    try:
        k = TRANSFORMS_DATA.get('kpis', {}) or {}
        import json as _json

        def _serialize(obj):
            try:
                import plotly.graph_objects as go
                import numpy as np
                import pandas as pd
            except Exception:
                go = np = pd = None
            if go is not None and isinstance(obj, go.Figure):
                try:
                    return obj.to_plotly_json()
                except Exception:
                    return str(obj)
            if pd is not None:
                if isinstance(obj, pd.Timestamp):
                    return str(obj)
                if isinstance(obj, pd.Series):
                    return obj.tolist()
                if isinstance(obj, pd.DataFrame):
                    return obj.to_dict(orient='records')
            if isinstance(obj, dict):
                return {str(k): _serialize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_serialize(v) for v in obj]
            try:
                if obj is None or isinstance(obj, (str, bool, int, float)):
                    return obj
                if hasattr(obj, '__float__'):
                    return float(obj)
            except Exception:
                pass
            return str(obj)

        payload = _serialize(k)
        try:
            body = _json.dumps(payload, default=str)
        except Exception:
            body = _json.dumps({}, default=str)
        return server.response_class(body, mimetype='application/json')
    except Exception:
        return {}


# --- Lightweight admin endpoint: reload thresholds file without restarting the server ---
@server.route('/api/reload_thresholds')
def _reload_thresholds():
    try:
        import json
        path = os.path.join(os.getcwd(), 'model_store', 'recommendation_thresholds.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            return data
        return {}
    except Exception:
        return {}


# --- Health endpoints ---
@server.route('/healthz')
def _healthz():
    return {"status": "ok", "app": app.title}


@server.route('/readyz')
def _readyz():
    try:
        ready = bool(TRANSFORMS_DATA)
        return {"status": "ready" if ready else "starting"}
    except Exception:
        return {"status": "starting"}
