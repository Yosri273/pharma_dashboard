"""Programmatic wrapper for scripts/validate_metrics_end_to_end.py
so the app can call it during startup and expose results via an API.
"""
import os
import json
import importlib.util


def _load_validator_module():
    # Load scripts/validate_metrics_end_to_end.py as a module by path
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    path = os.path.join(base, 'scripts', 'validate_metrics_end_to_end.py')
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location('validate_metrics_end_to_end', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_validator(base_path=None, save=True):
    try:
        mod = _load_validator_module()
        if mod and hasattr(mod, 'run_all'):
            res = mod.run_all()
        else:
            res = {'error': 'validator_unavailable'}
    except Exception:
        res = {'error': 'validator_exception'}
    if save:
        outdir = os.path.join(base_path or os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')), 'cache-directory')
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, 'metrics_validation.json'), 'w') as f:
            json.dump(res, f, indent=2, default=str)
    return res
