import os
import json
from typing import Dict, Any
from config.settings import MODEL_STORE_PATH


def _read_status_files() -> Dict[str, Any]:
    out = {}
    try:
        for fname in os.listdir(MODEL_STORE_PATH):
            if fname.endswith('_status.json'):
                path = os.path.join(MODEL_STORE_PATH, fname)
                try:
                    with open(path, 'r') as fh:
                        payload = json.load(fh)
                        out[payload.get('job_id', fname)] = payload
                except Exception:
                    continue
    except Exception:
        # directory might not exist yet
        return {}
    return out


def list_jobs() -> Dict[str, Any]:
    return _read_status_files()


def get_job(job_id: str) -> Dict[str, Any]:
    try:
        path = os.path.join(MODEL_STORE_PATH, f"{job_id}_status.json")
        if not os.path.exists(path):
            return {}
        with open(path, 'r') as fh:
            return json.load(fh)
    except Exception:
        return {}
