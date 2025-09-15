"""
Lightweight background training runner.

Provides a small thread-based job enqueue for local development. Jobs are
run in background threads; each job writes a basic status file under
`model_store/<job_id>_status.json` and supports a callback when complete.

This is intentionally simple so it can be replaced by Celery/airflow later.
"""
import threading
import uuid
import time
import json
import os
import logging
import tempfile
from typing import Callable, Any

from config.settings import MODEL_STORE_PATH

logger = logging.getLogger(__name__)


def _write_status(job_id: str, status: str, details: dict = None):
    path = os.path.join(MODEL_STORE_PATH, f"{job_id}_status.json")
    payload = {"job_id": job_id, "status": status, "details": details or {}, "updated_at": time.time()}
    try:
        # Ensure model store directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Write atomically: write to a temp file in the same dir then atomically replace
        dirpath = os.path.dirname(path) or '.'
        with tempfile.NamedTemporaryFile('w', dir=dirpath, delete=False) as tf:
            json.dump(payload, tf)
            tf.flush()
            try:
                os.fsync(tf.fileno())
            except Exception:
                # Some platforms/FS may not support fsync on temp files; ignore if it fails
                pass
            tmpname = tf.name
        # Atomic replace
        os.replace(tmpname, path)
    except Exception:
        logger.exception("Failed to write job status file")
    # Also persist to job registry DB if possible
    try:
        from services.model_registry import register_job
        register_job(job_id, status, details or {})
    except Exception:
        # DB persistence is best-effort; keep file-based fallback
        logger.debug('DB job registration not available')


def update_status(job_id: str, status: str, details: dict = None):
    """Public helper for training functions to report detailed progress.

    Writes the same payload as _write_status but is intended for use from
    inside long-running training functions.
    """
    _write_status(job_id, status, details)


def enqueue_training(func: Callable[[], Any], on_complete: Callable[[str, Any], None] = None) -> str:
    """Start `func()` in a background thread and return a job id.

    `func` is expected to perform training and return a result object
    (or raise). `on_complete(job_id, result)` will be called on success.
    """
    job_id = f"train_{uuid.uuid4().hex[:8]}"

    # If Celery is configured, submit the job to Celery instead of threading
    try:
        from infra.celery import celery_app
        if celery_app and getattr(celery_app, 'conf', None) and getattr(celery_app.conf, 'broker_url', None):
            # Register initial running status
            _write_status(job_id, 'queued')
            # Submit as generic task by serializing the function name is tricky; provide a simple mapping for common tasks
            # If func is run_churn_training_job, call the example celery task
            try:
                # Attempt to find a matching celery task
                if func.__name__ == 'run_churn_training_job':
                    res = celery_app.send_task('infra.celery.train_churn_task')
                    _write_status(job_id, 'running', {'celery_id': res.id})
                else:
                    # Fallback: run in thread if unknown function
                    raise RuntimeError('Unknown function for Celery submission')
            except Exception:
                logger.exception('Celery submission failed; falling back to thread')
                # Fallthrough to thread runner
            else:
                return job_id
    except Exception:
        # Celery not configured or import failed; continue with thread-runner
        pass

    def _worker():
        _write_status(job_id, 'running')
        try:
            # If the provided func accepts a job_id parameter, pass it so
            # the function can emit progress updates via update_status().
            import inspect
            try:
                sig = inspect.signature(func)
                if len(sig.parameters) >= 1:
                    result = func(job_id)
                else:
                    result = func()
            except Exception:
                # signature introspection failed — call without job_id
                result = func()

            # Consider a falsy result (False, None) as failure so the status
            # reflects the actual runtime outcome. If the function returns
            # truthy (e.g., True or object), mark success.
            if result:
                _write_status(job_id, 'success', {'result': str(type(result))})
            else:
                _write_status(job_id, 'failed', {'result': str(result)})
            if on_complete:
                try:
                    on_complete(job_id, result)
                except Exception:
                    logger.exception("on_complete callback failed")
        except Exception as e:
            logger.exception("Training job failed")
            _write_status(job_id, 'failed', {'error': str(e)})

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return job_id
