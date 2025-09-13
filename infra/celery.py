from celery import Celery
import os

# Simple Celery app for background tasks. Configure BROKER_URL and RESULT_BACKEND
# via environment variables for production use. Example: export BROKER_URL=redis://localhost:6379/0
BROKER_URL = os.environ.get('BROKER_URL', '')
RESULT_BACKEND = os.environ.get('RESULT_BACKEND', '')

if not BROKER_URL:
    # Fallback: do not raise here; tasks will run only if configured
    BROKER_URL = None

celery_app = Celery('pharma_tasks')
if BROKER_URL:
    celery_app.conf.broker_url = BROKER_URL
if RESULT_BACKEND:
    celery_app.conf.result_backend = RESULT_BACKEND

@celery_app.task(bind=True)
def train_churn_task(self):
    """Example Celery task that wraps the existing run_churn_training_job helper."""
    from app.utils.analytics_helpers import run_churn_training_job
    result = run_churn_training_job()
    return {'result': bool(result)}
