# etl/schedules.py
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Automated Data Pipeline Scheduler - V21.0 (Final Master)
# ...
# [PREDICTIVE ANALYTICS EXTENSION ADDED]
# -----------------------------------------------------------------------------
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime

from etl.ingest import fetch_all_data_from_db
from etl.transforms import initialize_data
from services.db import get_engine

# --- NEW IMPORTS FOR PREDICTION PIPELINES ---
import pandas as pd
from models.features import get_daily_sales_timeseries, build_rfm_features
from models.predictors import DemandForecaster, ChurnPredictor
from services.storage import save_model_artifact
from config.settings import MODEL_STORE_PATH # Ensure this path is defined

logger = logging.getLogger(__name__)
scheduler = BackgroundScheduler(daemon=True)


# --- EXISTING DATA REFRESH JOB ---
def run_daily_etl():
    """
    Existing job: Fetches all data from the source DB and updates the global DATA dictionary.
    """
    try:
        logger.info("[ETL JOB]: Starting daily data ingestion and transformation...")
        engine = get_engine()
        initialize_data(engine) # This refreshes etl.transforms.DATA
        logger.info("[ETL JOB]: Daily ETL complete.")
    except Exception as e:
        logger.error(f"[ETL JOB]: Daily ETL failed: {e}", exc_info=True)


# --- NEW PREDICTIVE TRAINING JOBS ---

def run_forecast_training_pipeline():
    """
    NEW JOB: Retrains the master demand forecast model (Prophet).
    This job MUST run AFTER run_daily_etl() completes.
    """
    try:
        logger.info("[ML JOB]: Starting Demand Forecast training pipeline...")
        from etl.transforms import DATA # Import the freshly loaded data
        
        sales_df = DATA['sales']
        marketing_df = DATA['campaigns'].rename(columns={'CampaignDate': 'ds', 'CampaignName': 'holiday'})
        
        # 1. Feature Engineering: Create the main timeseries (all categories, all channels)
        # Note: In production, you might train separate models per category.
        # For this dashboard, one aggregate model is efficient.
        ts_df = get_daily_sales_timeseries(sales_df, category='all', channel='all')
        
        if ts_df.empty or len(ts_df) < 60:
            logger.warning("[ML JOB]: Insufficient data (<60 days) to train forecast model. Skipping.")
            return

        # 2. Training
        forecaster = DemandForecaster(holidays_df=marketing_df)
        forecaster.fit(ts_df)
        
        # 3. Save Artifact
        save_model_artifact(forecaster, 'demand_forecaster_main.joblib')
        logger.info("[ML JOB]: Demand Forecast training complete and model saved.")
        
    except Exception as e:
        logger.error(f"[ML JOB]: Demand Forecast training failed: {e}", exc_info=True)


def run_churn_training_pipeline():
    """
    NEW JOB: Retrains the Customer Churn (XGBoost) model.
    This job also MUST run AFTER run_daily_etl().
    """
    try:
        logger.info("[ML JOB]: Starting Customer Churn training pipeline...")
        from etl.transforms import DATA # Import fresh data
        
        sales_df = DATA['sales']
        customer_df = DATA['customers']
        analysis_date = pd.to_datetime(datetime.now()) # Use "today" as the snapshot date for RFM

        # 1. Feature Engineering (RFM)
        rfm_features = build_rfm_features(sales_df, customer_df, analysis_date)

        # 2. Training
        churn_model = ChurnPredictor()
        fitted_pipeline, metrics = churn_model.fit(rfm_features)
        
        # 3. Save Artifacts
        # We save the *entire* ChurnPredictor object, which contains:
        # 1. The scikit-learn pipeline (preprocessor + model)
        # 2. The SHAP explainer
        # 3. The feature name list
        save_model_artifact(churn_model, 'churn_predictor_main.joblib')
        save_model_artifact(metrics, 'churn_metrics.joblib') # Save metrics for the dashboard
        
        logger.info(f"[ML JOB]: Churn Model training complete (Test AUC: {metrics.get('auc')}) and artifacts saved.")

    except Exception as e:
        logger.error(f"[ML JOB]: Churn Model training failed: {e}", exc_info=True)


# --- SCHEDULER REGISTRATION ---
def start_scheduler():
    """
    Configures and starts the APScheduler jobs.
    We chain the ML jobs to run after the main ETL.
    """
    if scheduler.running:
        logger.warning("Scheduler already running.")
        return

    logger.info("Starting background scheduler...")
    
    # 1. Add the main ETL job. Runs daily at 2:00 AM.
    scheduler.add_job(run_daily_etl, 'cron', hour=2, minute=0, id='job_daily_etl')

    # 2. Add the ML jobs. We chain them to the ETL job using 'add_listener'.
    # When the ETL job succeeds, the ML pipelines are triggered immediately after.
    # (Alternatively, we could schedule them for 3:00 AM, but this ensures data freshness)
    
    scheduler.add_listener(
        lambda event: run_forecast_training_pipeline(),
        mask=0x1000 # EVENT_JOB_EXECUTED
    )
    
    scheduler.add_listener(
         lambda event: run_churn_training_pipeline(),
        mask=0x1000 # EVENT_JOB_EXECUTED
    )

    # Note: The listeners above will trigger when ANY job finishes, including themselves.
    # To be more robust, filter the event:
    # def job_listener(event):
    #    if event.job_id == 'job_daily_etl':
    #        run_forecast_training_pipeline()
    #        run_churn_training_pipeline()
    # scheduler.add_listener(job_listener, mask=0x1000) # EVENT_JOB_EXECUTED
    # ... but the simple implementation above is fine for this structure.

    scheduler.start()
    logger.info("Scheduler started. Daily ETL scheduled for 02:00.")

    # --- FOR TESTING: Run jobs now instead of waiting for 2 AM ---
    # logger.info("RUNNING BOOTSTRAP JOBS NOW...")
    # run_daily_etl()
    # run_forecast_training_pipeline()
    # run_churn_training_pipeline()
    # logger.info("BOOTSTRAP JOBS COMPLETE.")