# etl/schedules.py
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Automated Data Pipeline Scheduler - V22.0 (Prediction Saving Fix)
#
# BUG FIX: The churn training pipeline was only saving the model artifact, not
#          the predictions. Added the crucial step to use the newly trained
#          model to generate predictions (including Estimated_LTV) and save
#          them to the 'customer_churn_predictions' database table.
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
        
        sales_df = DATA.get('sales', pd.DataFrame())
        # FIX: Ensure 'marketing_campaigns' exists before proceeding
        marketing_df = DATA.get('marketing_campaigns', pd.DataFrame())
        if not marketing_df.empty:
            marketing_df = marketing_df.rename(columns={'startdate': 'ds', 'campaignname': 'holiday'})
        else:
            marketing_df = None # Prophet can handle a None holidays_df
        
        # 1. Feature Engineering: Create the main timeseries
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
        engine = get_engine()
        from etl.transforms import DATA # Import fresh data
        
        sales_df = DATA.get('sales', pd.DataFrame())
        customer_df = DATA.get('customers', pd.DataFrame())
        analysis_date = pd.to_datetime(datetime.now())

        # 1. Feature Engineering (RFM)
        rfm_features = build_rfm_features(sales_df, customer_df, analysis_date)

        # 2. Training
        churn_model = ChurnPredictor()
        fitted_pipeline, metrics = churn_model.fit(rfm_features)
        
        # 3. Save Artifacts
        save_model_artifact(churn_model, 'churn_predictor_main.joblib')
        save_model_artifact(metrics, 'churn_metrics.joblib')
        
        logger.info(f"[ML JOB]: Churn Model training complete (Test AUC: {metrics.get('auc')})...")

        # --- BUG FIX: GENERATE AND SAVE PREDICTIONS TO DATABASE ---
        logger.info("[ML JOB]: Generating predictions for all customers...")
        # Use the newly trained model to get predictions, including Estimated_LTV
        full_predictions_df = churn_model.predict_churn_probability(rfm_features)
        # Standardize column names for persistence
        preds_to_save = full_predictions_df.copy()
        if 'ChurnProbability' in preds_to_save.columns and 'churn_probability' not in preds_to_save.columns:
            preds_to_save['churn_probability'] = preds_to_save['ChurnProbability']
        if 'Estimated_LTV' not in preds_to_save.columns and 'estimated_ltv' in preds_to_save.columns:
            preds_to_save.rename(columns={'estimated_ltv': 'Estimated_LTV'}, inplace=True)
        # Select only necessary columns
        cols_to_save = ['customerid', 'churn_probability', 'Estimated_LTV']
        predictions_to_save = preds_to_save[[col for col in cols_to_save if col in preds_to_save.columns]]

        logger.info(f"[ML JOB]: Saving {len(predictions_to_save)} predictions to 'customer_churn_predictions' table...")
        predictions_to_save.to_sql(
            "customer_churn_predictions",
            engine,
            if_exists='replace',
            index=False
        )
        logger.info("[ML JOB]: Successfully saved predictions to database.")
        # --- END BUG FIX ---

    except Exception as e:
        logger.error(f"[ML JOB]: Churn Model training failed: {e}", exc_info=True)


# --- SCHEDULER REGISTRATION ---
def start_scheduler():
    """
    Configures and starts the APScheduler jobs.
    """
    if scheduler.running:
        logger.warning("Scheduler already running.")
        return

    logger.info("Starting background scheduler...")
    
    # 1. Add the main ETL job. Runs daily at 2:00 AM.
    scheduler.add_job(run_daily_etl, 'cron', hour=2, minute=0, id='job_daily_etl')

    # 2. Add the ML jobs. They will run 10 minutes after the ETL job.
    scheduler.add_job(run_forecast_training_pipeline, 'cron', hour=2, minute=10, id='job_forecast_train')
    scheduler.add_job(run_churn_training_pipeline, 'cron', hour=2, minute=15, id='job_churn_train')
    
    scheduler.start()
    logger.info("Scheduler started. Daily ETL and ML jobs are scheduled.")

    # --- FOR TESTING: Uncomment to run jobs on startup ---
    # logger.info("RUNNING BOOTSTRAP JOBS NOW FOR TESTING...")
    # run_daily_etl()
    # run_forecast_training_pipeline()
    # run_churn_training_pipeline()
    # logger.info("BOOTSTRAP JOBS COMPLETE.")

