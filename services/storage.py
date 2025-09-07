# services/storage.py
"""
Service module for serializing (saving) and deserializing (loading) 
trained ML model artifacts (pipelines, models, scalers, etc.)
"""
import os
import joblib
import logging
from config.settings import MODEL_STORE_PATH

logger = logging.getLogger(__name__)

# Ensure the model storage directory exists
os.makedirs(MODEL_STORE_PATH, exist_ok=True)


def save_model_artifact(artifact: object, file_name: str):
    """
    Saves a model artifact (e.g., model, pipeline, explainer) to the model store.
    
    Example file_names: 
    - 'demand_forecaster_all.joblib'
    - 'churn_predictor_pipeline.joblib'
    - 'churn_shap_explainer.joblib'
    """
    file_path = os.path.join(MODEL_STORE_PATH, file_name)
    try:
        joblib.dump(artifact, file_path)
        logger.info(f"Successfully saved artifact to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save artifact {file_name}. Error: {e}")

def load_model_artifact(file_name: str) -> object:
    """
    Loads a model artifact from the model store.
    Returns the artifact, or None if it doesn't exist.
    """
    file_path = os.path.join(MODEL_STORE_PATH, file_name)
    if not os.path.exists(file_path):
        logger.warning(f"Artifact file not found: {file_path}. Model must be trained first.")
        return None
        
    try:
        artifact = joblib.load(file_path)
        logger.info(f"Successfully loaded artifact from {file_path}")
        return artifact
    except Exception as e:
        logger.error(f"Failed to load artifact {file_name}. Error: {e}")
        return None