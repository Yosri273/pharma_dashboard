# models/evaluation.py
"""
Model evaluation metrics (RMSE, MAE, AUC).
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, accuracy_score

def get_regression_metrics(y_true, y_pred):
    """Returns RMSE and MAE."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"rmse": rmse, "mae": mae}

def get_classification_metrics(y_true, y_pred_proba, y_pred_binary):
    """Returns AUC and Accuracy."""
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        auc = 0.5  # Handle cases with only one class present in a fold
    acc = accuracy_score(y_true, y_pred_binary)
    return {"auc": auc, "accuracy": acc}