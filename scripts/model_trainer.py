# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Customer Churn Model Trainer - V21.0
#
# This script reads all customer/sales data, engineers features, trains
# an XGBoost model, saves the model artifact (PKL file), AND
# saves all customer predictions to the 'customer_churn_predictions' table.
#
# This logic is from pharma_dashboard_backup/model_trainer.py
# -----------------------------------------------------------------------------

import sys
import os
import logging

# --- ADD THIS 5-LINE BLOCK ---
# Add the project root directory (one level up from 'scripts') to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- END BLOCK ---

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from datetime import datetime

# --- Updated Imports for New Structure ---
from services.db import get_engine, load_data_safely
from config.settings import settings # This imports your settings instance

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIG ---
# Define the model store path relative to the project root
MODEL_STORE_PATH = os.path.join(project_root, "model_store")
if not os.path.exists(MODEL_STORE_PATH):
    os.makedirs(MODEL_STORE_PATH)

# --- 1. FEATURE ENGINEERING ---
def engineer_features(sales_df: pd.DataFrame, customers_df: pd.DataFrame) -> pd.DataFrame:
    """Creates the RFM and behavioral features needed for the model."""
    sales_df['timestamp'] = pd.to_datetime(sales_df['timestamp'])
    
    # Calculate RFM metrics
    current_date = datetime.now()
    rfm_df = sales_df.groupby('customerid').agg(
        last_purchase_date=('timestamp', 'max'),
        frequency=('orderid', 'nunique'),
        monetary=('netsale', 'sum')
    ).reset_index()
    
    rfm_df['recency'] = (current_date - rfm_df['last_purchase_date']).dt.days
    
    # Calculate additional behavioral features
    behavior_df = sales_df.groupby('customerid').agg(
        avg_basket_value=('netsale', 'mean'),
        unique_categories=('category', 'nunique'),
        total_quantity=('quantity', 'sum'),
        avg_discount_value=('discountvalue', 'mean')
    ).reset_index()

    # Join customer data, RFM, and behavior data
    features_df = pd.merge(customers_df, rfm_df, on='customerid', how='left')
    features_df = pd.merge(features_df, behavior_df, on='customerid', how='left')
    
    # Define the churn target variable (churn = 1 if recency > 180 days)
    features_df['churned'] = (features_df['recency'] > 180).astype(int)
    
    # Handle missing values for customers who never purchased (fill with 0s)
    features_df['recency'] = features_df['recency'].fillna(999) # Assign high recency to non-purchasers
    features_df = features_df.fillna(0) # Fill all other NA values with 0
    
    return features_df

# --- 2. MODEL TRAINING ---
def train_model(features_df: pd.DataFrame):
    """Trains, evaluates, and saves the churn model and scaler."""
    # Define features (X) and target (y)
    # One-hot encode categorical features
    categorical_cols = ['city', 'segment']
    X_categorical = pd.get_dummies(features_df[categorical_cols], drop_first=True)
    
    numerical_cols = ['frequency', 'monetary', 'recency', 'avg_basket_value', 'unique_categories', 'total_quantity', 'avg_discount_value']
    X_numerical = features_df[numerical_cols]
    
    X = pd.concat([X_numerical, X_categorical], axis=1)
    y = features_df['churned']
    
    # Save the columns to ensure prediction consistency
    joblib.dump(X.columns.tolist(), os.path.join(MODEL_STORE_PATH, 'model_columns.pkl'))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale numerical data
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # Train XGBoost Classifier
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    
    logger.info("\n--- Model Evaluation ---")
    logger.info(f"Accuracy on test set: {accuracy*100:.2f}%")
    logger.info(f"Classification Report:\n{report}")
    logger.info(f"Confusion Matrix:\n{matrix}")
    logger.info("------------------------\n")
    
    # Save the model and the scaler
    joblib.dump(model, os.path.join(MODEL_STORE_PATH, "churn_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_STORE_PATH, "scaler.pkl"))
    logger.info(f"Model and scaler have been saved to '{MODEL_STORE_PATH}'.")
    
    # Return trained objects for full dataset prediction
    return model, scaler, X.columns, numerical_cols

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    engine = get_engine()
    
    logger.info("Loading data for feature engineering...")
    sales_df = load_data_safely("sales", engine)
    customers_df = load_data_safely("customers", engine)

    if sales_df.empty or customers_df.empty:
        logger.error("Could not load sales or customer data. Aborting training.")
        sys.exit(1)

    logger.info("Engineering features from raw data...")
    full_feature_dataset = engineer_features(sales_df, customers_df)
    logger.info(f"Feature engineering complete. Dataset has {len(full_feature_dataset)} customers.")

    logger.info("Splitting data and training churn prediction model...")
    model, scaler, model_cols, num_cols = train_model(full_feature_dataset)
    
    # --- THIS IS THE CRITICAL MISSING BLOCK ---
    logger.info("Generating predictions for ALL customers and saving to database...")

    # Prepare the FULL dataset for prediction (must match training columns)
    X_full_categorical = pd.get_dummies(full_feature_dataset[ ['city', 'segment'] ], drop_first=True)
    X_full_numerical = full_feature_dataset[num_cols]
    X_full = pd.concat([X_full_numerical, X_full_categorical], axis=1)
    
    # Add any missing columns that were in training (if data drift occurred)
    for col in model_cols:
        if col not in X_full.columns:
            X_full[col] = 0
    X_full = X_full[model_cols] # Ensure exact column order

    # Scale the full numerical dataset
    X_full[num_cols] = scaler.transform(X_full[num_cols])
    
    # Use the trained model to predict probabilities on the full dataset
    full_feature_dataset['churn_probability'] = model.predict_proba(X_full)[:, 1]
    
    # Create the final predictions DataFrame
    predictions_df = full_feature_dataset[['customerid', 'churn_probability']]
    
    # Save predictions to the database
    try:
        predictions_df.to_sql(
            "customer_churn_predictions", # This is the table the app is looking for
            engine,
            if_exists='replace',
            index=False
        )
        logger.info(f"Successfully saved {len(predictions_df)} predictions to 'customer_churn_predictions' table.")
    except Exception as e:
        logger.error(f"Failed to save predictions to database: {e}", exc_info=True)