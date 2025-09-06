# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Customer Churn Prediction Model Trainer - V21.0 (Final Master)
#
# This script is a one-time "factory" that:
# 1. Loads historical customer and sales data from the database.
# 2. Engineers features relevant to churn (e.g., RFM, tenure).
# 3. Trains a classification model to predict churn.
# 4. Saves the trained model and the data scaler to files for later use.
# -----------------------------------------------------------------------------

import logging
import pandas as pd
from datetime import datetime, timedelta
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import from our central modules
from database import get_engine, load_data_safely

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. FEATURE ENGINEERING ---
def create_features(engine):
    """Loads data and creates features for model training."""
    logger.info("Loading data for feature engineering...")
    sales_df = load_data_safely('sales', engine)
    customer_df = load_data_safely('customers', engine)

    if sales_df.empty or customer_df.empty:
        logger.error("Sales or Customer data is missing. Cannot create features.")
        return None

    logger.info("Engineering features from raw data...")
    sales_df['timestamp'] = pd.to_datetime(sales_df['timestamp'])
    customer_df['joindate'] = pd.to_datetime(customer_df['joindate'])

    # Use the most recent transaction date as our snapshot for calculating recency
    snapshot_date = sales_df['timestamp'].max() + timedelta(days=1)
    
    rfm = sales_df.groupby('customerid').agg(
        Recency=('timestamp', lambda date: (snapshot_date - date.max()).days),
        Frequency=('orderid', 'nunique'),
        MonetaryValue=('netsale', 'sum')
    ).reset_index()

    customer_features = sales_df.groupby('customerid').agg(
        avg_basket_value=('netsale', 'mean'),
        total_quantity=('quantity', 'sum')
    ).reset_index()

    features_df = rfm.merge(customer_features, on='customerid')
    features_df = features_df.merge(customer_df[['customerid', 'joindate', 'segment']], on='customerid')
    
    features_df['tenure_days'] = (snapshot_date - features_df['joindate']).dt.days

    # --- Define the Churn Target Variable ---
    # A customer is considered "churned" if their last purchase was > 180 days ago.
    features_df['Churn'] = (features_df['Recency'] > 180).astype(int)
    
    # Prepare final feature set for the model
    features_df = features_df.drop(['customerid', 'joindate'], axis=1)
    features_df = pd.get_dummies(features_df, columns=['segment'], drop_first=True)

    logger.info(f"Feature engineering complete. Dataset has {len(features_df)} customers.")
    return features_df

# --- 2. MODEL TRAINING ---
def train_model(df):
    """Trains a logistic regression model and saves it to a file."""
    if df is None or df.empty:
        logger.error("Feature DataFrame is empty. Aborting training.")
        return

    logger.info("Splitting data and training churn prediction model...")
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Ensure all required columns are present after dummy creation
    required_cols = ['Recency', 'Frequency', 'MonetaryValue', 'avg_basket_value', 'total_quantity', 'tenure_days', 'segment_Silver', 'segment_Gold']
    for col in required_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[required_cols]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use class_weight='balanced' to handle imbalanced churn data
    model = LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear')
    model.fit(X_train_scaled, y_train)

    # --- Evaluate the model's performance ---
    y_pred = model.predict(X_test_scaled)
    logger.info("\n--- Model Evaluation ---")
    logger.info(f"Accuracy on test set: {accuracy_score(y_test, y_pred):.2%}")
    logger.info("Classification Report:\n" + classification_report(y_test, y_pred))
    logger.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))
    logger.info("------------------------\n")

    # --- Save the final model and scaler to disk ---
    joblib.dump(model, 'churn_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    logger.info("Model and scaler have been saved to 'churn_model.pkl' and 'scaler.pkl'.")

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        engine = get_engine()
        features = create_features(engine)
        train_model(features)
    except Exception as e:
        logger.critical(f"An error occurred during the model training pipeline: {e}", exc_info=True)

