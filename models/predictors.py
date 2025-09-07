# models/predictors.py
"""
Core Predictive Model wrapper classes for:
1. DemandForecaster (Prophet)
2. ChurnPredictor (XGBoost)
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from models.evaluation import get_classification_metrics

# --- 1. DEMAND FORECASTING ---

class DemandForecaster:
    """Wrapper for Prophet model to handle training, prediction, and simulation."""
    
    def __init__(self, holidays_df=None):
        # Prophet initialization with seasonality and holiday configs
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            holidays=holidays_df # Pass marketing_campaigns.csv here
        )

    def fit(self, ts_df: pd.DataFrame):
        """Trains the prophet model on the provided timeseries data (ds, y)."""
        self.model.fit(ts_df)
        print("Prophet model fitting complete.")
        return self

    def predict(self, future_days: int = 90):
        """Generates a forecast for N future days with confidence intervals."""
        future_df = self.model.make_future_dataframe(periods=future_days)
        forecast = self.model.predict(future_df)
        return forecast

    def predict_simulation(self, future_days: int = 90, promo_uplift_pct: float = 0.0):
        """
        Runs a "what-if" simulation by applying an uplift percentage to the standard forecast.
        
        NOTE: A production-grade approach would add 'promo_uplift_pct' as an
        external regressor during training. For this rapid implementation, we apply
        it as a simple multiplier to the baseline forecast (yhat).
        """
        if promo_uplift_pct == 0:
            # If no simulation, just return the standard forecast
            return self.predict(future_days)

        # 1. Get the standard forecast
        base_forecast = self.predict(future_days)
        
        # 2. Create the simulated forecast
        sim_forecast = base_forecast.copy()
        multiplier = 1.0 + (promo_uplift_pct / 100.0)

        # Apply multiplier ONLY to future dates
        past_rows = sim_forecast['ds'] <= self.model.history_dates.max()
        future_rows = sim_forecast['ds'] > self.model.history_dates.max()

        sim_forecast.loc[future_rows, ['yhat', 'yhat_lower', 'yhat_upper']] *= multiplier
        
        # Label this forecast for the dashboard
        sim_forecast['forecast_type'] = 'Simulation'
        base_forecast['forecast_type'] = 'Baseline'

        # Combine them for plotting
        combined_forecast = pd.concat([base_forecast, sim_forecast])
        
        return combined_forecast


# --- 2. CHURN PREDICTION & LTV ESTIMATION ---

class ChurnPredictor:
    """Wrapper for XGBoost Churn model and LTV estimation logic."""
    
    def __init__(self):
        # Define model features. These MUST match models/features.py output
        self.numeric_features = ['Recency', 'Frequency', 'Monetary', 'Tenure']
        self.categorical_features = ['City', 'Segment'] # From customer_data.csv
        
        # Define the target variable
        self.target = 'Churned'

        # Create the scikit-learn preprocessing pipeline
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])

        # Define the XGBoost Classifier model
        # These parameters are starting points; tune them with GridSearchCV later.
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            use_label_encoder=False,
            random_state=42
        )

        # Full pipeline
        self.pipeline = Pipeline(steps=[('preprocessor', self.preprocessor),
                                        ('classifier', self.model)])
        self.feature_names = None
        self.shap_explainer = None


    def fit(self, features_df: pd.DataFrame):
        """Trains the preprocessing pipeline and the XGBoost classifier."""
        X = features_df[self.numeric_features + self.categorical_features]
        y = features_df[self.target]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Fit the full pipeline
        self.pipeline.fit(X_train, y_train)

        # --- Evaluate and Store Metrics ---
        y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
        y_pred_binary = self.pipeline.predict(X_test)
        
        metrics = get_classification_metrics(y_test, y_pred_proba, y_pred_binary)
        print(f"Churn Model Training Complete. Test AUC: {metrics['auc']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

        # --- Generate Feature Importance (SHAP) ---
        # Get processed feature names after one-hot encoding
        try:
            ohe_categories = self.pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(self.categorical_features)
            self.feature_names = self.numeric_features + list(ohe_categories)
        except:
             # Fallback for older sklearn versions
            self.feature_names = self.numeric_features + list(self.pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names(self.categorical_features))

        # Create SHAP explainer for key drivers
        X_train_transformed = self.pipeline.named_steps['preprocessor'].transform(X_train)
        self.shap_explainer = shap.TreeExplainer(self.pipeline.named_steps['classifier'], data=X_train_transformed)

        return self, metrics

    def predict_churn_probability(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts churn probability for new data and estimates LTV.
        Returns the original DataFrame with new prediction columns.
        """
        if not hasattr(self.pipeline, 'classes_'):
             raise RuntimeError("Model has not been fitted yet. Run fit() first.")
             
        X_features = features_df[self.numeric_features + self.categorical_features]
        
        # Predict probability of Churn (class 1)
        churn_proba = self.pipeline.predict_proba(X_features)[:, 1]
        
        results_df = features_df.copy()
        results_df['ChurnProbability'] = churn_proba
        
        # --- 4. Customer LTV Estimation (Goal 4) ---
        # This is a basic LTV model using the outputs we just generated:
        # LTV = Average Order Value (AOV) * Purchase Frequency * Customer Lifespan
        # We can simplify: Lifespan = 1 / Churn Rate. 
        # So, LTV ~ Total Historical Value / ChurnProbability
        # We add epsilon to avoid divide-by-zero for "perfect" customers.
        epsilon = 1e-6
        results_df['Estimated_LTV'] = results_df['Monetary'] / (results_df['ChurnProbability'] + epsilon)
        
        # Cap LTV for realism (e.g., at 5x the highest historical monetary value)
        max_ltv_cap = results_df[results_df['Monetary'] > 0]['Monetary'].max() * 5
        if pd.isna(max_ltv_cap):
             max_ltv_cap = 50000 # Default cap if no monetary value exists
             
        results_df['Estimated_LTV'] = results_df['Estimated_LTV'].clip(upper=max_ltv_cap)
        # For customers who already churned (proba > 99%), set LTV to their historical value
        results_df.loc[results_df['ChurnProbability'] > 0.99, 'Estimated_LTV'] = results_df['Monetary']
        
        return results_df.sort_values(by='ChurnProbability', ascending=False)


    def get_key_drivers_df(self) -> pd.DataFrame:
        """
        Returns a DataFrame of global SHAP values for the Feature Importance plot.
        """
        if self.shap_explainer is None:
            raise RuntimeError("SHAP Explainer not available. Run fit() first.")

        # Get the underlying transformed data used by the explainer
        X_transformed_shap = pd.DataFrame(self.shap_explainer.data, columns=self.feature_names)
        
        # Calculate SHAP values
        shap_values = self.shap_explainer.values_for_class(1) # Values for "Churned" class
        
        # Calculate mean absolute SHAP value per feature
        mean_abs_shap = pd.DataFrame(
            np.abs(shap_values).mean(axis=0),
            index=self.feature_names,
            columns=['FeatureImportance']
        ).reset_index()
        
        mean_abs_shap = mean_abs_shap.rename(columns={'index': 'Feature'})
        mean_abs_shap = mean_abs_shap.sort_values(by='FeatureImportance', ascending=False)
        
        return mean_abs_shap