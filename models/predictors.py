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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
import logging

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

        # Apply multiplier ONLY to future dates. Prophet stores training history in self.model.history (DataFrame)
        try:
            history_max = self.model.history['ds'].max()
        except Exception:
            history_max = None
        if history_max is not None:
            past_rows = sim_forecast['ds'] <= history_max
            future_rows = sim_forecast['ds'] > history_max
        else:
            past_rows = sim_forecast['ds'] <= sim_forecast['ds'].min()
            future_rows = sim_forecast['ds'] > sim_forecast['ds'].min()

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
        # Best-effort SHAP artifacts
        self.shap_explainer = None
        # Persisted mean |SHAP| summary for stable UI plots without recomputation
        self.shap_summary_ = None


    def fit(self, features_df: pd.DataFrame):
        """Trains the preprocessing pipeline and the XGBoost classifier.

        This method is tolerant to different column casings produced by ETL and
        feature builders. It will map lowercase/underscored names to the
        canonical names expected by the pipeline.
        """
        # Normalize input column names to a predictable form and map to expected casing
        cols_map = {c.lower(): c for c in features_df.columns}

        def _map_col(name_variants):
            for v in name_variants:
                if v.lower() in cols_map:
                    return cols_map[v.lower()]
            return None

        # Build lists of existing column names matching expected features
        numeric_cols = []
        for nf in self.numeric_features:
            mapped = _map_col([nf, nf.lower()])
            if mapped is None:
                raise KeyError(f"Required numeric feature '{nf}' not found in features_df columns: {list(features_df.columns)}")
            numeric_cols.append(mapped)

        categorical_cols = []
        for cf in self.categorical_features:
            mapped = _map_col([cf, cf.lower()])
            if mapped is None:
                # If a categorical column is missing, create a placeholder column with NaNs so OneHotEncoder can handle it
                features_df[cf] = pd.NA
                categorical_cols.append(cf)
            else:
                categorical_cols.append(mapped)

        target_col = _map_col([self.target, self.target.lower()])
        if target_col is None:
            raise KeyError(f"Target column '{self.target}' not found in features_df columns")

        X = features_df[numeric_cols + categorical_cols]
        y = features_df[target_col]

        # --- Train/validation split ---
        # Prefer stratified split on the target to preserve class balance when possible.
        stratify_arg = None
        try:
            # if y has at least 2 classes and enough samples per class, stratify
            class_counts = y.value_counts()
            if class_counts.min() >= 2 and class_counts.shape[0] >= 2:
                stratify_arg = y
        except Exception:
            stratify_arg = None

        # If data has a datetime-like index or 'joindate'/'timestamp', a time-aware split is preferable.
        # For now, fall back to stratified random split.
        if stratify_arg is not None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_arg)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Basic class-balance check
        try:
            train_counts = y_train.value_counts().to_dict()
            test_counts = y_test.value_counts().to_dict()
        except Exception:
            train_counts, test_counts = {}, {}

        # --- Fit preprocessing separately so we can train classifier with eval_set and early stopping ---
        # Fit preprocessor on training data
        self.preprocessor.fit(X_train)
        X_train_trans = self.preprocessor.transform(X_train)
        X_test_trans = self.preprocessor.transform(X_test)

        # Fit classifier with early stopping using validation set
        clf = self.model
        try:
            clf.fit(X_train_trans, y_train, eval_set=[(X_test_trans, y_test)], early_stopping_rounds=10, verbose=False)
        except TypeError:
            # older sklearn/xgboost wrappers may not accept early stopping via fit kwargs through sklearn API
            clf.fit(X_train_trans, y_train)

        # Optionally calibrate probabilities to improve probability quality
        calibrated = None
        try:
            calibrator = CalibratedClassifierCV(base_estimator=clf, cv='prefit', method='isotonic')
            calibrator.fit(X_test_trans, y_test)
            calibrated = calibrator
            final_clf = calibrated
        except Exception:
            # fall back to uncalibrated classifier
            final_clf = clf

        # Assemble the final pipeline with preprocessor and the trained (optionally calibrated) classifier
        self.pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('classifier', final_clf)])

        # --- Evaluate and Store Metrics ---
        try:
            y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
            y_pred_binary = self.pipeline.predict(X_test)
        except Exception:
            # if predict_proba on pipeline fails (rare), fall back to classifier directly
            try:
                y_pred_proba = final_clf.predict_proba(X_test_trans)[:, 1]
                y_pred_binary = final_clf.predict(X_test_trans)
            except Exception:
                y_pred_proba = None
                y_pred_binary = None

        metrics = get_classification_metrics(y_test, y_pred_proba, y_pred_binary)
        try:
            if y_pred_proba is not None:
                metrics['brier'] = float(brier_score_loss(y_test, y_pred_proba))
        except Exception:
            pass

        print(f"Churn Model Training Complete. Test AUC: {metrics.get('auc', 0):.4f}, Accuracy: {metrics.get('accuracy', 0):.4f}")

        # --- Generate Feature Importance (SHAP) ---
        try:
            # Extract fitted OneHotEncoder to compute output feature names
            ohe = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
            if hasattr(ohe, 'get_feature_names_out'):
                ohe_categories = ohe.get_feature_names_out(self.categorical_features)
            else:
                ohe_categories = ohe.get_feature_names(self.categorical_features)
            self.feature_names = self.numeric_features + list(ohe_categories)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not extract OHE feature names reliably: {e}")
            self.feature_names = self.numeric_features + self.categorical_features

        # Create SHAP explainer and a compact summary for key drivers (best-effort)
        try:
            # prefer the underlying xgboost booster where possible
            clf_for_shap = None
            if hasattr(final_clf, 'base_estimator'):
                clf_for_shap = getattr(final_clf, 'base_estimator')
            elif hasattr(final_clf, 'estimator'):
                clf_for_shap = getattr(final_clf, 'estimator')
            else:
                clf_for_shap = final_clf

            X_train_transformed = self.preprocessor.transform(X_train)
            self.shap_explainer = shap.TreeExplainer(clf_for_shap, data=X_train_transformed)

            # Compute mean |SHAP| summary on a limited background sample to bound size
            try:
                # Sample up to 2000 rows to keep artifact smaller
                import numpy as _np
                bg = X_train_transformed
                try:
                    if hasattr(bg, 'toarray'):
                        # ColumnTransformer may yield sparse matrices
                        bg = bg.toarray()
                except Exception:
                    pass
                if getattr(bg, 'shape', (0,))[0] > 2000:
                    idx = _np.random.RandomState(42).choice(bg.shape[0], size=2000, replace=False)
                    bg_sample = bg[idx]
                else:
                    bg_sample = bg

                # SHAP API differences across versions: prefer call() then fallback to .shap_values
                try:
                    exp = self.shap_explainer(bg_sample)
                    shap_vals = getattr(exp, 'values', exp)
                except Exception:
                    shap_vals = self.shap_explainer.shap_values(bg_sample)

                # Binary models sometimes return a list [neg_class, pos_class]
                if isinstance(shap_vals, list) and len(shap_vals) > 1:
                    shap_vals = shap_vals[1]

                feature_names = self.feature_names or [f'f{i}' for i in range(shap_vals.shape[1])]
                mean_abs = _np.abs(shap_vals).mean(axis=0)
                self.shap_summary_ = pd.DataFrame({
                    'Feature': feature_names,
                    'FeatureImportance': mean_abs
                }).sort_values('FeatureImportance', ascending=False).reset_index(drop=True)
            except Exception:
                # If summary computation fails, keep explainer only
                logging.getLogger(__name__).warning('Failed to compute SHAP summary; will rely on explainer only', exc_info=True)
        except Exception:
            logging.getLogger(__name__).exception('Failed to create SHAP explainer')

        # Add training metadata to metrics for later persistence
        metrics['_train_counts'] = train_counts
        metrics['_test_counts'] = test_counts

        return self, metrics

    def predict_churn_probability(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts churn probability for new data and estimates LTV.
        Returns the original DataFrame with new prediction columns.
        """
        if not hasattr(self.pipeline, 'classes_'):
             raise RuntimeError("Model has not been fitted yet. Run fit() first.")
             
        # Map incoming features similarly to fit()
        cols_map = {c.lower(): c for c in features_df.columns}

        def _map_col(name_variants):
            for v in name_variants:
                if v.lower() in cols_map:
                    return cols_map[v.lower()]
            return None

        numeric_cols = [_map_col([nf, nf.lower()]) for nf in self.numeric_features]
        categorical_cols = []
        for cf in self.categorical_features:
            mapped = _map_col([cf, cf.lower()])
            if mapped is None:
                # create placeholder column if missing
                features_df[cf] = pd.NA
                categorical_cols.append(cf)
            else:
                categorical_cols.append(mapped)

        X_features = features_df[numeric_cols + categorical_cols]
        
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
        # First, prefer a precomputed summary if available
        if getattr(self, 'shap_summary_', None) is not None and not getattr(self.shap_summary_, 'empty', True):
            return self.shap_summary_.copy()

        if self.shap_explainer is None:
            raise RuntimeError("SHAP Explainer not available. Run fit() first.")

        # Some SHAP explainer implementations expose different attributes across versions.
        try:
            shap_data = getattr(self.shap_explainer, 'data', None)
            shap_values = None
            # Try computing on stored background data if present; otherwise attempt direct attributes
            try:
                if shap_data is not None:
                    try:
                        if hasattr(shap_data, 'toarray'):
                            shap_bg = shap_data.toarray()
                        else:
                            shap_bg = shap_data
                    except Exception:
                        shap_bg = shap_data
                    exp = self.shap_explainer(shap_bg)
                    shap_values = getattr(exp, 'values', exp)
            except Exception:
                pass
            if shap_values is None:
                # shap.Kernel or Tree explainers may expose `values` or `values_for_class` depending on model
                if hasattr(self.shap_explainer, 'values_for_class'):
                    shap_values = self.shap_explainer.values_for_class(1)
                elif hasattr(self.shap_explainer, 'values'):
                    shap_values = self.shap_explainer.values

            if shap_values is None:
                raise RuntimeError('Unable to extract shap values from explainer')

            # Ensure feature_names length matches shap_values second-dimension
            feature_names = self.feature_names or [f'f{i}' for i in range(shap_values.shape[1])]

            mean_abs_shap = pd.DataFrame(
                np.abs(shap_values).mean(axis=0),
                index=feature_names,
                columns=['FeatureImportance']
            ).reset_index()

            mean_abs_shap = mean_abs_shap.rename(columns={'index': 'Feature'})
            mean_abs_shap = mean_abs_shap.sort_values(by='FeatureImportance', ascending=False)
            return mean_abs_shap
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception(f"Failed to compute SHAP summary: {e}")
            # Return an empty DataFrame with expected columns to keep UI stable
            return pd.DataFrame(columns=['Feature', 'FeatureImportance'])