# train_models.py
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Example: train demand forecaster
sales = pd.read_csv("data/sales_history.csv")  # Your historical sales
X = sales[["units_sold", "price"]]  # Features (simplified)
y = sales["future_demand"]           # Target you want to predict

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = xgb.XGBRegressor()
model.fit(X_train, y_train)

joblib.dump(model, "model_store/demand_forecaster.pkl")
print("✅ Demand forecaster saved!")

# Example: churn predictor
customers = pd.read_csv("data/customers_history.csv")
Xc = customers[["visits", "orders", "returns"]]
yc = customers["churned"]

churn_model = xgb.XGBClassifier()
churn_model.fit(Xc, yc)

joblib.dump(churn_model, "model_store/churn_predictor.pkl")
print("✅ Churn predictor saved!")