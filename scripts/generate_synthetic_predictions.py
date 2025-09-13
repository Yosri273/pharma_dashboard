"""Generate synthetic customer churn predictions CSV for local testing.
Generates `model_store/synthetic_customer_churn_predictions.csv` using real
customer IDs from `sales_data.csv` when available, otherwise creates a set
of synthetic IDs.
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SALES_CSV = os.path.join(BASE, 'sales_data.csv')
OUT_DIR = os.path.join(BASE, 'model_store')
OUT_CSV = os.path.join(OUT_DIR, 'synthetic_customer_churn_predictions.csv')

os.makedirs(OUT_DIR, exist_ok=True)

# Try to pull real customer IDs from sales_data.csv
if os.path.exists(SALES_CSV):
    try:
        sales = pd.read_csv(SALES_CSV)
        if 'CustomerID' in sales.columns:
            custs = sales['CustomerID'].dropna().unique().tolist()
        elif 'CustomerId' in sales.columns:
            custs = sales['CustomerId'].dropna().unique().tolist()
        else:
            custs = []
    except Exception:
        custs = []
else:
    custs = []

# If not enough real IDs, generate synthetic ones
n = max(500, len(custs))
if not custs:
    custs = [f'C{1000 + i}' for i in range(n)]
else:
    # ensure at least 500 ids
    if len(custs) < n:
        custs = list(custs) + [f'C_SYN_{i}' for i in range(n - len(custs))]

np.random.seed(42)
churn_probs = np.round(np.random.beta(2, 5, size=len(custs)), 4)
ltv = np.round(np.random.lognormal(mean=5, sigma=0.8, size=len(custs)), 2)

df = pd.DataFrame({
    'customerid': custs,
    'churn_probability': churn_probs,
    'Estimated_LTV': ltv
})

df.to_csv(OUT_CSV, index=False)
print(f'Wrote {len(df)} synthetic predictions to: {OUT_CSV}')
