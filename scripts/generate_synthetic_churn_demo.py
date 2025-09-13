"""Generate a more realistic synthetic dataset for churn training demo.
This script creates customers and sales with richer signals (support tickets,
returns, recency patterns) and writes `sales_demo.csv` and `customers_demo.csv`
under the workspace root to be consumed by the ETL or loaded directly.
"""
import pandas as pd
import random
from datetime import datetime, timedelta

NUM_CUSTOMERS = 2000
NUM_SALES = 20000

random.seed(42)

customers = []
start_date = datetime(2020,1,1)
for i in range(NUM_CUSTOMERS):
    join_date = start_date + timedelta(days=random.randint(0, 900))
    city = random.choice(['Riyadh','Jeddah','Dammam','Mecca','Medina'])
    segment = random.choices(['Retail','VIP','Corporate'], weights=[0.8,0.1,0.1])[0]
    support_tickets = random.choices([0,1,2,3], weights=[0.7,0.2,0.07,0.03])[0]
    customers.append({'customerid': f'CUST_{i:05d}', 'joindate': join_date.date(), 'city': city, 'segment': segment, 'support_tickets': support_tickets})

sales = []
order_id = 1
for _ in range(NUM_SALES):
    cust = random.choice(customers)
    cust_join = datetime.combine(cust['joindate'], datetime.min.time())
    sale_date = cust_join + timedelta(days=random.randint(1, (datetime.now() - cust_join).days))
    netsale = round(random.uniform(20, 1000),2)
    # occasional long inactivity
    if random.random() < 0.02:
        sale_date = datetime.now() - timedelta(days=random.randint(120, 730))
    sales.append({'orderid': f'ORD_{order_id:07d}', 'customerid': cust['customerid'], 'date': sale_date.date(), 'timestamp': sale_date, 'netsale': netsale, 'category': random.choice(['Medication','Wellness','Personal Care']), 'city': cust['city'], 'segment': cust['segment']})
    order_id += 1

customers_df = pd.DataFrame(customers)
sales_df = pd.DataFrame(sales)

customers_df.to_csv('customers_demo.csv', index=False)
sales_df.to_csv('sales_demo.csv', index=False)
print('Wrote customers_demo.csv and sales_demo.csv')
