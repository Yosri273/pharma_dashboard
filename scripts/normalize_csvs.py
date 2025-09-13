import pandas as pd
import os

# Load customer master data
customer_df = pd.read_csv('customer_data.csv')
customer_map = {}
for idx, row in customer_df.iterrows():
    # Map all possible user/customer IDs to CustomerID
    customer_map[row['CustomerID']] = row['CustomerID']
    # If you have a mapping file, extend here

# Helper to map user_id/userXXX to CustomerID
# For now, just replace 'user' with 'C' and pad with zeros if needed
# You can extend this logic if you have a mapping file

def user_to_customer_id(user_id):
    if isinstance(user_id, str) and user_id.startswith('user'):
        num = user_id.replace('user','')
        return f"C{num.zfill(3)}"
    return user_id

# Normalize CRM data
crm = pd.read_csv('crm_data.csv')
crm['CustomerID'] = crm['customer_id'].apply(user_to_customer_id)
crm['City'] = crm['CustomerID'].map(customer_df.set_index('CustomerID')['City'])
# Always ensure only one Segment column
if 'Segment' in crm.columns:
    crm.drop('Segment', axis=1, inplace=True)
crm['Segment'] = crm['CustomerID'].map(customer_df.set_index('CustomerID')['Segment'])
crm.drop('customer_id', axis=1, inplace=True)
# Remove any duplicate columns just in case
crm = crm.loc[:,~crm.columns.duplicated()]
crm.to_csv('crm_data.csv', index=False)

# Normalize web analytics
web = pd.read_csv('web_analytics.csv')
web['CustomerID'] = web['user_id'].apply(user_to_customer_id)
web['City'] = web['CustomerID'].map(customer_df.set_index('CustomerID')['City'])
web.drop('user_id', axis=1, inplace=True)
web.to_csv('web_analytics.csv', index=False)

# Normalize mobile analytics
mobile = pd.read_csv('mobile_analytics.csv')
mobile['CustomerID'] = mobile['user_id'].apply(user_to_customer_id)
mobile['City'] = mobile['CustomerID'].map(customer_df.set_index('CustomerID')['City'])
mobile.drop('user_id', axis=1, inplace=True)
mobile.to_csv('mobile_analytics.csv', index=False)

# Normalize support tickets
support = pd.read_csv('support_tickets.csv')
support['CustomerID'] = support['customer_id'].apply(user_to_customer_id)
support['City'] = support['CustomerID'].map(customer_df.set_index('CustomerID')['City'])
support.drop('customer_id', axis=1, inplace=True)
support.to_csv('support_tickets.csv', index=False)

# You can repeat similar logic for other CSVs as needed
print('Normalization complete. Original CSVs have been overwritten with normalized data.')
