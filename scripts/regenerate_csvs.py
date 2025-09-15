"""Regenerate synthetic CSV datasets for the pharma_dashboard project.
This script will create coherent, linked datasets with row counts between 200 and 5000.
Files generated: customer_data.csv, sales_data.csv, marketing_campaigns.csv, delivery_data.csv,
web_analytics.csv, mobile_analytics.csv, marketing_attribution.csv, crm_data.csv,
support_tickets.csv, funnel_data.csv, ad_platform_data.csv, competitor_data.csv, model_store/synthetic_customer_churn_predictions.csv

Use cautiously: this will overwrite existing CSV files in the repo root.
"""
import os
import random
from datetime import datetime, timedelta
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_STORE = os.path.join(ROOT, 'model_store')
if not os.path.exists(MODEL_STORE):
    os.makedirs(MODEL_STORE, exist_ok=True)

random.seed(42)

# Helper generators
CITIES = ['Riyadh', 'Jeddah', 'Dammam', 'Mecca', 'Medina', 'Khobar']
SEGMENTS = ['Retail', 'VIP', 'Corporate', 'Online']
CHANNELS = ['Web', 'Mobile', 'Retail']
CATEGORIES = ['Vitamins', 'Personal Care', 'Skincare', 'Supplements', 'Pain Relief', 'Medical Devices', 'Wellness', 'First Aid']

# Choose sizes: ensure between 200 and 5000
n_customers = 1200
n_sales = 5000
n_campaigns = 250
n_deliveries = 1200
n_web = 2500
n_mobile = 2500
n_attribution = 3000
n_crm = 800
n_support = 600
n_funnel = 2000
n_ad_platform = 400
n_competitor = 300

# 1) customers (canonical column names)
customers = []
for i in range(1, n_customers + 1):
    cid = f"C{i:05d}"
    join = datetime.now() - timedelta(days=random.randint(30, 1500))
    city = random.choice(CITIES)
    seg = random.choices(SEGMENTS, weights=[0.7,0.05,0.05,0.2])[0]
    customers.append({'customerid': cid, 'joindate': join.strftime('%Y-%m-%d %H:%M:%S'), 'city': city, 'segment': seg})
customers_df = pd.DataFrame(customers)
customers_df.to_csv(os.path.join(ROOT, 'customer_data.csv'), index=False)

# 2) products catalogue - curated Saudi pharmacy SKUs
# We'll define a curated list of common pharmacy products/brands found in Saudi Arabia
curated_products = [
    # Analgesics & antipyretics
    ("Panadol Extra", "Pain Relief", 12.0, 4.5),
    ("Panadol Advance", "Pain Relief", 18.0, 7.0),
    ("Aspirin 100mg", "Pain Relief", 8.0, 3.0),
    ("Voltaren Emulgel 50g", "Pain Relief", 45.0, 18.0),
    ("Brufen 400mg", "Pain Relief", 20.0, 8.0),
    # Cold & flu
    ("Vicks VapoRub 50g", "Cold & Flu", 15.0, 6.0),
    ("NeoCitran Hot Drink", "Cold & Flu", 22.0, 9.0),
    ("Otrivin Nasal Spray", "Cold & Flu", 25.0, 10.0),
    # Vitamins & supplements
    ("Centrum Multivitamins", "Vitamins", 95.0, 40.0),
    ("Redoxon Vitamin C 1000mg", "Vitamins", 30.0, 12.0),
    ("Omega-3 Fish Oil", "Vitamins", 85.0, 36.0),
    # Skincare & personal care
    ("Cetaphil Gentle Cleanser", "Skincare", 75.0, 30.0),
    ("Neutrogena Hydro Boost", "Skincare", 120.0, 48.0),
    ("Nivea Body Lotion 400ml", "Personal Care", 35.0, 14.0),
    ("Sensodyne Repair & Protect", "Oral Care", 28.0, 11.0),
    # Antiseptics & first aid
    ("Savlon Antiseptic 100ml", "First Aid", 18.0, 7.0),
    ("Band-Aid Assorted", "First Aid", 12.0, 4.8),
    # Dermatology & prescription OTC
    ("Clobetasol Cream 0.05% 30g", "Dermatology", 95.0, 40.0),
    ("Mometasone Furoate 0.1% 15g", "Dermatology", 110.0, 44.0),
    # Eye & ear care
    ("Refresh Tears Eye Drops", "Eye Care", 38.0, 15.0),
    ("Earol Ear Drops", "Ear Care", 22.0, 8.8),
    # Gastrointestinal
    ("Gaviscon 200ml", "Gastrointestinal", 45.0, 18.0),
    ("Imodium 2mg", "Gastrointestinal", 27.0, 10.8),
    # Baby care
    ("Johnson's Baby Oil 200ml", "Baby Care", 40.0, 16.0),
    ("Pampers Baby Dry", "Baby Care", 120.0, 48.0),
    # Women's health
    ("Duphaston 10mg", "Women's Health", 75.0, 30.0),
    ("Femibion Prenatal", "Vitamins", 160.0, 64.0),
    # Respiratory
    ("Ventolin Inhaler", "Respiratory", 220.0, 88.0),
    ("Symbicort Inhaler", "Respiratory", 420.0, 168.0),
    # Oral care extras
    ("Colgate Total", "Oral Care", 20.0, 8.0),
    # Misc common OTC
    ("Multivitamin Gummies Kids", "Vitamins", 55.0, 22.0),
]

# Expand curated list to reach a reasonable catalog size by creating variations
products = []
pid_seq = 1
for name, cat, base_price, base_cost in curated_products:
    for pack_variant in ["Pack", "Bottle", "Box"]:
        pid = f"P{pid_seq:04d}"
        pname = f"{name} {pack_variant}"
        # introduce small random price variance to create a catalog
        price = round(base_price * random.uniform(0.9, 1.2), 2)
        cost = round(base_cost * random.uniform(0.9, 1.2), 2)
        products.append({'productid': pid, 'productname': pname, 'category': cat, 'price': price, 'cost': cost})
        pid_seq += 1

# If still small, add generative 'store brand' variants
while len(products) < 400:
    pid = f"P{pid_seq:04d}"
    exemplar = random.choice(products)
    # exemplar uses lowercase keys
    pname = exemplar.get('productname', 'Store Product') + " - Store Brand"
    price = round(exemplar.get('price', 10.0) * random.uniform(0.6, 0.85), 2)
    cost = round(exemplar.get('cost', 4.0) * random.uniform(0.6, 0.85), 2)
    products.append({'productid': pid, 'productname': pname, 'category': exemplar.get('category', 'Misc'), 'price': price, 'cost': cost})
    pid_seq += 1

products_df = pd.DataFrame(products)
products_df.to_csv(os.path.join(ROOT, 'products_catalog.csv'), index=False)

# 3) sales
sales = []
start_date = datetime.now() - timedelta(days=540)
for i in range(1, n_sales+1):
    oid = f"ORD{i:06d}"
    ts = start_date + timedelta(seconds=random.randint(0, 540*24*3600))
    prod = products_df.sample(1).iloc[0]
    qty = random.choices([1,2,3,4,5], weights=[0.4,0.25,0.15,0.12,0.08])[0]
    # products_df uses lowercase keys
    gross = round(prod['price'] * qty, 2)
    discount = round(gross * random.choice([0, 0.05, 0.1, 0.15]) , 2)
    cogs = round(prod['cost'] * qty, 2)
    cust = customers_df.sample(1).iloc[0]['customerid']
    city = customers_df[customers_df['customerid']==cust].iloc[0]['city']
    loc = f"E{random.randint(1,120):03d}"
    channel = random.choices(CHANNELS, weights=[0.5,0.35,0.15])[0]
    status = random.choices(['Completed','Returned','Cancelled'], weights=[0.92,0.06,0.02])[0]
    # Add cart_abandoned flag and total_price for AOV calculations
    cart_abandoned = random.choice([0,0,0,1])
    total_price = round(gross - discount, 2)
    sales.append({'orderid': oid, 'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'), 'productid': prod['productid'], 'productname': prod['productname'], 'category': prod['category'], 'quantity': qty, 'grossvalue': gross, 'discountvalue': discount, 'costofgoodssold': cogs, 'customerid': cust, 'city': city, 'locationid': loc, 'channel': channel, 'orderstatus': status, 'cart_abandoned': cart_abandoned, 'total_price': total_price})
sales_df = pd.DataFrame(sales)
sales_df.to_csv(os.path.join(ROOT, 'sales_data.csv'), index=False)

# 4) marketing campaigns
campaigns = []
for i in range(1, n_campaigns+1):
    cid = f"CAMP{i:04d}"
    name = random.choice(['Vitamins Push', 'Skincare Promo','Ramadan Sale','Back to School','Winter Health','Summer Wellness']) + f" {i}"
    channel = random.choice(['Google','Meta','Snapchat','TikTok'])
    cost = round(random.uniform(2000, 50000), 2)
    impressions = random.randint(10000, 2000000)
    clicks = int(impressions * random.uniform(0.01, 0.08))
    sd = (datetime.now() - timedelta(days=random.randint(0,365))).date()
    ed = sd + timedelta(days=random.randint(7,120))
    campaigns.append({'campaignid': cid, 'campaignname': name, 'channel': channel, 'totalcost': cost, 'impressions': impressions, 'clicks': clicks, 'conversions': max(1, int(clicks * random.uniform(0.01,0.05))), 'startdate': sd.isoformat(), 'enddate': ed.isoformat()})
campaigns_df = pd.DataFrame(campaigns)
campaigns_df.to_csv(os.path.join(ROOT, 'marketing_campaigns.csv'), index=False)

# 5) delivery_data (one row per order delivery event—link to sales)
delivery = []
for i, row in sales_df.sample(n=min(len(sales_df), n_deliveries)).reset_index(drop=True).iterrows():
    did = f"D{i+1:06d}"
    order = row['orderid']
    ts = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S') + timedelta(days=random.randint(0,5))
    driver = f"DRV{random.randint(100,999)}"
    veh = random.choice(['Bike','Car','Van'])
    status = random.choices(['Delivered','Failed','In Transit'], weights=[0.9,0.05,0.05])[0]
    # Add support ticket counts and on_time probability
    support_tickets_for_order = random.choices([0,0,1], weights=[0.85,0.1,0.05])[0]
    delivery_cost = round(random.uniform(5.0, 25.0), 2)
    promised_dt = (ts - timedelta(days=random.randint(0,2))).strftime('%Y-%m-%d %H:%M:%S')
    # mark is_returned if original orderstatus indicates return or small random chance
    is_returned_flag = 1 if (row.get('orderstatus','').lower() == 'returned' or random.random() < 0.02) else 0
    delivery.append({'deliveryid': did, 'orderid': order, 'orderdate': row.get('timestamp', ''), 'actualdeliverydate': ts.strftime('%Y-%m-%d %H:%M:%S'), 'driverid': driver, 'vehicletype': veh, 'city': row.get('city', ''), 'status': status, 'promiseddate': promised_dt, 'deliverycost': delivery_cost, 'support_tickets': support_tickets_for_order, 'on_time': int(random.random() < 0.9), 'is_returned': is_returned_flag})
delivery_df = pd.DataFrame(delivery)
delivery_df.to_csv(os.path.join(ROOT, 'delivery_data.csv'), index=False)

# 6) web/mobile analytics: random sessions tied to customers or anonymous ids
def mk_sessions(n, channel):
    rows = []
    for i in range(n):
        sid = f"S{random.randint(100000,999999)}"
        ts = (datetime.now() - timedelta(days=random.randint(0,365))).strftime('%Y-%m-%d %H:%M:%S')
        cust = random.choice(list(customers_df['customerid']) + [None]*30)
        pageviews = random.randint(1,12)
        events = random.randint(0,6)
        city = random.choice(CITIES)
    rows.append({'sessionid': sid, 'timestamp': ts, 'customerid': cust or '', 'pageviews': pageviews, 'events': events, 'city': city, 'channel': channel, 'bounce': random.choice([0,0,0,1]), 'session_duration': random.randint(30,600), 'conversion': random.choice([0,0,1,0,0]), 'source': random.choice(['organic','paid','referral'])})
    return rows
web_df = pd.DataFrame(mk_sessions(n_web, 'Web'))
mobile_df = pd.DataFrame(mk_sessions(n_mobile, 'Mobile'))
web_df.to_csv(os.path.join(ROOT, 'web_analytics.csv'), index=False)
mobile_df.to_csv(os.path.join(ROOT, 'mobile_analytics.csv'), index=False)

# 7) marketing attribution (map sales to campaigns randomly)
attrib = []
for i, srow in sales_df.sample(n=min(len(sales_df), n_attribution)).reset_index(drop=True).iterrows():
    aid = f"A{i+1:06d}"
    campaign = campaigns_df.sample(1).iloc[0]['campaignid']
    attrib.append({'attributionid': aid, 'orderid': srow['orderid'], 'campaignid': campaign, 'revenue': srow['grossvalue'], 'channel': srow['channel'], 'platform': random.choice(['google','meta','snapchat','tiktok'])})
attrib_df = pd.DataFrame(attrib)
attrib_df.to_csv(os.path.join(ROOT, 'marketing_attribution.csv'), index=False)

# 8) CRM data: customers + interactions
crm = []
for i, c in customers_df.iterrows():
    cid = c['customerid']
    last_touch = datetime.now() - timedelta(days=random.randint(0,400))
    score = random.randint(10,100)
    # Include nps_score and clv placeholder for CLV card
    nps = random.randint(20, 100)
    crm.append({'customerid': cid, 'lasttouch': last_touch.strftime('%Y-%m-%d %H:%M:%S'), 'leadscore': score, 'city': c['city'], 'nps_score': nps, 'clv': round(random.uniform(50, 15000), 2)})
crm_df = pd.DataFrame(crm)
crm_df.to_csv(os.path.join(ROOT, 'crm_data.csv'), index=False)

# 9) support tickets
tickets = []
for i in range(1, n_support+1):
    tid = f"T{i:06d}"
    cust = customers_df.sample(1).iloc[0]['customerid']
    created = datetime.now() - timedelta(days=random.randint(0,400))
    status = random.choice(['open','closed','escalated'])
    category = random.choice(['Order Issue','Return','Product Question','Delivery'])
    tickets.append({'ticketid': tid, 'customerid': cust, 'createdat': created.strftime('%Y-%m-%d %H:%M:%S'), 'status': status, 'issue_type': category, 'resolution_time': random.randint(0,72), 'channel': random.choice(['email','phone','chat'])})
tickets_df = pd.DataFrame(tickets)
tickets_df.to_csv(os.path.join(ROOT, 'support_tickets.csv'), index=False)

# 10) funnel data (visitors -> cart -> checkout -> purchase aggregated daily)
funnel = []
for i in range(200):
    day = (datetime.now() - timedelta(days=i)).date()
    visitors = random.randint(200, 20000)
    atc = int(visitors * random.uniform(0.05, 0.25))
    checkout = int(atc * random.uniform(0.2, 0.6))
    purchases = int(checkout * random.uniform(0.5, 0.9))
    funnel.append({'date': day.isoformat(), 'visit': visitors, 'add_to_cart': atc, 'checkout': checkout, 'purchase': purchases})
funnel_df = pd.DataFrame(funnel)
funnel_df.to_csv(os.path.join(ROOT, 'funnel_data.csv'), index=False)

# 11) ad platform data
ads = []
for i in range(1, n_ad_platform+1):
    aid = f"AD{i:06d}"
    platform = random.choice(['Google','Meta','Snapchat'])
    cost = round(random.uniform(100, 20000), 2)
    clicks = random.randint(10, 5000)
    impressions = clicks * random.randint(10, 300)
    ads.append({'adid': aid, 'platform': platform, 'cost': cost, 'clicks': clicks, 'impressions': impressions, 'conversions': int(clicks * random.uniform(0.01,0.05)), 'roas': round(random.uniform(0.5, 5.0), 2)})
ad_df = pd.DataFrame(ads)
ad_df.to_csv(os.path.join(ROOT, 'ad_platform_data.csv'), index=False)

# 12) competitor data
comp = []
for i in range(1, n_competitor+1):
    pid = f"CP{i:04d}"
    name = f"Competitor {i}"
    price_index = round(random.uniform(0.7, 1.3), 2)
    comp.append({'competitorid': pid, 'name': name, 'price_index': price_index})
comp_df = pd.DataFrame(comp)
comp_df.to_csv(os.path.join(ROOT, 'competitor_data.csv'), index=False)

# 13) synthetic churn predictions (for model_store)
synth = []
for i, c in customers_df.sample(n=min(1000, len(customers_df))).reset_index(drop=True).iterrows():
    cid = c['customerid']
    churn_prob = round(random.random(), 3)
    est_ltv = round(random.uniform(50, 20000) * (1 - churn_prob), 2)
    synth.append({'customerid': cid, 'churn_probability': churn_prob, 'Estimated_LTV': est_ltv})
synth_df = pd.DataFrame(synth)
synth_df.to_csv(os.path.join(MODEL_STORE, 'synthetic_customer_churn_predictions.csv'), index=False)

print('Wrote synthetic CSVs:')
for fn in ['customer_data.csv','sales_data.csv','marketing_campaigns.csv','delivery_data.csv','web_analytics.csv','mobile_analytics.csv','marketing_attribution.csv','crm_data.csv','support_tickets.csv','funnel_data.csv','ad_platform_data.csv','competitor_data.csv', os.path.join('model_store','synthetic_customer_churn_predictions.csv')]:
    print(' -', fn)

print('Done.')
