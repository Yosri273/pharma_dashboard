# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Realistic Mock Data Generator - V2.0 (Refactored for Quality)
#
# This script creates large, insightful, and realistic-looking datasets for
# the entire Pharma Analytics Hub application. It simulates a year's worth
# of data to enable meaningful analysis. This version has been refactored
# to adhere to professional coding standards (PEP 8).
# -----------------------------------------------------------------------------

import pandas
import random
from faker import Faker
from datetime import datetime, timedelta

# --- 1. CONFIGURATION ---
NUM_CUSTOMERS = 15000
NUM_PRODUCTS = 5000
NUM_SALES = 50000  # Total number of sales transactions
START_DATE = datetime(2024, 9, 1)
END_DATE = datetime(2025, 9, 1)

# Initialize Faker for Saudi Arabia to get realistic city names
fake = Faker('ar_SA')


# --- 2. MASTER DATA DEFINITION ---

def create_product_catalog():
    """
    Creates a master DataFrame of products with varied categories,
    prices, and costs.
    """
    products = []
    categories = [
        'Pain Relief', 'Vitamins', 'Supplements', 'First Aid',
        'Skincare', 'Personal Care', 'Medical Devices', 'Wellness'
    ]
    for i in range(1, NUM_PRODUCTS + 1):
        category = random.choice(categories)
        base_price = random.uniform(10, 200)
        products.append({
            'ProductID': f'P{i:03d}',
            'ProductName': f'{category} Product {i}',
            'Category': category,
            'BasePrice': round(base_price, 2),
            # Cost is simulated as 40-60% of the base price
            'CostOfGoodsSold': round(base_price * random.uniform(0.4, 0.6), 2)
        })
    return pandas.DataFrame(products)


def create_customers():
    """
    Creates a master DataFrame of customers with varied segments and join dates.
    """
    customers = []
    segments = ['Gold', 'Silver', 'Bronze']
    for i in range(1, NUM_CUSTOMERS + 1):
        # Customers can have joined up to a year before the start of our data
        join_date = fake.date_time_between(
            start_date=START_DATE - timedelta(days=365),
            end_date=END_DATE
        )
        customers.append({
            'CustomerID': f'C{i:03d}',
            'JoinDate': join_date.strftime('%Y-%m-%d %H:%M:%S'),
            'City': random.choice(['Riyadh', 'Jeddah', 'Dammam']),
            'Segment': random.choices(segments, weights=[0.2, 0.5, 0.3], k=1)[0]
        })
    return pandas.DataFrame(customers)


# --- 3. TRANSACTIONAL DATA GENERATION ---

def generate_sales_and_deliveries(customers_df, products_df):
    """
    Generates interconnected DataFrames for sales, deliveries, and marketing
    attribution based on the master customer and product data.
    """
    sales_data = []
    delivery_data = []
    attribution_data = []

    # Make loyal customers (Gold/Silver) more likely to purchase
    customer_weights = customers_df['Segment'].map(
        {'Gold': 4, 'Silver': 2, 'Bronze': 1}
    ).values
    customer_weights = customer_weights / customer_weights.sum()

    for i in range(1, NUM_SALES + 1):
        customer = customers_df.sample(weights=customer_weights).iloc[0]
        product = products_df.sample().iloc[0]

        timestamp = fake.date_time_between(start_date=START_DATE, end_date=END_DATE)
        quantity = random.randint(1, 5)
        gross_value = product['BasePrice'] * quantity
        # 30% chance of a discount up to 20%
        discount = gross_value*random.uniform(0,0.2) if random.random()<0.3 else 0
        # 5% return rate
        order_status = 'Returned' if random.random() < 0.05 else 'Completed'

        sales_data.append({
            'OrderID': f'ORD{i:03d}',
            'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'ProductID': product['ProductID'], 'ProductName': product['ProductName'],
            'Category': product['Category'], 'Quantity': quantity,
            'GrossValue': round(gross_value, 2),
            'DiscountValue': round(discount, 2),
            'CostOfGoodsSold': product['CostOfGoodsSold'] * quantity,
            'CustomerID': customer['CustomerID'], 'City': customer['City'],
            'LocationID': f'E{random.randint(1, 50):03d}',
            'Channel': random.choices(['Web','Mobile','Retail'],weights=[0.6,0.3,0.1])[0],
            'OrderStatus': order_status
        })

        # Generate a corresponding delivery record for each sale
        order_date = timestamp.date()
        promised_date = order_date + timedelta(days=random.randint(3, 5))
        delivery_status = 'Pending'
        actual_delivery = None

        if (END_DATE.date() - order_date).days > 7:  # Order is not very recent
            on_time = random.random() < 0.9  # 90% on-time rate
            delivery_offset = random.randint(2, 4) if on_time else random.randint(1, 3)
            actual_delivery_date = (promised_date - timedelta(days=1)) if on_time else (promised_date + timedelta(days=delivery_offset))
            actual_delivery = actual_delivery_date.strftime('%Y-%m-%d')
            delivery_status = 'Failed' if random.random() < 0.03 else 'Delivered'
        elif (END_DATE.date() - order_date).days > 3: # Order is recent
            delivery_status = 'Shipped'

        delivery_data.append({
            'DeliveryID': f'D{i:03d}', 'OrderID': f'ORD{i:03d}',
            'OrderDate': order_date.strftime('%Y-%m-%d'),
            'PromisedDate': promised_date.strftime('%Y-%m-%d'),
            'ActualDeliveryDate': actual_delivery, 'Status': delivery_status,
            'DeliveryPartner': random.choice(['Aramex', 'SMSA Express', 'DHL', 'FedEx', 'Local Carrier']),
            'City': customer['City'],
            'DeliveryCost': round(random.uniform(15, 25), 2)
        })

        # 70% of orders are attributed to a marketing campaign
        if random.random() < 0.7:
            attribution_data.append({
                'OrderID': f'ORD{i:03d}',
                'CampaignID': f'CAMP{random.randint(1, 5):03d}'
            })

    return (pandas.DataFrame(sales_data),
            pandas.DataFrame(delivery_data),
            pandas.DataFrame(attribution_data))


def generate_marketing_campaigns():
    """Generates a static list of sample marketing campaigns."""
    campaigns = [
        {'CampaignID': 'CAMP001', 'CampaignName': 'Google Ads - Q4 Push', 'Channel': 'Google', 'TotalCost': 15000, 'Impressions': 500000, 'Clicks': 25000},
        {'CampaignID': 'CAMP002', 'CampaignName': 'Meta Ads - Ramadan', 'Channel': 'Meta', 'TotalCost': 25000, 'Impressions': 800000, 'Clicks': 20000},
        {'CampaignID': 'CAMP003', 'CampaignName': 'Snapchat - Wellness Promo', 'Channel': 'Snapchat', 'TotalCost': 10000, 'Impressions': 1200000, 'Clicks': 30000},
        {'CampaignID': 'CAMP004', 'CampaignName': 'Google Ads - Vitamins', 'Channel': 'Google', 'TotalCost': 8000, 'Impressions': 300000, 'Clicks': 18000},
        {'CampaignID': 'CAMP005', 'CampaignName': 'Meta Ads - Skincare', 'Channel': 'Meta', 'TotalCost': 12000, 'Impressions': 600000, 'Clicks': 15000},
    ]
    for camp in campaigns:
        camp['StartDate'] = START_DATE.strftime('%Y-%m-%d')
        camp['EndDate'] = END_DATE.strftime('%Y-%m-%d')
    return pandas.DataFrame(campaigns)


# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    print("--- Starting Realistic Data Generation ---")

    products_df = create_product_catalog()
    customers_df = create_customers()
    sales_df, delivery_df, attribution_df = generate_sales_and_deliveries(
        customers_df, products_df
    )
    campaigns_df = generate_marketing_campaigns()
    # Create simple placeholders for other data files
    funnel_df = pandas.DataFrame({
        'Week': pandas.to_datetime(['2025-01-06', '2025-01-13']),
        'Visits': [10000, 12000], 'Carts': [1000, 1200], 'Orders': [250, 300]
    })
    competitor_df = pandas.DataFrame({
        'Date': ['2025-09-01'], 'Competitor': ['Nahdi'], 'ProductID': ['P001'],
        'ProductName': ['Aspirin 100mg'], 'Price': [12.50], 'OnPromotion': [True]
    })

    # Save all files to CSV
    sales_df.to_csv("sales_data.csv", index=False)
    delivery_df.to_csv("delivery_data.csv", index=False)
    customers_df.to_csv("customer_data.csv", index=False)
    campaigns_df.to_csv("marketing_campaigns.csv", index=False)
    attribution_df.to_csv("marketing_attribution.csv", index=False)
    funnel_df.to_csv("funnel_data.csv", index=False)
    competitor_df.to_csv("competitor_data.csv", index=False)

    print("\n--- Data Generation Complete ---")
    print(f"Generated {len(sales_df)} sales records.")
    print(f"Generated {len(customers_df)} customers.")
    print("All CSV files have been created/overwritten.")

