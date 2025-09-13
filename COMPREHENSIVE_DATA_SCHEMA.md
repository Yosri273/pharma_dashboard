This file documents the exact CSV/table schemas the Comprehensive tab expects.

Purpose
- Provide the enterprise data team with exact column names and example rows so the app's KPIs and charts render fully.

Notes
- Column name matching is case-sensitive in some places; prefer the canonical names below.
- The app accepts both canonical TABLE_CONFIG table names and some legacy names; ensure these columns exist in whichever table is supplied.

Tables and required columns

1) web_analytics (web_analytics.csv)
- date (YYYY-MM-DD), session_id, source, medium, device, pageviews (int), session_duration (seconds, int), bounce (boolean), conversion (boolean/int), user_id (optional)
- Example row:
  2025-09-01, sess_123, organic, organic, Desktop, 3, 120, False, 0, user_1

2) mobile_analytics (mobile_analytics.csv)
- date, app_session_id (or session_id), device (or os), events (int), session_duration (int), conversion (boolean/int), customerid (optional)
- Example row:
  2025-09-01, msess_1, Mobile, 4, 300, 1, demo_c_1

3) ad_platform_data (ad_platform_data.csv)
- platform, campaign_id, impressions (int), clicks (int), spend (float), conversions (int)
- Example row:
  Facebook, cmp_123, 10000, 500, 250.00, 20

4) marketing_attribution (marketing_attribution.csv)
- orderid, campaignid
- Example row:
  order_1, cmp_123

5) marketing_campaigns
- campaignid, campaignname, channel, totalcost, impressions, clicks, startdate, enddate

6) sales_data (sales_data.csv)
- OrderID, Timestamp (ISO), ProductID, ProductName, Category, Quantity (int), GrossValue (float), DiscountValue (float), NetSale (float), CostOfGoodsSold (float), CustomerID, City, LocationID, Channel, OrderStatus
- Example row:
  demo_ord_1, 2025-09-01T12:34:56, P-1, "Demo Product", Demo, 2, 200.00, 20.00, 180.00, 90.00, demo_c_1, Riyadh, L-1, Online, Completed

7) delivery_data (delivery_data.csv)
- DeliveryID, OrderID, OrderDate, PromisedDate, ActualDeliveryDate, Status, DriverID, VehicleType, City, DeliveryCost

8) crm_data / customer_data
- CustomerID, JoinDate (ISO or YYYY-MM-DD), City, Segment, nps_score (optional numeric)

9) funnel_data
- Week (YYYY-WW), Visits (int), Carts (int), Orders (int)

10) support_tickets
- ticket_id, date, issue_type, status, resolution_time (hours), customerid, city

11) competitor_data (optional)
- Date, Competitor, ProductID, ProductName, Price, OnPromotion (boolean)

Deployment notes
- The demo insertion script (`scripts/insert_demo_data.py`) writes demo rows with identifiable prefixes (e.g., `demo_ord_`) so they can be filtered/removed.
- If your warehouse or ETL uses different column names, normalize them to the above canonical names before inserting into the DB.

Contact
- For questions about mapping or normalization, update the `TABLE_CONFIG` mapping in `config/settings.py` and the `scripts/insert_demo_data.py` normalizer.
