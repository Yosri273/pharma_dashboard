# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# E-commerce Web Scraper - V1.0
#
# This engine is responsible for scraping competitor websites to gather
# pricing and promotional data. Initially, it works with local mock files.
# -----------------------------------------------------------------------------

import pandas
from requests_html import HTML
from datetime import datetime

# --- CONFIGURATION ---
# In Phase 2, we will replace these file paths with live URLs
COMPETITOR_TARGETS = {
    "Nahdi": {"type": "file", "path": "nahdi_mock.html"},
    "Al-Dawaa": {"type": "file", "path": "al_dawaa_mock.html"},
}
OUTPUT_FILE = "scraped_competitor_data.csv"

# --- PARSING LOGIC ---
# Each competitor website has a different structure, so we need a
# specific function to parse each one.

def parse_nahdi(html_content):
    """Parses the HTML content from the mock Nahdi website."""
    data = []
    products = html_content.find('.product-card')
    for product in products:
        try:
            name = product.find('.product-name', first=True).text
            price = float(product.find('.product-price', first=True).text.replace(' SAR', ''))
            on_promo = bool(product.find('.promo-tag', first=True))
            data.append({'productname': name, 'price': price, 'onpromotion': on_promo})
        except AttributeError:
            # This handles cases where a product card might be missing a price or name
            print("  [WARNING] Skipping a malformed product card on Nahdi.")
            continue
    return data

def parse_al_dawaa(html_content):
    """Parses the HTML content from the mock Al-Dawaa website."""
    data = []
    products = html_content.find('.item')
    for product in products:
        try:
            name = product.find('h1', first=True).text
            price = float(product.find('.price', first=True).text.replace('Price: ', ''))
            on_promo = bool(product.find('strong', first=True))
            data.append({'productname': name, 'price': price, 'onpromotion': on_promo})
        except (AttributeError, ValueError):
            # This handles cases where a product might be missing a price/name or price is not a number
            print("  [WARNING] Skipping a malformed product card on Al-Dawaa.")
            continue
    return data

# A mapping to call the correct parsing function
PARSER_MAP = {
    "Nahdi": parse_nahdi,
    "Al-Dawaa": parse_al_dawaa
}

# --- MAIN SCRAPING ORCHESTRATOR ---
def run_scraper():
    """Main function to orchestrate the scraping process."""
    print("--- Starting Competitor Scraping Engine ---")
    all_competitor_data = []
    today = datetime.now().strftime('%Y-%m-%d')

    for competitor, target in COMPETITOR_TARGETS.items():
        print(f"Scraping {competitor} from '{target['path']}'...")
        try:
            with open(target['path'], 'r', encoding='utf-8') as f:
                html_source = f.read()
            
            html = HTML(html=html_source)
            parser_func = PARSER_MAP[competitor]
            scraped_data = parser_func(html)
            
            for item in scraped_data:
                item['competitor'] = competitor
                item['date'] = today
            
            all_competitor_data.extend(scraped_data)
            print(f"  [SUCCESS] Scraped {len(scraped_data)} items from {competitor}.")

        except Exception as e:
            print(f"  [ERROR] Failed to scrape {competitor}. Error: {e}")

    if not all_competitor_data:
        print("Scraping finished with no data collected.")
        return

    # Convert to DataFrame and save to CSV
    df = pandas.DataFrame(all_competitor_data)
    # Reorder columns for clarity
    df = df[['date', 'competitor', 'productname', 'price', 'onpromotion']]
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n--- Scraping Finished ---")
    print(f"Successfully saved {len(df)} total records to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    run_scraper()

