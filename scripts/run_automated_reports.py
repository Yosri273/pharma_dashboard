"""
Main Entrypoint for Automated Reporting Jobs.

This script is intended to be run by a scheduler (e.g., cron, Airflow).
It connects all the enterprise services:
1. Connects to the DB.
2. Runs scheduled logic (from etl.schedules) to get the report data payload.
3. Generates the PDF (using app.reporting).
4. Emails the report (using services.mailer).
"""

import sys
import os
from datetime import datetime
import logging

# --- Add Project Root to PATH ---
# This allows the script to import all our modules (config, services, app, etl)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- End Path Setup ---

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Import Project Modules ---
from services.db import get_engine
from config import settings
from etl.schedules import _get_weekly_sales_report_data
from app.reporting import generate_pdf_report
from services.mailer import send_report_email


def run_weekly_sales_report_job():
    """Orchestrates the complete weekly sales report job."""
    logger.info("Starting automated weekly sales report job...")
    
    try:
        engine = get_engine()
    except Exception as e:
        logger.critical(f"Failed to connect to database. Cannot run report. Error: {e}")
        return

    # 1. Get the data payload from our scheduled logic module
    payload = _get_weekly_sales_report_data(engine)
    
    if not payload:
        logger.error("Failed to generate report data payload. Job aborted.")
        return

    # 2. Generate the PDF in-memory (reusing our existing reporting module)
    try:
        pdf_bytes_io = generate_pdf_report(
            kpi_data=payload["kpi_data"],
            filters_dict=payload["filters_dict"],
            main_dataframe=payload["main_dataframe"],
            figures_list=payload["figures_list"],
            report_title=payload["report_title"],
            table_title=payload["table_title"]
        )
        logger.info("Successfully generated PDF report in memory.")
    except Exception as e:
        logger.error(f"Failed during PDF generation: {e}", exc_info=True)
        return
        
    # 3. Email the report
    today_str = datetime.now().strftime("%Y-%m-%d")
    subject = f"Pharma Weekly Sales Summary: {today_str}"
    body = "Please find the automated Weekly Sales Summary report attached.\n\nThis is an automated message."
    attachment_name = f"Weekly_Sales_Report_{today_str}.pdf"
    
    send_report_email(
        subject=subject,
        body=body,
        recipients=settings.REPORT_RECIPIENTS_LIST,
        pdf_attachment=pdf_bytes_io,
        attachment_name=attachment_name
    )
    
    logger.info("Automated weekly sales report job finished.")


if __name__ == "__main__":
    # This script can be expanded to accept arguments for different reports
    # (e.g., python run_automated_reports.py --report=sales)
    run_weekly_sales_report_job()