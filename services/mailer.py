"""
Enterprise Mailer Service

This module provides a centralized, secure function for sending emails with attachments.
It reads all SMTP configuration (host, user, password) directly from the 
central settings/environment variables.
"""

import smtplib
import ssl
from email.message import EmailMessage
from typing import List
from io import BytesIO
import logging

# Import configuration
from config import settings

logger = logging.getLogger(__name__)

def send_report_email(
    subject: str,
    body: str,
    recipients: List[str],
    pdf_attachment: BytesIO,
    attachment_name: str = "report.pdf"
):
    """
    Connects to the SMTP server and sends an email with a PDF attachment.
    
    All configuration is pulled from config.settings.
    """
    
    if not settings.SMTP_HOST or not settings.SMTP_USER:
        logger.error("SMTP_HOST or SMTP_USER not configured. Cannot send email.")
        return False

    if not recipients:
        logger.warning("No recipients specified for email report. Skipping.")
        return False

    # Create the base email message object
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings.SMTP_USER
    msg["To"] = ", ".join(recipients)
    msg.set_content(body)

    # Attach the PDF
    try:
        pdf_bytes = pdf_attachment.getvalue()
        msg.add_attachment(
            pdf_bytes,
            maintype="application",
            subtype="pdf",
            filename=attachment_name
        )
    except Exception as e:
        logger.error(f"Failed to attach PDF bytes: {e}", exc_info=True)
        return False

    # Create a secure SSL context and send the email
    context = ssl.create_default_context()
    logger.info(f"Connecting to SMTP server {settings.SMTP_HOST}:{settings.SMTP_PORT}...")
    
    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
            server.starttls(context=context)  # Secure the connection
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"Successfully sent report '{subject}' to {len(recipients)} recipient(s).")
        return True
    except smtplib.SMTPException as e:
        logger.error(f"SMTP Error: Failed to send report email: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during email sending: {e}", exc_info=True)
        return False