"""
Notification Dispatch Module

This module handles sending alerts to the configured channels (Email, Slack).
It integrates with the existing services/mailer.py for email
and adds a new function for Slack webhooks.
"""
import requests
import os
from config import settings # Assumes settings.py loads env vars
from services.mailer import send_report_email # INTEGRATION: Uses existing mailer

# You MUST add this to your config/settings.py or environment variables
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")

class Notifier:
    """Handles dispatching notifications for triggered alerts."""

    def __init__(self):
        if not SLACK_WEBHOOK_URL:
            print("WARNING: SLACK_WEBHOOK_URL is not set. Slack notifications will fail.")
        
    def dispatch(self, channels: list, recipients: dict, alert_name: str, message: str):
        """
        Routes the alert message to all specified channels.
        
        :param channels: List of strings, e.g., ['email', 'slack']
        :param recipients: Dict mapping channels to recipient lists/names
        :param alert_name: The name of the alert (for subject/title)
        :param message: The detailed alert content
        """
        print(f"Dispatching alert '{alert_name}' to channels: {channels}")
        
        if "email" in channels and "email" in recipients:
            try:
                # INTEGRATION: This function must exist in services/mailer.py
                # Assumes signature: send_report_email(subject, body, recipient_list)
                send_report_email(
                    subject=f"KPI Alert: {alert_name}",
                    body=message,
                    recipient_list=recipients["email"]
                )
                print("Successfully dispatched to Email.")
            except Exception as e:
                print(f"ERROR: Failed to send email: {e}")

        if "slack" in channels and "slack_channel" in recipients:
            try:
                self._send_slack_message(
                    channel=recipients["slack_channel"],
                    alert_name=alert_name,
                    message=message
                )
                print("Successfully dispatched to Slack.")
            except Exception as e:
                print(f"ERROR: Failed to send Slack message: {e}")

    def _send_slack_message(self, channel: str, alert_name: str, message: str):
        """Helper function to send a formatted Slack message via Webhook."""
        if not SLACK_WEBHOOK_URL:
            raise ValueError("Cannot send Slack message: SLACK_WEBHOOK_URL is not configured.")
            
        payload = {
            "channel": channel,
            "username": "KPI Alert Bot",
            "icon_emoji": ":warning:",
            "attachments": [
                {
                    "color": "#D00000", # Red alert color
                    "pretext": f"*KPI Alert: {alert_name}*",
                    "text": message
                }
            ]
        }
        response = requests.post(SLACK_WEBHOOK_URL, json=payload)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
