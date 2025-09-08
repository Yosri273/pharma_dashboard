"""
Main Alerting Engine (Corrected with Relative Imports)

This module loads all alert configurations from config/alerts.yml,
dynamically executes the corresponding check function from alerting.checks,
and triggers the notifier if an alert condition is met.

It also manages alert STATE to prevent spamming. An alert is only
sent when the state changes from OK -> ALERT. A resolution message
is sent when it changes from ALERT -> OK.
"""
import yaml
import json
import os
from datetime import datetime

# --- IMPORT FIX APPLIED ---
# Use relative imports '.' to correctly import sibling modules within the same package.
from . import checks
from .notifiers import Notifier
# --------------------------

CONFIG_PATH = "config/alerts.yml"
# This state file acts as our simple database of active alerts.
# In production, this should be a SQL table or Redis.
STATE_FILE_PATH = "cache-directory/active_alerts.json" 

class AlertMonitor:
    def __init__(self):
        self.notifier = Notifier()
        self.alert_configs = self._load_config()
        self.alert_state = self._load_state()

    def _load_config(self):
        print("Loading alert configurations...")
        try:
            with open(CONFIG_PATH, 'r') as f:
                config = yaml.safe_load(f)
            return config.get("alerts", [])
        except FileNotFoundError:
            print(f"ERROR: Alert configuration file not found at {CONFIG_PATH}")
            return []


    def _load_state(self):
        """Loads the current state of active alerts from the state file."""
        os.makedirs(os.path.dirname(STATE_FILE_PATH), exist_ok=True)
        if not os.path.exists(STATE_FILE_PATH):
            return {}
        try:
            with open(STATE_FILE_PATH, 'r') as f:
                content = f.read()
                if not content: # Handle empty file
                    return {}
                return json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError):
            print("Warning: Could not decode alert state file. Starting fresh.")
            return {}

    def _save_state(self):
        """Saves the updated state of active alerts to the state file."""
        os.makedirs(os.path.dirname(STATE_FILE_PATH), exist_ok=True)
        with open(STATE_FILE_PATH, 'w') as f:
            json.dump(self.alert_state, f, indent=2)

    def run_all_checks(self):
        """
        Main entry point. Iterates all configured alerts, runs checks, 
        and manages state transitions.
        """
        print(f"\nRunning all alert checks at {datetime.now().isoformat()}...")
        if not self.alert_configs:
            print("No alerts configured in config/alerts.yml.")
            return

        for alert_config in self.alert_configs:
            if not alert_config.get("enabled", False):
                continue

            alert_name = alert_config["name"]
            was_active = self.alert_state.get(alert_name, {}).get("is_active", False)

            try:
                check_function_name = alert_config["kpi_check_function"]
                check_function = getattr(checks, check_function_name, None)
                
                if not check_function:
                    print(f"ERROR: Check function '{check_function_name}' not found in alerting/checks.py.")
                    continue
                
                params = alert_config.get("parameters", {})
                is_triggered, message = check_function(**params)
                
                if is_triggered and not was_active:
                    print(f"  [NEW ALERT]: {alert_name}")
                    self.notifier.dispatch(
                        channels=alert_config["channels"],
                        recipients=alert_config["recipients"],
                        alert_name=alert_name,
                        message=message
                    )
                    self.alert_state[alert_name] = {"is_active": True, "message": message, "last_triggered": datetime.now().isoformat()}
                
                elif not is_triggered and was_active:
                    print(f"  [RESOLVED]: {alert_name}")
                    self.alert_state[alert_name] = {"is_active": False, "message": "Resolved", "last_resolved": datetime.now().isoformat()}
                
                elif is_triggered and was_active:
                    print(f"  [STILL ACTIVE]: {alert_name}")
                    # Update message in case the value changed (e.g., delay is now 20 hours instead of 10)
                    self.alert_state[alert_name]["message"] = message
                    self.alert_state[alert_name]["last_triggered"] = datetime.now().isoformat()

            except Exception as e:
                print(f"ERROR executing check for '{alert_name}': {e}")

        self._save_state()
        print("Alert checks complete.")


def run_monitor_cycle():
    """Main function to run one cycle of the monitor."""
    monitor = AlertMonitor()
    monitor.run_all_checks()

if __name__ == "__main__":
    run_monitor_cycle()