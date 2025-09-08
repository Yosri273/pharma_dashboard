from celery import shared_task
from services.db import get_db_connection # INTEGRATION: Uses your existing DB service
from alerting import checks # The file with check functions (from plan V1)
from alerting.notifiers import Notifier
from datetime import datetime

@shared_task(name="alerting.run_all_checks")
def run_all_checks():
    """
    This task is run by a Celery worker. It queries the DB for checks
    that are due to be run and executes them.
    """
    db = get_db_connection()
    notifier = Notifier()

    # Get all active alert configs that are due to be run now
    # (e.g., last_run_time + frequency_minutes <= now)
    due_alerts = db.query("SELECT * FROM AlertConfiguration WHERE is_enabled = true AND ...") 
    
    current_states = db.query("SELECT * FROM ActiveAlertState").to_dict(key='alert_config_id')

    for alert_config in due_alerts:
        # 1. Dynamically get and run the check function from alerting.checks
        #    This check MUST query the production DB, not CSVs.
        #    e.g., check_function = getattr(checks, alert_config.check_function_name)
        #         is_triggered, message = check_function(**alert_config.config_params_json)
        
        check_function = getattr(checks, alert_config.check_function_name)
        params = alert_config.config_params_json
        (is_triggered, message) = check_function(db_conn=db, **params) # Pass the DB connection in

        # 2. Get previous state
        was_active = current_states.get(alert_config.id, {}).get("is_active", False)

        # 3. Apply state change logic
        now = datetime.utcnow()
        if is_triggered and not was_active:
            # STATE CHANGE: OK -> ALERT
            print(f"NEW ALERT: {alert_config.alert_name}")
            notifier.dispatch(...)
            db.execute(
                """INSERT INTO ActiveAlertState (alert_config_id, is_active, current_message, last_triggered_at) 
                   VALUES (?, true, ?, ?) ON CONFLICT(alert_config_id) DO UPDATE SET ...""",
                (alert_config.id, message, now)
            )
            
        elif not is_triggered and was_active:
            # STATE CHANGE: ALERT -> OK
            print(f"RESOLVED: {alert_config.alert_name}")
            # (Optionally send resolution notification)
            db.execute(
                "UPDATE ActiveAlertState SET is_active = false, last_resolved_at = ? WHERE alert_config_id = ?",
                (now, alert_config.id)
            )
        
        elif is_triggered and was_active:
            # STATE: REMAINS ACTIVE. Just update the message/timestamp
             db.execute(
                "UPDATE ActiveAlertState SET current_message = ?, last_triggered_at = ? WHERE alert_config_id = ?",
                (message, now, alert_config.id)
            )