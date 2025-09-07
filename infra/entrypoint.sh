#!/bin/sh
# infra/entrypoint.sh

echo "Starting Application Entrypoint..."

# Run the bootstrap script to initialize the database
echo "Running bootstrap script to set up database..."
python -m scripts.bootstrap

# Check if bootstrap was successful (e.g., check if db file exists)
if [ ! -f "sales.db" ]; then
    echo "Database file was not created. Bootstrap failed. Exiting."
    exit 1
fi

echo "Bootstrap complete. Starting Gunicorn server..."

# Start the Gunicorn production server
# It points to "app:server" which is the 'server' object in 'app/__init__.py'
exec gunicorn "app:server" \
    --bind 0.0.0.0:8000 \
    --workers 4 \
    --log-level $LOG_LEVEL \
    --access-logfile - \
    --error-logfile -