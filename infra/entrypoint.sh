#!/bin/sh

# Exit immediately if any command fails
set -e

# 1. Run the Database Bootstrap Script
echo "Running bootstrap script to set up database..."
python -m scripts.bootstrap

# 2. Start the Gunicorn Web Server
# If the script gets here, bootstrap was successful.
echo "Bootstrap successful. Starting web server..."

# 'exec' replaces the shell with the Gunicorn process.
# This binds to the PORT variable that Render provides automatically.
exec gunicorn run:server --bind 0.0.0.0:${PORT}