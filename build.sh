#!/usr/bin/env bash
# exit on error
set -o errexit

# Install the system dependencies that Kaleido's bundled Chromium needs
apt-get update && apt-get install -y \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libgtk-3-0 \
    libxss1 \
    libasound2 \
    lsb-release \
    xdg-utils

# This runs your original build command
pip install -r requirements.txt