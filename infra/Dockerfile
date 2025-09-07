# infra/Dockerfile

# --- Stage 1: Builder ---
# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster AS builder

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir=/app/wheels -r requirements.txt


# --- Stage 2: Final Application ---
FROM python:3.10-slim-buster AS final

WORKDIR /app

# Copy python dependencies from builder
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copy the entire application source code
# Make sure you have a .dockerignore file to exclude .venv, .git, etc.
COPY . .

# --- Application Entrypoint ---
# 1. Run the bootstrap script to create the DB and load data
# 2. Start the Gunicorn server
# This entrypoint script ensures the DB is ready before the server starts
COPY infra/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose the port Gunicorn will run on
EXPOSE 8000

# Set user (good practice to not run as root)
RUN useradd --create-home appuser
USER appuser

ENTRYPOINT ["/app/entrypoint.sh"]