# infra/Dockerfile

# --- Stage 1: Builder ---
FROM python:3.11-slim-bullseye AS builder

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libffi-dev \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and build wheels
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir=/app/wheels -r requirements.txt

# --- Stage 2: Final Application ---
FROM python:3.11-slim-bullseye AS final

WORKDIR /app

# Copy Python wheels from builder
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copy the entire application code
COPY . .

# Copy entrypoint script and make executable
COPY infra/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose Dash/Gunicorn port
EXPOSE 8000

# Create non-root user
RUN useradd --create-home appuser
USER appuser

# Entrypoint to run bootstrap + server
ENTRYPOINT ["/app/entrypoint.sh"]