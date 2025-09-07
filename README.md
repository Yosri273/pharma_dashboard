# E-Commerce Analytics Dashboard

This repository contains the code for a Dash-based analytics dashboard for monitoring e-commerce sales data. The project is structured as a scalable, enterprise-ready Python package with a clean separation of concerns.

## Project Structure

The project follows a modular layout separating the web application layer from business logic and data services.

- **/app/**: Contains the Dash application itself.
  - `__init__.py`: The application factory (`create_app`) which initializes Dash, caching, and logging, and wires up the layout and callbacks.
  - `layout.py`: Defines all Dash/Plotly layout components and visual plotting helper functions.
  - `callbacks.py`: A "thin" controller layer containing only Dash callback wiring. It coordinates calls to services and transform modules.
- **/etl/**: The core Extract, Transform, Load (ETL) pipeline and business logic.
  - `ingest.py`: Functions for extracting data from sources (CSVs in this case) and loading it into the database.
  - `transforms.py`: All data manipulation, business logic, and aggregation functions (using Pandas). This module is 100% UI-agnostic.
  - `schedules.py`: Contains scheduler logic (APScheduler) to run ingestion jobs automatically. Run as a separate process.
- **/services/**: Modules for interacting with external infrastructure.
  - `db.py`: A dedicated service for all database interactions (connections, queries, inserts).
- **/models/**: Pydantic domain models for defining clear data contracts within the application.
  - `domain.py`: Defines models like `DashboardKPIs` to ensure type-safe data passing.
- **/config/**: Application configuration.
  - `settings.py`: Pydantic `BaseSettings` model to load all config from environment variables.
- **/scripts/**: Standalone utility scripts.
  - `bootstrap.py`: An initialization script to create DB tables and run the first data ingestion.
- **/infra/**: Deployment and CI/CD infrastructure.
  - `Dockerfile`: A multi-stage production Docker build.
  - `docker-compose.yml`: (To be added) For local development.
  - `entrypoint.sh`: Helper script for the Docker container to run bootstrap and start the server.
- **/tests/**: Unit and integration tests.
  - `test_transforms.py`: Pytest unit tests for our core business logic.

---

## Getting Started

### Prerequisites

- Python 3.9+
- Pip (Python Package Installer)
- (Optional) Docker and Docker Compose

### 1. Local Development Setup

**1. Create a Virtual Environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate