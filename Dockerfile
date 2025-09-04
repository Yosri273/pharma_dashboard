# -----------------------------------------------------------------------------
# Dockerfile for the Pharma Analytics Hub
# This file is the blueprint for creating a standardized, portable container
# for our Python Dash application.
# -----------------------------------------------------------------------------

# Step 1: Start from an official Python base image.
# We'll use a specific version for consistency.
FROM python:3.11-slim

# Step 2: Set the working directory inside the container.
# This is where our application files will live.
WORKDIR /app

# Step 3: Copy the list of required libraries into the container.
COPY requirements.txt .

# Step 4: Install the Python libraries.
# The `--no-cache-dir` option makes the image smaller.
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy all the application files (app.py, etc.) into the container.
COPY . .

# Step 6: Tell Docker that the application will run on port 8050.
EXPOSE 8050

# Step 7: Define the command to run when the container starts.
# This is the same as running `python app.py` in your terminal.
CMD ["python", "app.py"]
