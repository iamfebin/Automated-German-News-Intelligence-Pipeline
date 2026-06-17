FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies
# libgomp1 is required by faiss-cpu on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files first to leverage caching
COPY requirements.txt requirements_etl.txt ./

# Install python dependencies for both Streamlit and ETL pipeline
RUN pip install --no-cache-dir -r requirements.txt -r requirements_etl.txt

# Copy application files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run streamlit as default command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
