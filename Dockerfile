FROM python:3.12-slim

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

# Install python dependencies for both FastAPI application and ETL pipeline
RUN pip install --no-cache-dir -r requirements.txt -r requirements_etl.txt

# Copy application files
COPY . .

# Expose FastAPI port
EXPOSE 7860

# Run uvicorn as default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
