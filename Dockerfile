# Multi-stage Dockerfile for Enterprise Financial Document Intelligence RAG
FROM python:3.11-slim

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and data
COPY src/ ./src/
COPY api/ ./api/
COPY ui/ ./ui/
COPY data/ ./data/
COPY .env.example .env

# Expose FastAPI (8000) and Streamlit (8501) ports
EXPOSE 8000 8501

# Default command: launch FastAPI server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

