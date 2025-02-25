FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir .

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models
ENV DATA_PATH=/app/data

# Default command
CMD ["python", "-m", "src.api.main"] 