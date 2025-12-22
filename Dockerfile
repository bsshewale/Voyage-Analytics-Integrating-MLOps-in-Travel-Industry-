# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app ./app
COPY Models ./Models

# Expose Flask port
EXPOSE 8000

# Run app
CMD ["python", "app/api_price_predictor.py"]
