FROM python:3.10-slim

WORKDIR /app

# Install system deps (needed for xgboost)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies first (CACHE LAYER)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy app and models
COPY app ./app
COPY Models ./Models

EXPOSE 8000

CMD ["python", "app/api_price_predictor.py"]
