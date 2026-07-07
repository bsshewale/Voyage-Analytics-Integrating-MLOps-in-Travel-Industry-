FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/flight_price_service ./flight_price_service

WORKDIR /app/flight_price_service

EXPOSE 5000

CMD ["python", "api.py"]