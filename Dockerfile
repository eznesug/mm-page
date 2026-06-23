# syntax=docker/dockerfile:1
FROM python:3.13

WORKDIR /app

# Install system dependencies needed for scikit-learn C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clear any corrupted pip cache from previous builds
RUN pip cache purge

COPY requirements.txt .
RUN pip install --no-cache-dir --force-reinstall -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]