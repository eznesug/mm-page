# syntax=docker/dockerfile:1
FROM python:3.13

WORKDIR /app

# Install system dependencies needed for scikit-learn C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Clear Railway's broken PIP_FIND_LINKS that points to stale cached wheels
# --isolated ignores all pip config files, --no-index prevents local dir search
RUN PIP_FIND_LINKS="" pip install --isolated --no-cache-dir --force-reinstall -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]