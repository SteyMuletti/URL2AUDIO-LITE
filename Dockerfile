FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN printf '#!/bin/sh\nexec gunicorn server:app --workers 2 --timeout 300 --bind "0.0.0.0:${PORT:-8000}"\n' > /start.sh && chmod +x /start.sh

CMD ["/start.sh"]
