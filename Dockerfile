FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    poppler-utils \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

COPY app/req.txt /app/req.txt
RUN pip install --no-cache-dir -r /app/req.txt

COPY app/main.py /app/main.py

ENV FIREBASE_CRED_PATH=/run/secrets/serviceAccount.json

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
