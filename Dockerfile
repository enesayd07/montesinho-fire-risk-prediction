FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY data/holdout/ ./data/holdout/

EXPOSE 7860

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "7860"]