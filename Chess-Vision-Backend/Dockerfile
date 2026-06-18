# Hugging Face Space (Docker SDK) listens on 7860.
FROM python:3.12-slim

# Stockfish (apt build = portable across CPUs) + opencv runtime libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        stockfish libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models_onnx ./models_onnx

ENV STOCKFISH_PATH=/usr/games/stockfish \
    MODELS_DIR=/app/models_onnx

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
