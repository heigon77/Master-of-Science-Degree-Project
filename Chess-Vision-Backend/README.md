---
title: Chess Vision Backend
emoji: ♟️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# ♟️ Chess Vision — Backend (API)

FastAPI service for my MSc project: **chess board digitization** + **human-like
move prediction**. Upload a board image to get its FEN, then ask for the moves a
human would likely play (CNN trained on Lichess games) combined with **Stockfish**.

Models are served as quantized **ONNX** (the original ~444 MB of PyTorch weights →
~47 MB ONNX), so the image is small and CPU inference is fast.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness probe |
| `POST` | `/digitize` | multipart image → FEN board placement |
| `POST` | `/predict-move` | `{ "fen": "...", "top_n": 3 }` → CNN / Stockfish / hybrid moves |
| `GET`  | `/docs` | Swagger UI |

```bash
curl -X POST .../predict-move -H 'Content-Type: application/json' \
  -d '{"fen":"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1","top_n":3}'
# -> {"cnn":["e7e5","g8f6","d7d5"], "stockfish":["e7e5","c7c5","e7e6"], "hybrid":[...]}
```

## How it works
- **Digitization** (`app/digitize.py`): fixed crop → Canny → Hough grid (exact MSc
  pipeline) → 64 square crops → ONNX MobileNetV2 piece classifier → FEN.
- **Move prediction** (`app/predict.py`): board → 12×8×8 tensor → a piece-type CNN +
  six destination-square CNNs → legal human-like moves; black is mirrored; Stockfish
  adds engine moves and a non-blundering hybrid.

## Run locally
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Stockfish: apt install stockfish (Linux) and set STOCKFISH_PATH, or omit for CNN-only
uvicorn app.main:app --reload --port 7860
```

## Models
`models_onnx/` holds the quantized ONNX weights (digitizer + piece + 6 square
classifiers). Regenerate from the original `.pth` with `convert_to_onnx.py`.

> The demo backend runs on a free Hugging Face Space that sleeps after inactivity;
> the first request after that triggers a short cold start.
