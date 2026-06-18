"""Chess Vision API — board digitization + human-like move prediction.

  GET  /health
  POST /digitize      multipart image -> FEN (board placement)
  POST /predict-move  { fen, top_n } -> CNN / Stockfish / hybrid moves
  GET  /docs          Swagger UI
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2 as cv
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .digitize import image_to_fen
from .predict import MovePredictor
from .sanitize import sanitize

ROOT = Path(__file__).parent.parent
MODELS_DIR = os.environ.get("MODELS_DIR", str(ROOT / "models_onnx"))
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", str(ROOT / "stockfish" / "stockfish"))

app = FastAPI(
    title="Chess Vision API",
    description="Digitize a chess board photo to a FEN, then predict human-like moves "
                "(CNN trained on Lichess games) combined with Stockfish.",
    version="0.1.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# loaded once at startup
_digitizer = ort.InferenceSession(str(Path(MODELS_DIR) / "digitizer3d.fp32.onnx"),
                                  providers=["CPUExecutionProvider"])
_predictor = MovePredictor(MODELS_DIR, stockfish_path=STOCKFISH_PATH)


class PredictRequest(BaseModel):
    fen: str = Field(..., description="FEN (full or placement-only).",
                     examples=["rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"])
    top_n: int = Field(3, ge=1, le=8)


@app.get("/")
def root():
    return {"service": "Chess Vision API", "version": app.version,
            "stockfish": _predictor.stockfish_path is not None, "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "stockfish": _predictor.stockfish_path is not None}


@app.post("/digitize")
async def digitize(file: UploadFile = File(...)):
    """Upload a board image (render-style) and get its FEN board placement."""
    data = await file.read()
    img = cv.imdecode(np.frombuffer(data, np.uint8), cv.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode the image.")
    placement = image_to_fen(img, _digitizer)
    if placement is None:
        raise HTTPException(422, "Could not detect a full 8x8 board in the image.")
    fixed = sanitize(placement)  # repair impossible positions (back-rank pawns, king count, …)
    return {
        "raw_placement": placement,
        "placement": fixed["placement"],
        "fen": fixed["fen"],
        "corrections": fixed["corrections"],
        "valid": fixed["valid"],
    }


@app.post("/predict-move")
def predict_move(req: PredictRequest):
    try:
        return _predictor.predict(req.fen, top_n=req.top_n)
    except ValueError as e:
        raise HTTPException(400, f"Invalid FEN: {e}")
