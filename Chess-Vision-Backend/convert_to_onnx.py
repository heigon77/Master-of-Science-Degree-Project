#!/usr/bin/env python3
"""Convert the trained .pth models to quantized ONNX (INT8) for serving.

Reads the weights from the Master's project, exports each model to ONNX and
applies dynamic INT8 quantization to shrink the ~444 MB of .pth files.

    python convert_to_onnx.py
"""

from __future__ import annotations

from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from app.chess_models import PieceClassifier, PieceImageClassifier, SquareClassifier

PTH_DIR = Path("../Master-of-Science-Degree-Project/Models")
OUT = Path("models_onnx")
OUT.mkdir(exist_ok=True)


def _find_state_dict(obj) -> dict:
    """Accept a bare state_dict or a checkpoint with an arbitrarily-named
    sub-dict of tensors (e.g. 'model_state_dict_PC')."""
    if isinstance(obj, dict) and obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, dict) and v and all(isinstance(t, torch.Tensor) for t in v.values()):
                return v
    raise ValueError("no state_dict found in checkpoint")


def load(model: torch.nn.Module, pth: str) -> torch.nn.Module:
    obj = torch.load(PTH_DIR / pth, map_location="cpu", weights_only=False)
    model.load_state_dict(_find_state_dict(obj))
    model.eval()
    return model


def export(model: torch.nn.Module, dummy: torch.Tensor, name: str, dyn: dict):
    fp32 = OUT / f"{name}.onnx"
    torch.onnx.export(
        model, dummy, str(fp32),
        input_names=["input"], output_names=["output"],
        dynamic_axes=dyn, opset_version=17, dynamo=False,
    )
    int8 = OUT / f"{name}.int8.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    fp32.unlink()  # keep only the quantized one
    return int8.stat().st_size / 1e6


def main() -> None:
    jobs = [
        (PieceImageClassifier(), "imageClassifierReal.pth", "digitizer",
         torch.randn(1, 3, 96, 96), {"input": {0: "b", 2: "h", 3: "w"}}),
        (PieceClassifier(), "pieceClassifier.pth", "piece",
         torch.randn(1, 12, 8, 8), {"input": {0: "b"}}),
    ]
    for piece in ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"]:
        jobs.append((SquareClassifier(), f"square{piece}Classifier.pth",
                     f"square_{piece.lower()}", torch.randn(1, 12, 8, 8), {"input": {0: "b"}}))

    total = 0.0
    for model, pth, name, dummy, dyn in jobs:
        try:
            load(model, pth)
            mb = export(model, dummy, name, dyn)
            total += mb
            print(f"  ✓ {name:16s} <- {pth:28s} {mb:6.1f} MB (int8)")
        except Exception as e:
            print(f"  ✗ {name:16s} <- {pth:28s} FAILED: {type(e).__name__}: {e}")

    print(f"\nTotal ONNX INT8: {total:.1f} MB  (from ~444 MB of .pth)")


if __name__ == "__main__":
    main()
