"""Human-like move prediction: CNN (piece + square classifiers) + Stockfish.

The CNN models predict the move a *human* would likely play (trained for white to
move; black positions are mirrored). Stockfish provides engine-strength moves and
a hybrid that keeps human-like moves which don't blunder.
"""
from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import onnxruntime as ort

PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
SQUARE_NAMES = ["pawn", "knight", "bishop", "rook", "queen", "king"]


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class MovePredictor:
    def __init__(self, models_dir: str, stockfish_path: str | None = None):
        d = Path(models_dir)
        prov = ["CPUExecutionProvider"]
        self.piece = ort.InferenceSession(str(d / "piece.int8.onnx"), providers=prov)
        self.squares = [ort.InferenceSession(str(d / f"square_{s}.int8.onnx"), providers=prov)
                        for s in SQUARE_NAMES]
        self.stockfish_path = (stockfish_path if stockfish_path and Path(stockfish_path).exists()
                               else None)

    # ── encoding: 12x8x8, ch 0-5 white P,N,B,R,Q,K; 6-11 black; row0 = rank8 ──
    def _encode(self, board: chess.Board):
        enc = np.zeros((12, 8, 8), np.float32)
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p:
                c = PIECE_ORDER.index(p.piece_type) + (0 if p.color == chess.WHITE else 6)
                enc[c, 7 - (sq >> 3), sq & 7] = 1.0
        return enc[None]

    def _cnn_white(self, board: chess.Board) -> dict:
        enc = self._encode(board)
        pieces = _softmax(self.piece.run(None, {"input": enc})[0].flatten())
        legal = set(board.legal_moves)
        mp: dict[str, float] = {}
        for i, ptype in enumerate(PIECE_ORDER):
            sq = (_softmax(self.squares[i].run(None, {"input": enc})[0].flatten()) + pieces[i]) / 2
            for fsq in board.pieces(ptype, chess.WHITE):
                fs = chess.square_name(fsq)
                for j in range(64):
                    try:
                        mv = chess.Move.from_uci(fs + chess.square_name(j))
                    except ValueError:
                        continue
                    if mv in legal:
                        mp[mv.uci()] = float(sq[j])
        return mp

    def cnn_scores(self, board: chess.Board) -> dict:
        """Move->score; handles black by mirroring the board and the moves."""
        if board.turn == chess.WHITE:
            return self._cnn_white(board)
        mirrored = self._cnn_white(board.mirror())
        out = {}
        for uci, s in mirrored.items():
            mv = chess.Move.from_uci(uci)
            real = chess.Move(chess.square_mirror(mv.from_square),
                              chess.square_mirror(mv.to_square), mv.promotion)
            out[real.uci()] = s
        return out

    def predict(self, fen: str, top_n: int = 3) -> dict:
        board = chess.Board(fen)
        cnn = [m for m, _ in sorted(self.cnn_scores(board).items(), key=lambda x: -x[1])]
        out = {"fen": fen, "turn": "white" if board.turn else "black",
               "cnn": cnn[:top_n], "stockfish": [], "hybrid": cnn[:top_n]}

        if self.stockfish_path:
            from stockfish import Stockfish
            sf = Stockfish(path=self.stockfish_path, parameters={"Threads": 1})
            sf.set_fen_position(fen)
            sf.update_engine_parameters({"MultiPV": top_n})
            out["stockfish"] = [t["Move"] for t in sf.get_top_moves(top_n) if t.get("Move")]

            # hybrid: among the top human-like CNN moves, prefer the one Stockfish
            # rates best (lowest eval for the opponent after the move).
            scored = []
            for mv in cnn[:max(6, top_n * 2)]:
                board.push_uci(mv)
                sf.set_fen_position(board.fen())
                ev = sf.get_evaluation()
                val = ev.get("value", 0) if ev.get("type") == "cp" else (10000 if ev.get("value", 0) > 0 else -10000)
                board.pop()
                scored.append((mv, val))      # val = opponent's eval after our move (lower = better for us)
            out["hybrid"] = [m for m, _ in sorted(scored, key=lambda x: x[1])][:top_n]
        return out
