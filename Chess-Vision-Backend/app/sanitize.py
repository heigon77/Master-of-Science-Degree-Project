"""Make a digitized board placement into a legal, playable position.

Digitization can misclassify squares, producing impossible boards (a pawn on the
back rank, two kings of a colour, too many pieces). This fixes the common cases
so chess.js / python-chess accept the position, and reports what it changed.
"""
from __future__ import annotations

import chess

MAX_PER_SIDE = 16
MAX_PAWNS = 8
COLOR_NAME = {chess.WHITE: "white", chess.BLACK: "black"}


def _parse(placement: str) -> dict[int, chess.Piece]:
    pm: dict[int, chess.Piece] = {}
    rank, file = 7, 0
    for ch in placement.split()[0]:
        if ch == "/":
            rank, file = rank - 1, 0
        elif ch.isdigit():
            file += int(ch)
        elif 0 <= rank <= 7 and 0 <= file <= 7:
            try:
                pm[chess.square(file, rank)] = chess.Piece.from_symbol(ch)
            except ValueError:
                pass
            file += 1
    return pm


def sanitize(placement: str) -> dict:
    """Returns {fen, placement, corrections, valid}."""
    pm = _parse(placement)
    fixes: list[str] = []

    # 1. pawns can't be on rank 1 or 8
    for sq, p in list(pm.items()):
        if p.piece_type == chess.PAWN and chess.square_rank(sq) in (0, 7):
            del pm[sq]
            fixes.append(f"removed pawn on {chess.square_name(sq)} (impossible rank)")

    # 2. exactly one king per colour
    for color in (chess.WHITE, chess.BLACK):
        kings = [sq for sq, p in pm.items() if p.piece_type == chess.KING and p.color == color]
        for extra in kings[1:]:
            del pm[extra]
            fixes.append(f"removed extra {COLOR_NAME[color]} king on {chess.square_name(extra)}")
        if not kings:
            home = chess.E1 if color == chess.WHITE else chess.E8
            target = home if home not in pm else next(s for s in chess.SQUARES if s not in pm)
            pm[target] = chess.Piece(chess.KING, color)
            fixes.append(f"added missing {COLOR_NAME[color]} king on {chess.square_name(target)}")

    # 3. cap pawns (<=8) and total pieces (<=16) per side
    for color in (chess.WHITE, chess.BLACK):
        pawns = [sq for sq, p in pm.items() if p.piece_type == chess.PAWN and p.color == color]
        for sq in pawns[MAX_PAWNS:]:
            del pm[sq]
            fixes.append(f"removed extra {COLOR_NAME[color]} pawn on {chess.square_name(sq)}")
        own = [sq for sq, p in pm.items() if p.color == color]
        if len(own) > MAX_PER_SIDE:
            removable = [s for s in own if pm[s].piece_type != chess.KING]
            for sq in removable[: len(own) - MAX_PER_SIDE]:
                del pm[sq]
                fixes.append(f"removed excess {COLOR_NAME[color]} piece on {chess.square_name(sq)}")

    # 4. bishops: at most one per square colour (the natural pair → max 2 per side)
    for color in (chess.WHITE, chess.BLACK):
        seen = set()
        for sq, p in list(pm.items()):
            if p.piece_type == chess.BISHOP and p.color == color:
                sq_color = (chess.square_file(sq) + chess.square_rank(sq)) % 2  # 0 dark, 1 light
                if sq_color in seen:
                    del pm[sq]
                    fixes.append(f"removed extra {COLOR_NAME[color]} bishop on "
                                 f"{chess.square_name(sq)} (same square colour)")
                else:
                    seen.add(sq_color)

    # build board, drop castling/en-passant (can't be inferred from an image)
    board = chess.Board(None)
    for sq, p in pm.items():
        board.set_piece_at(sq, p)
    board.castling_rights = 0

    # 4. choose a side to move that is legal (the side not to move must not be in check)
    board.turn = chess.WHITE
    if not board.is_valid():
        board.turn = chess.BLACK

    return {
        "fen": board.fen(),
        "placement": board.board_fen(),
        "corrections": fixes,
        "valid": board.is_valid(),
    }
