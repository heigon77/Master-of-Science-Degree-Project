"""Image -> FEN digitization (the exact Preprocess.ipynb pipeline).

Runs on any uploaded image (no corner marking): fixed crop, Canny, Hough,
group/filter lines, remove the spurious 3rd-from-bottom row, crop each square
(bbox + 90px), classify with the ONNX piece model, and detect empties by
model-confidence + edge density.
"""
from __future__ import annotations

import cv2 as cv
import numpy as np

IDX2SYM = {0:"P",1:"N",2:"B",3:"R",4:"Q",5:"K",6:"p",7:"n",8:"b",9:"r",10:"q",11:"k"}
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


# ── exact Preprocess.ipynb grid ──────────────────────────────────────────────
def _crop_image(im):
    cx, cy = im.shape[1] // 2, im.shape[0] // 2
    return im[cy - 520:cy + 450, cx - 550:cx + 550]

def _group(lines):
    g = []
    for l in lines:
        rho, theta = l[0]
        if not any(abs(rho - a[0][0]) < 30 and abs(theta - a[0][1]) < np.pi / 18 for a in g):
            g.append(l)
    return g

def _filter(g):
    mr = None
    for l in g:
        rho, theta = l[0]
        if 1 < theta < 3 and (mr is None or rho < mr):
            mr = rho
    return [l for l in g if l[0][0] != mr]

def _intersections(board_lines):
    lp = []
    for l in board_lines:
        rho, theta = l[0]; a, b = np.cos(theta), np.sin(theta); x0, y0 = a * rho, b * rho
        lp.append([(int(x0 + 1e4 * -b), int(y0 + 1e4 * a)),
                   (int(x0 - 1e4 * -b), int(y0 - 1e4 * a)), theta])
    rows = []
    for i in range(len(lp)):
        pts = []
        for j in range(len(lp)):
            (x1, y1), (x2, y2) = lp[i][0], lp[i][1]
            (x3, y3), (x4, y4) = lp[j][0], lp[j][1]
            det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if det:
                ix = int(((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det)
                iy = int(((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det)
                if 0 <= ix <= 1920 and 0 <= iy <= 1080 and 1 < lp[i][2] < 3:
                    pts.append((ix, iy))
        if pts:
            pts.sort(key=lambda p: p[0]); rows.append(pts)
    rows.sort(key=lambda p: p[0][0])
    return rows

def _squares(rows):
    casas = []
    for i in range(len(rows) - 1):
        for j in range(len(rows[i]) - 1):
            casas.append([rows[i][j], rows[i][j + 1], rows[i + 1][j], rows[i + 1][j + 1]])
    return casas


def _preprocess(bgr):
    rgb = cv.cvtColor(cv.resize(bgr, (224, 224)), cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - _MEAN) / _STD).transpose(2, 0, 1)[None]


def _grid_to_fen(grid) -> str:
    out = []
    for row in grid:
        s, e = "", 0
        for cell in row:
            if cell == ".":
                e += 1
            else:
                if e:
                    s += str(e); e = 0
                s += cell
        if e:
            s += str(e)
        out.append(s)
    return "/".join(out)


def image_to_fen(bgr, session) -> str | None:
    """Full board state (piece-placement FEN) from a board image, or None if the
    64-square grid couldn't be detected."""
    im = _crop_image(bgr)
    lines = cv.HoughLines(cv.Canny(im, 100, 150, apertureSize=3), 1, np.pi / 180, 160)
    if lines is None:
        return None
    rows = _intersections(_filter(_group(lines)))
    if len(rows) >= 3:  # drop spurious 3rd-from-bottom horizontal line
        by_y = sorted(rows, key=lambda r: np.mean([p[1] for p in r]), reverse=True)
        rows = [r for r in rows if r is not by_y[2]]
    casas = _squares(rows)
    if len(casas) != 64:
        return None

    syms = []
    for p in casas:
        xs = [q[0] for q in p]; ys = [q[1] for q in p]
        crop = im[max(0, min(ys) - 90):max(ys), min(xs):max(xs)]
        if crop.size == 0:
            syms.append("."); continue
        out = session.run(None, {"input": _preprocess(crop)})[0].flatten()
        sm = np.exp(out) / np.exp(out).sum()
        edge = cv.Canny(cv.cvtColor(crop, cv.COLOR_BGR2GRAY), 100, 150).mean() / 255
        occupied = sm.max() >= 0.8 or edge >= 0.03      # confidence + edge density
        syms.append(IDX2SYM[int(sm.argmax())] if occupied else ".")

    grid = np.flipud(np.array(syms, object).reshape(8, 8))  # flipUD orientation
    return _grid_to_fen(grid)
