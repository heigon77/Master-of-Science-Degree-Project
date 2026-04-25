# Chess Board Digitization & Human Move Prediction

> Master's Degree Project — Computer Vision + Deep Learning applied to Chess

## Overview

This project tackles two interconnected problems in computational chess:

1. **Automatic Chess Board Digitization** — Given a photograph of a physical chess board (rendered in 3D or captured in the real world), detect and classify all 64 squares to reconstruct the board state as a FEN string.
2. **Human Move Prediction** — Given a board state, predict the most likely move a human player would make, combining deep learning models with Stockfish engine analysis.

---

## Project Structure

```
Project/
├── Notebooks/
│   ├── Preprocess.ipynb                              # Image preprocessing pipeline (Canny, Hough, square segmentation)
│   ├── Digitization_Chess_Board_Pre_Train_With_3D.ipynb   # Board digitization pre-training on synthetic 3D data
│   ├── Digitization_Chess_Board_Train_With_Real_Data.ipynb # Fine-tuning on real board photographs
│   └── Predict_Human_Move_Train.ipynb                # Human move prediction (piece + square classifiers)
├── Models/
│   ├── imageClassifier.pth          # MobileNetV2 — piece classifier (pre-trained on 3D data)
│   ├── imageClassifierReal.pth      # MobileNetV2 — piece classifier (fine-tuned on real images)
│   ├── pieceClassifier.pth          # CNN — which piece type is likely to move next
│   ├── squarePawnClassifier.pth     # CNN — destination square for Pawns
│   ├── squareKnightClassifier.pth   # CNN — destination square for Knights
│   ├── squareBishopClassifier.pth   # CNN — destination square for Bishops
│   ├── squareRookClassifier.pth     # CNN — destination square for Rooks
│   ├── squareQueenClassifier.pth    # CNN — destination square for Queens
│   ├── squareKingClassifier.pth     # CNN — destination square for Kings
│   └── autoencoder.pth              # Autoencoder for board state representation
├── Datasets/                        # ⚠️ Not included (see Datasets section below)
│   ├── Images/
│   │   ├── 3d_images/               # Synthetic 3D-rendered chess board images
│   │   ├── 3d_images_squares/       # Cropped square images (output of Preprocess.ipynb)
│   │   ├── 3d_images_label/         # FEN labels (img_fen.csv)
│   │   └── processed_images/        # Preprocessed tensors (.pt, .npy)
│   └── Moves/
│       ├── train/                   # Board state arrays for move prediction training
│       │   ├── pieces/              # All-piece classifier data
│       │   ├── pawn/ knight/ bishop/ rook/ queen/ king/
│       └── test/                    # Test partitions (first_part, second_part, third_part)
└── stockfish/
    └── stockfish-ubuntu-x86-64-avx2  # Stockfish engine binary (Linux x86-64 AVX2)
```

---

## Pipeline

### 1 — Image Preprocessing (`Preprocess.ipynb`)

Processes raw 3D-rendered chess board images into labeled square crops:

- Loads images paired with their FEN strings from a CSV
- Crops and centers the board region
- Applies **Canny edge detection** and **Hough line transform** to locate the grid
- Groups and filters lines; computes intersections to build the 64-square grid
- Crops each square and saves it alongside its piece label (using the `python-chess` library to parse FEN)

Output: individual square images + a CSV mapping `image_square → piece symbol`

### 2 — Board Digitization (3D Pre-training)

`Digitization_Chess_Board_Pre_Train_With_3D.ipynb`

Trains a **MobileNetV2**-based classifier on synthetic 3D images to recognize the 12 piece classes (K, Q, R, B, N, P for white and black) plus empty square.

- Transfer learning from ImageNet weights
- Custom classification head: `AdaptiveAvgPool2d → Linear(1280, 512) → ReLU → Linear(512, 12)`
- Weighted random sampling to compensate for class imbalance
- Adam optimizer, CrossEntropyLoss, StepLR scheduler
- Saves model as `imageClassifier.pth`

### 3 — Board Digitization (Fine-tuning on Real Images)

`Digitization_Chess_Board_Train_With_Real_Data.ipynb`

Fine-tunes the pre-trained model on real board photographs to bridge the domain gap between synthetic and real images.

- Loads `imageClassifier.pth` and resumes training
- 50/20/30 train/val/test split with shuffling
- Handles class imbalance for Kings (class 0 and 6)
- Saves as `imageClassifierReal.pth`

### 4 — Human Move Prediction (`Predict_Human_Move_Train.ipynb`)

Predicts the most likely human move from a given board position using a two-stage CNN approach:

**Stage 1 — Piece Classifier (`PieceClassifier`)**  
A CNN with 12-channel input (one channel per piece type in the 8×8 board representation) that predicts which piece type will be moved.

**Stage 2 — Square Classifiers (one per piece type)**  
Six independent `SquareClassifier` CNNs (Pawn, Knight, Bishop, Rook, Queen, King), each trained to predict the destination square given the board state.

**Inference (`test_model`)**  
Combines three strategies evaluated against real human games:
- **CNN only** — pure model prediction
- **Stockfish only** — top-N engine moves ranked by score
- **Hybrid** — CNN move probabilities combined with Stockfish evaluation

Performance is measured as top-1, top-2, and top-3 accuracy against the actual human move across three independent test partitions.

---

## Models Summary

| File | Architecture | Task | Classes |
|---|---|---|---|
| `imageClassifier.pth` | MobileNetV2 | Square piece classification (3D) | 12 |
| `imageClassifierReal.pth` | MobileNetV2 | Square piece classification (real) | 12 |
| `pieceClassifier.pth` | Custom CNN (12-ch input) | Piece type to move | 6 |
| `squarePawnClassifier.pth` | Custom CNN (12-ch input) | Destination square — Pawn | 64 |
| `squareKnightClassifier.pth` | Custom CNN (12-ch input) | Destination square — Knight | 64 |
| `squareBishopClassifier.pth` | Custom CNN (12-ch input) | Destination square — Bishop | 64 |
| `squareRookClassifier.pth` | Custom CNN (12-ch input) | Destination square — Rook | 64 |
| `squareQueenClassifier.pth` | Custom CNN (12-ch input) | Destination square — Queen | 64 |
| `squareKingClassifier.pth` | Custom CNN (12-ch input) | Destination square — King | 64 |
| `autoencoder.pth` | Autoencoder | Board state representation | — |

---

## Datasets

The `Datasets/` directory is **not included** in this repository due to file size constraints. The data is publicly available on Kaggle:

> 📦 **[chess-moves-from-lichess-with-3d-boards](https://www.kaggle.com/datasets/heigonsoldera/chess-moves-from-lichess-with-3d-boards)**

The pipeline expects the following data:

| Path | Description |
|---|---|
| `Datasets/Images/3d_images/` | Synthetic 3D-rendered board images (`.png`) |
| `Datasets/Images/3d_images_label/img_fen.csv` | Image names paired with FEN strings |
| `Datasets/Images/3d_images_squares/` | Output directory for cropped squares |
| `Datasets/Images/processed_images/imagens_casa_tensor*.pt` | Pre-processed image tensors |
| `Datasets/Images/processed_images/pecas_casa_tensor*.npy` | Corresponding piece labels |
| `Datasets/Images/processed_images/imagens_casa_tensor_real.pt` | Real image tensors |
| `Datasets/Moves/train/{pieces,pawn,knight,bishop,rook,queen,king}/` | Board state arrays for move prediction |
| `Datasets/Moves/test/first_part*.{npy,csv}` … `third_part*.{npy,csv}` | Test partitions |

---

## Dependencies

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

The Stockfish binary for Linux x86-64 (AVX2) is included in `stockfish/`. For other platforms, download from [stockfishchess.org/download](https://stockfishchess.org/download/).

---

## Usage

Run the notebooks in order:

```
1. Preprocess.ipynb
2. Digitization_Chess_Board_Pre_Train_With_3D.ipynb
3. Digitization_Chess_Board_Train_With_Real_Data.ipynb
4. Predict_Human_Move_Train.ipynb
```

Pre-trained model weights are available in `Models/` and can be loaded directly to skip training.

---

## License

This project is distributed under the GNU General Public License v3. See [LICENSE](LICENSE) for details.

The Stockfish engine is also distributed under the GPL v3. See `stockfish/Copying.txt`.
