# 😄 Emojify — Facial Expression Recognition → Emoji Mapping

An end-to-end deep learning project that recognizes human facial expressions
from images using a Convolutional Neural Network (CNN), and maps the
predicted emotion to a matching emoji/avatar — with both a webcam demo and
a browser (Streamlit) app.

## Overview

1. **Dataset**: [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) —
   48×48 grayscale face crops labeled with 7 emotions:
   `angry, disgust, fear, happy, neutral, sad, surprise`.
   35,887 images total (28,709 train / 7,178 test).
2. **Model**: a compact VGG-style CNN (4 conv blocks + dense head, ~1.18M
   parameters) trained from scratch with data augmentation and class
   weighting (FER2013 is imbalanced — `disgust` has ~9x fewer examples than
   `happy`).
3. **Emoji mapping**: each of the 7 predicted classes maps to a matching
   procedurally-generated emoji avatar (`emojis/*.png`) — no external/
   copyrighted emoji assets required.
4. **Apps**:
   - `src/predict.py` — single-image CLI inference → photo + emoji composite.
   - `src/webcam_app.py` — real-time OpenCV webcam demo.
   - `app.py` — Streamlit web app (upload a photo or use your browser camera).

## Project structure

```
Emojify/
├── data/                    # FER2013 dataset (train/ and test/, 7 class folders each)
├── emojis/                  # 7 generated emoji PNGs (angry.png, happy.png, ...)
├── models/                  # saved model (emojify_cnn.keras), labels.json, history.json
├── outputs/                 # training logs, metrics, confusion matrix, demo images
├── src/
│   ├── config.py            # paths & hyperparameters
│   ├── generate_emojis.py   # draws the 7 emoji avatars with PIL
│   ├── data_loader.py       # Keras ImageDataGenerator pipeline (+ augmentation)
│   ├── model.py             # CNN architecture
│   ├── train.py             # training loop (callbacks, class weights, checkpoints)
│   ├── evaluate.py          # test-set evaluation, confusion matrix, curves
│   ├── predict.py           # inference: face -> emotion -> emoji
│   ├── demo_grid.py         # builds a one-shot demo figure (7 sample predictions)
│   └── webcam_app.py        # live webcam demo
├── app.py                   # Streamlit GUI
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## 1. Generate the emoji avatars

```bash
python -m src.generate_emojis
```
Draws 7 simple, expressive emoji faces (one per emotion class) into `emojis/`.

## 2. Train the CNN

```bash
python -m src.train                 # full training run
python -m src.train --epochs 5      # quick smoke test
```

What it does:
- Loads `data/train` (90/10 train/validation split) and `data/test` via
  `ImageDataGenerator`, with rotation/shift/zoom/flip augmentation on the
  training set.
- Builds the CNN (`src/model.py`): 4 conv blocks (32→64→128→256 filters)
  with BatchNorm + Dropout, then a 256-unit dense head, softmax over 7
  classes.
- Uses **class weighting** to counter FER2013's imbalance (e.g. `disgust`
  is heavily under-represented).
- Callbacks: `ModelCheckpoint` (saves the best model by val_accuracy),
  `EarlyStopping`, `ReduceLROnPlateau`, `CSVLogger`.
- Saves:
  - `models/emojify_cnn.keras` — the trained model
  - `models/labels.json` — index→class-name mapping (guarantees inference
    always agrees with the training label order)
  - `models/history.json` — training curves
  - `outputs/test_metrics.json` — final test accuracy/loss

## 3. Evaluate

```bash
python -m src.evaluate
```
Produces `outputs/classification_report.txt`, `outputs/confusion_matrix.png`,
and `outputs/training_curves.png`.

## 4. Run inference

**Single image → emotion + emoji composite:**
```bash
python -m src.predict --image path/to/photo.jpg
```

**One-shot demo grid (7 sample test images, one per class):**
```bash
python -m src.demo_grid
```

**Streamlit web app** (upload a photo or use your camera):
```bash
streamlit run app.py
```

**Real-time webcam demo** (run on a machine with a physical webcam):
```bash
python -m src.webcam_app
```
Detects your face live (Haar cascade), classifies the expression every
frame, and overlays the matching emoji next to your face feed. Press `q`
to quit.

## How the emotion → emoji mapping works

`src/predict.py`'s `EmojifyPredictor`:
1. Detects the largest face in the image with OpenCV's Haar cascade
   (falls back to using the whole image if no face is found — useful for
   pre-cropped 48×48 FER2013-style images).
2. Resizes the face crop to 48×48 grayscale and normalizes it to `[0, 1]`.
3. Feeds it through the trained CNN → softmax probabilities over the 7
   classes.
4. Looks up `emojis/<predicted_label>.png` and composes it next to the
   original photo.

The label order used at inference is always read from `models/labels.json`
(saved during training), so it can never drift out of sync with how the
model was trained even if folder-scan ordering changes.

## Notes & possible extensions

- Model was trained from scratch (no transfer learning) to keep the
  pipeline simple and fast; swapping in a pretrained backbone (e.g. a small
  MobileNet on grayscale-to-RGB-replicated inputs) is a natural next step
  for higher accuracy.
- FER2013 itself is a noisy, low-resolution dataset (even humans disagree
  on FER2013 labels roughly 30-35% of the time), so state-of-the-art models
  on this exact dataset top out around 70-75% test accuracy.
- The emoji avatars are procedurally drawn in `src/generate_emojis.py`
  (simple PIL shapes) rather than downloaded, so the project has no
  external asset/licensing dependencies — swap in your own emoji/avatar
  image set by just replacing the PNGs in `emojis/` (keep the same
  filenames: `angry.png`, `disgust.png`, `fear.png`, `happy.png`,
  `neutral.png`, `sad.png`, `surprise.png`).
