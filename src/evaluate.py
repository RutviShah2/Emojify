"""
evaluate.py
-----------
Loads the trained model, evaluates it on the FER2013 test set, and saves:
  - outputs/classification_report.txt
  - outputs/confusion_matrix.png
  - outputs/training_curves.png (if history.json exists)

Usage:
    python -m src.evaluate
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

from . import config as cfg
from .data_loader import get_generators


def plot_training_curves():
    if not os.path.exists(cfg.HISTORY_PATH):
        print("No history.json found, skipping training curves plot.")
        return
    with open(cfg.HISTORY_PATH) as f:
        hist = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(hist["accuracy"], label="train")
    axes[0].plot(hist["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(hist["loss"], label="train")
    axes[1].plot(hist["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(cfg.OUTPUT_DIR, "training_curves.png")
    plt.savefig(out_path, dpi=130)
    plt.close()
    print("Saved", out_path)


def main():
    _, _, test_gen = get_generators(batch_size=cfg.BATCH_SIZE)
    model = load_model(cfg.MODEL_PATH)

    steps = int(np.ceil(test_gen.samples / test_gen.batch_size))
    test_gen.reset()
    y_prob = model.predict(test_gen, steps=steps, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)[: test_gen.samples]
    y_true = test_gen.classes[: test_gen.samples]

    idx_to_class = {v: k for k, v in test_gen.class_indices.items()}
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    report = classification_report(y_true, y_pred, target_names=target_names, digits=3)
    print(report)
    with open(os.path.join(cfg.OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    cm_path = os.path.join(cfg.OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=130)
    plt.close()
    print("Saved", cm_path)

    plot_training_curves()


if __name__ == "__main__":
    main()
