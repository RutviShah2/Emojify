"""
resume_train.py
----------------
Trains for a small number of epochs starting from the existing checkpoint
(if any), saving after every epoch. Designed to be safely re-invoked
multiple times in short bursts (e.g. if the process gets interrupted).

Usage:
    python -m src.resume_train --epochs 2 --batch_size 256
"""

import argparse
import csv
import json
import os

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model

from . import config as cfg
from .data_loader import get_generators
from .model import build_cnn


class SaveEveryEpoch(Callback):
    """Always save after each epoch (regardless of val_accuracy) so progress
    is never lost if the process gets killed mid-run."""

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(cfg.MODEL_PATH)
        print(f"[checkpoint] saved model after epoch (logs={logs})")


def append_csv_log(logs, epoch_offset):
    path = os.path.join(cfg.OUTPUT_DIR, "training_log.csv")
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "accuracy", "loss", "val_accuracy", "val_loss"])
        writer.writerow([
            epoch_offset, logs.get("accuracy"), logs.get("loss"),
            logs.get("val_accuracy"), logs.get("val_loss"),
        ])


def get_next_epoch_index():
    path = os.path.join(cfg.OUTPUT_DIR, "training_log.csv")
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        rows = list(csv.reader(f))
    if len(rows) <= 1:
        return 0
    last_epoch = rows[-1][0]
    try:
        return int(float(last_epoch)) + 1
    except ValueError:
        return 0


def main(epochs=2, batch_size=256):
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    train_gen, val_gen, test_gen = get_generators(batch_size=batch_size)

    if not os.path.exists(cfg.LABELS_PATH):
        idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
        with open(cfg.LABELS_PATH, "w") as f:
            json.dump(idx_to_class, f, indent=2)

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(train_gen.classes), y=train_gen.classes,
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    if os.path.exists(cfg.MODEL_PATH):
        print("Resuming from existing checkpoint:", cfg.MODEL_PATH)
        model = load_model(cfg.MODEL_PATH)
    else:
        print("No checkpoint found, building a new model.")
        model = build_cnn(input_shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, 1), num_classes=len(cfg.EMOTIONS))
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    start_epoch = get_next_epoch_index()
    print(f"Starting at logical epoch {start_epoch}, training {epochs} more epoch(s)")

    for i in range(epochs):
        hist = model.fit(
            train_gen, validation_data=val_gen, epochs=1, verbose=1,
            class_weight=class_weight_dict, callbacks=[SaveEveryEpoch()],
        )
        logs = {k: v[0] for k, v in hist.history.items()}
        append_csv_log(logs, start_epoch + i)
        print(f"Finished logical epoch {start_epoch + i}: {logs}")

    print("Done with this burst. Model saved to", cfg.MODEL_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size)
