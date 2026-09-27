"""
train.py
--------
End-to-end training script for the Emojify CNN.

Usage:
    python -m src.train                # full training run (config.EPOCHS)
    python -m src.train --epochs 5      # override epoch count (e.g. for a quick smoke test)
"""

import argparse
import json
import os

import numpy as np
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
)
from sklearn.utils.class_weight import compute_class_weight

from . import config as cfg
from .data_loader import get_generators
from .model import build_cnn


def main(epochs=cfg.EPOCHS, batch_size=cfg.BATCH_SIZE):
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    train_gen, val_gen, test_gen = get_generators(batch_size=batch_size)

    # Save the label mapping so inference code always agrees with training.
    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    with open(cfg.LABELS_PATH, "w") as f:
        json.dump(idx_to_class, f, indent=2)

    # FER2013 is imbalanced (e.g. very few "disgust" examples) -> class weights.
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_gen.classes),
        y=train_gen.classes,
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print("Class weights:", class_weight_dict)

    model = build_cnn(input_shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, 1), num_classes=len(cfg.EMOTIONS))
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        ModelCheckpoint(cfg.MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        CSVLogger(os.path.join(cfg.OUTPUT_DIR, "training_log.csv")),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks,
    )

    with open(cfg.HISTORY_PATH, "w") as f:
        json.dump(history.history, f, indent=2)

    # Final evaluation on the held-out test set.
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"\nFinal TEST accuracy: {test_acc:.4f}  |  TEST loss: {test_loss:.4f}")

    with open(os.path.join(cfg.OUTPUT_DIR, "test_metrics.json"), "w") as f:
        json.dump({"test_loss": float(test_loss), "test_accuracy": float(test_acc)}, f, indent=2)

    # Ensure the best model is saved (ModelCheckpoint already does this, but
    # in case restore_best_weights swapped in different weights, save again).
    model.save(cfg.MODEL_PATH)
    print("Model saved to", cfg.MODEL_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size)
