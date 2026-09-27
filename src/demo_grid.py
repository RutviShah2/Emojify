"""
demo_grid.py
------------
Builds a single demo figure: for each of the 7 emotion classes, picks a
sample photo from the test set, runs it through the trained model, and
shows [photo -> predicted emoji] side by side. A nice one-shot visual proof
that the end-to-end pipeline (image -> CNN -> emotion -> emoji) works.

Usage:
    python -m src.demo_grid
"""

import os
import random

from PIL import Image, ImageDraw

from . import config as cfg
from .predict import EmojifyPredictor


def main(n_per_class=1, seed=7):
    random.seed(seed)
    predictor = EmojifyPredictor()

    tile_h = 160
    rows = []
    for label in cfg.EMOTIONS:
        class_dir = os.path.join(cfg.TEST_DIR, label)
        files = os.listdir(class_dir)
        chosen = random.sample(files, min(n_per_class, len(files)))
        for fname in chosen:
            img_path = os.path.join(class_dir, fname)
            pred_label, conf, _ = predictor.predict_image(img_path)

            photo = Image.open(img_path).convert("RGB").resize((tile_h, tile_h))
            emoji = Image.open(predictor.emoji_path(pred_label)).convert("RGBA").resize((tile_h, tile_h))

            row = Image.new("RGB", (tile_h * 2 + 260, tile_h + 10), "white")
            row.paste(photo, (0, 5))
            row.paste(emoji, (tile_h + 10, 5), emoji)

            d = ImageDraw.Draw(row)
            correct = "✓" if pred_label == label else "✗"
            d.text(
                (tile_h * 2 + 25, tile_h // 2 - 10),
                f"true: {label}\npred: {pred_label} ({conf*100:.0f}%) {correct}",
                fill="black",
            )
            rows.append(row)

    width = max(r.width for r in rows)
    total_h = sum(r.height for r in rows) + 10 * len(rows)
    sheet = Image.new("RGB", (width, total_h), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 10

    out_path = os.path.join(cfg.OUTPUT_DIR, "demo_grid.png")
    sheet.save(out_path)
    print("Saved", out_path)


if __name__ == "__main__":
    main()
