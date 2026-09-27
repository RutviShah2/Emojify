"""
webcam_app.py
-------------
Real-time webcam demo: detects your face, classifies the expression every
frame, and displays the live feed next to the matching emoji.

Run locally (requires a webcam):
    python -m src.webcam_app

Press 'q' to quit.
"""

import cv2
import numpy as np

from . import config as cfg
from .predict import EmojifyPredictor


def overlay_emoji(frame, emoji_bgra, x, y, size=120):
    emoji_resized = cv2.resize(emoji_bgra, (size, size))
    alpha = emoji_resized[:, :, 3] / 255.0
    for c in range(3):
        y1, y2 = y, y + size
        x1, x2 = x, x + size
        if y2 > frame.shape[0] or x2 > frame.shape[1]:
            continue
        frame[y1:y2, x1:x2, c] = (
            alpha * emoji_resized[:, :, c] + (1 - alpha) * frame[y1:y2, x1:x2, c]
        )
    return frame


def main():
    predictor = EmojifyPredictor()

    emojis_bgra = {}
    for label in cfg.EMOTIONS:
        img = cv2.imread(predictor.emoji_path(label), cv2.IMREAD_UNCHANGED)
        emojis_bgra[label] = img

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. This script must be run on a machine with a camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray, faces = predictor.detect_faces(frame)

        for (x, y, w, h) in faces:
            face_crop = gray[y:y + h, x:x + w]
            label, conf, _ = predictor.predict_emotion(face_crop)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(
                frame, f"{label} ({conf*100:.0f}%)", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2
            )

            emoji_img = emojis_bgra.get(label)
            if emoji_img is not None:
                frame = overlay_emoji(frame, emoji_img, x + w + 10, y, size=min(150, h))

        cv2.imshow("Emojify - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
