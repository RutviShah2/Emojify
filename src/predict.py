"""
predict.py
----------
Inference utilities:
  - load the trained model + label map
  - detect a face in an arbitrary image (Haar cascade)
  - preprocess it to 48x48 grayscale
  - predict the emotion
  - fetch/compose the matching emoji

Usage (single image -> emotion + emoji composite):
    python -m src.predict --image path/to/photo.jpg
"""

import argparse
import json
import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from . import config as cfg

_FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


class EmojifyPredictor:
    def __init__(self, model_path=cfg.MODEL_PATH, labels_path=cfg.LABELS_PATH):
        self.model = load_model(model_path)
        with open(labels_path) as f:
            idx_to_class = json.load(f)
        # JSON keys are strings; convert back to int keys.
        self.idx_to_class = {int(k): v for k, v in idx_to_class.items()}
        self.face_detector = cv2.CascadeClassifier(_FACE_CASCADE_PATH)

    def preprocess_face(self, face_gray_img):
        """face_gray_img: 2D numpy array (grayscale crop) -> (1, 48, 48, 1) float32."""
        face = cv2.resize(face_gray_img, (cfg.IMG_SIZE, cfg.IMG_SIZE))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=(0, -1))
        return face

    def predict_emotion(self, face_gray_img):
        """Returns (label:str, confidence:float, all_probs:dict)."""
        x = self.preprocess_face(face_gray_img)
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = self.idx_to_class[idx]
        all_probs = {self.idx_to_class[i]: float(p) for i, p in enumerate(probs)}
        return label, float(probs[idx]), all_probs

    def detect_faces(self, bgr_img):
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )
        return gray, faces

    def predict_image(self, image_path):
        """Detect the largest face in an image file and predict its emotion.
        If no face is detected, fall back to treating the whole image as the face
        (useful for pre-cropped FER2013-style 48x48 images)."""
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(image_path)

        gray, faces = self.detect_faces(bgr)
        if len(faces) == 0:
            face_crop = gray
        else:
            # take the largest detected face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_crop = gray[y:y + h, x:x + w]

        label, conf, all_probs = self.predict_emotion(face_crop)
        return label, conf, all_probs

    def emoji_path(self, label):
        return os.path.join(cfg.EMOJI_DIR, f"{label}.png")

    def make_side_by_side(self, image_path, out_path):
        """Create a composite image: original photo | predicted emoji, saved to out_path."""
        from PIL import Image, ImageDraw, ImageFont

        label, conf, _ = self.predict_image(image_path)

        photo = Image.open(image_path).convert("RGB")
        emoji = Image.open(self.emoji_path(label)).convert("RGBA")

        target_h = 300
        w = int(photo.width * target_h / photo.height)
        photo = photo.resize((w, target_h))
        emoji = emoji.resize((target_h, target_h))

        canvas = Image.new("RGB", (photo.width + emoji.width + 20, target_h + 40), "white")
        canvas.paste(photo, (0, 20))
        canvas.paste(emoji, (photo.width + 20, 20), emoji)

        draw = ImageDraw.Draw(canvas)
        text = f"{label.upper()}  ({conf * 100:.1f}%)"
        draw.text((10, 2), text, fill="black")

        canvas.save(out_path)
        return label, conf, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to an input image")
    parser.add_argument("--out", default=os.path.join(cfg.OUTPUT_DIR, "prediction.png"))
    args = parser.parse_args()

    predictor = EmojifyPredictor()
    label, conf, out_path = predictor.make_side_by_side(args.image, args.out)
    print(f"Predicted emotion: {label} ({conf * 100:.2f}% confidence)")
    print("Saved composite to:", out_path)
