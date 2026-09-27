"""
generate_emojis.py
-------------------
Procedurally draws a simple emoji-style avatar for each of the 7 FER2013
emotion classes and saves them as PNGs in the `emojis/` folder.

We draw our own emoji art (instead of downloading copyrighted emoji sets)
so the whole project is self-contained and license-free.

Run:
    python src/generate_emojis.py
"""

import os
import math
from PIL import Image, ImageDraw

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "emojis")
os.makedirs(OUT_DIR, exist_ok=True)

SIZE = 300
YELLOW = (255, 205, 15, 255)
OUTLINE = (60, 45, 0, 255)
BLACK = (30, 20, 10, 255)


def base_face(fill=YELLOW):
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = 10
    d.ellipse([pad, pad, SIZE - pad, SIZE - pad], fill=fill, outline=OUTLINE, width=6)
    return img, d


def save(img, name):
    path = os.path.join(OUT_DIR, f"{name}.png")
    img.save(path)
    print("saved", path)


def eyes(d, style="normal"):
    lx, rx, y = 100, 200, 120
    r = 14
    if style == "normal":
        d.ellipse([lx - r, y - r, lx + r, y + r], fill=BLACK)
        d.ellipse([rx - r, y - r, rx + r, y + r], fill=BLACK)
    elif style == "angry":
        d.line([lx - 22, y - 22, lx + 18, y - 2], fill=BLACK, width=8)
        d.line([rx + 22, y - 22, rx - 18, y - 2], fill=BLACK, width=8)
        d.ellipse([lx - 10, y, lx + 10, y + 20], fill=BLACK)
        d.ellipse([rx - 10, y, rx + 10, y + 20], fill=BLACK)
    elif style == "sad":
        d.ellipse([lx - r, y - r, lx + r, y + r], fill=BLACK)
        d.ellipse([rx - r, y - r, rx + r, y + r], fill=BLACK)
        d.line([lx - 20, y - 18, lx + 16, y - 26], fill=BLACK, width=6)
        d.line([rx + 20, y - 18, rx - 16, y - 26], fill=BLACK, width=6)
    elif style == "fear":
        r2 = 20
        d.ellipse([lx - r2, y - r2, lx + r2, y + r2], outline=BLACK, width=6)
        d.ellipse([lx - 6, y - 6, lx + 6, y + 6], fill=BLACK)
        d.ellipse([rx - r2, y - r2, rx + r2, y + r2], outline=BLACK, width=6)
        d.ellipse([rx - 6, y - 6, rx + 6, y + 6], fill=BLACK)
    elif style == "surprise":
        r2 = 22
        d.ellipse([lx - r2, y - r2, lx + r2, y + r2], fill=BLACK)
        d.ellipse([rx - r2, y - r2, rx + r2, y + r2], fill=BLACK)
    elif style == "disgust":
        d.line([lx - 20, y - 5, lx + 20, y - 15], fill=BLACK, width=8)
        d.ellipse([rx - r, y - r, rx + r, y + r], fill=BLACK)
    elif style == "wink_happy":
        d.ellipse([lx - r, y - r, lx + r, y + r], fill=BLACK)
        d.arc([rx - r - 4, y - r - 4, rx + r + 4, y + r + 4], 200, 340, fill=BLACK, width=8)
    elif style == "closed_content":
        d.arc([lx - r - 4, y - r - 4, lx + r + 4, y + r + 4], 200, 340, fill=BLACK, width=8)
        d.arc([rx - r - 4, y - r - 4, rx + r + 4, y + r + 4], 200, 340, fill=BLACK, width=8)


def eyebrows(d, style=None):
    if style == "angry":
        d.line([80, 85, 130, 105], fill=BLACK, width=10)
        d.line([220, 85, 170, 105], fill=BLACK, width=10)
    elif style == "sad":
        d.line([80, 95, 130, 80], fill=BLACK, width=8)
        d.line([220, 95, 170, 80], fill=BLACK, width=8)
    elif style == "surprise":
        d.line([80, 75, 130, 65], fill=BLACK, width=7)
        d.line([220, 75, 170, 65], fill=BLACK, width=7)


def mouth(d, style="smile"):
    cx, y = 150, 200
    if style == "smile":
        d.arc([cx - 65, y - 45, cx + 65, y + 35], 20, 160, fill=BLACK, width=10)
    elif style == "big_smile":
        d.pieslice([cx - 65, y - 40, cx + 65, y + 45], 10, 170, fill=BLACK)
        d.pieslice([cx - 50, y - 25, cx + 50, y + 20], 10, 170, fill=YELLOW)
    elif style == "frown":
        d.arc([cx - 60, y + 5, cx + 60, y + 65], 200, 340, fill=BLACK, width=10)
    elif style == "flat":
        d.line([cx - 45, y + 10, cx + 45, y + 10], fill=BLACK, width=9)
    elif style == "open_oval":
        d.ellipse([cx - 30, y - 15, cx + 30, y + 35], fill=BLACK)
    elif style == "small_o":
        d.ellipse([cx - 16, y - 5, cx + 16, y + 27], fill=BLACK)
    elif style == "zigzag":
        pts = [(cx - 45, y + 5), (cx - 25, y + 25), (cx - 5, y + 5),
               (cx + 15, y + 25), (cx + 35, y + 5), (cx + 50, y + 15)]
        d.line(pts, fill=BLACK, width=8, joint="curve")


def draw_tears(d, n=2):
    for i, x in enumerate([80, 220][:n]):
        d.ellipse([x - 10, 230, x + 10, 265], fill=(90, 170, 255, 255))


def emotion_angry():
    img, d = base_face(fill=(255, 205, 15, 255))
    eyebrows(d, "angry")
    eyes(d, "angry")
    mouth(d, "frown")
    return img


def emotion_disgust():
    img, d = base_face(fill=(160, 205, 60, 255))
    eyes(d, "disgust")
    mouth(d, "zigzag")
    return img


def emotion_fear():
    img, d = base_face(fill=(255, 220, 90, 255))
    eyebrows(d, "sad")
    eyes(d, "fear")
    mouth(d, "small_o")
    return img


def emotion_happy():
    img, d = base_face(fill=(255, 205, 15, 255))
    eyes(d, "normal")
    mouth(d, "big_smile")
    return img


def emotion_neutral():
    img, d = base_face(fill=(255, 205, 15, 255))
    eyes(d, "normal")
    mouth(d, "flat")
    return img


def emotion_sad():
    img, d = base_face(fill=(255, 205, 15, 255))
    eyebrows(d, "sad")
    eyes(d, "sad")
    mouth(d, "frown")
    draw_tears(d, n=1)
    return img


def emotion_surprise():
    img, d = base_face(fill=(255, 205, 15, 255))
    eyebrows(d, "surprise")
    eyes(d, "surprise")
    mouth(d, "open_oval")
    return img


GENERATORS = {
    "angry": emotion_angry,
    "disgust": emotion_disgust,
    "fear": emotion_fear,
    "happy": emotion_happy,
    "neutral": emotion_neutral,
    "sad": emotion_sad,
    "surprise": emotion_surprise,
}


if __name__ == "__main__":
    for name, fn in GENERATORS.items():
        save(fn(), name)
    print("All 7 emoji avatars generated in", OUT_DIR)
