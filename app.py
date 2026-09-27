"""
app.py
------
Streamlit front-end for Emojify.

Run:
    streamlit run app.py

Lets the user either upload a photo or take one with their browser camera,
runs face detection + the trained CNN, and displays the matching emoji
side-by-side with a confidence bar chart over all 7 emotions.
"""

import json
import os

import numpy as np
import streamlit as st
from PIL import Image

from src import config as cfg
from src.predict import EmojifyPredictor

st.set_page_config(page_title="Emojify", page_icon="😄", layout="centered")

st.title("😄 Emojify — Facial Expression to Emoji")
st.write(
    "Upload a photo (or take one with your camera) and the CNN will detect "
    "your facial expression and map it to a matching emoji."
)


@st.cache_resource
def load_predictor():
    return EmojifyPredictor()


if not os.path.exists(cfg.MODEL_PATH):
    st.error(
        "No trained model found yet at `models/emojify_cnn.keras`. "
        "Run `python -m src.train` first."
    )
    st.stop()

predictor = load_predictor()

tab1, tab2 = st.tabs(["📁 Upload a photo", "📷 Use camera"])
image_file = None
with tab1:
    image_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
with tab2:
    cam_file = st.camera_input("Take a picture")
    if cam_file is not None:
        image_file = cam_file

if image_file is not None:
    pil_img = Image.open(image_file).convert("RGB")
    tmp_path = os.path.join(cfg.OUTPUT_DIR, "_st_upload.jpg")
    pil_img.save(tmp_path)

    label, conf, all_probs = predictor.predict_image(tmp_path)

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="Your photo", use_container_width=True)
    with col2:
        emoji_img = Image.open(predictor.emoji_path(label))
        st.image(emoji_img, caption=f"{label.capitalize()} ({conf*100:.1f}%)", use_container_width=True)

    st.subheader("Confidence across all emotions")
    st.bar_chart(all_probs)
else:
    st.info("Upload a photo or take one with your camera to get started.")

st.divider()
st.caption(
    "Model: custom CNN trained from scratch on the FER2013 dataset "
    "(7 classes: angry, disgust, fear, happy, neutral, sad, surprise)."
)
