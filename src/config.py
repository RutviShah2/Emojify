"""Central configuration for the Emojify project."""

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

EMOJI_DIR = os.path.join(ROOT_DIR, "emojis")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

MODEL_PATH = os.path.join(MODEL_DIR, "emojify_cnn.keras")
HISTORY_PATH = os.path.join(MODEL_DIR, "history.json")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

IMG_SIZE = 48          # FER2013 images are 48x48 grayscale
BATCH_SIZE = 64
EPOCHS = 40
SEED = 42

# Class order is fixed by the folder names (alphabetical, matches Keras
# ImageDataGenerator's default `flow_from_directory` ordering).
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
