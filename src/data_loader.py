"""
data_loader.py
---------------
Builds Keras data generators for the FER2013-style dataset laid out as:

    data/train/<emotion>/*.jpg
    data/test/<emotion>/*.jpg

Applies light augmentation on the training set (rotation, zoom, shifts,
horizontal flip) to reduce overfitting on this relatively small dataset.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from . import config as cfg


def get_generators(batch_size=cfg.BATCH_SIZE, img_size=cfg.IMG_SIZE,
                    validation_split=0.1):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=12,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=validation_split,
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    common = dict(
        target_size=(img_size, img_size),
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=batch_size,
        seed=cfg.SEED,
    )

    train_gen = train_datagen.flow_from_directory(
        cfg.TRAIN_DIR, subset="training", shuffle=True, **common
    )
    val_gen = train_datagen.flow_from_directory(
        cfg.TRAIN_DIR, subset="validation", shuffle=False, **common
    )
    test_gen = test_datagen.flow_from_directory(
        cfg.TEST_DIR, shuffle=False, **common
    )

    return train_gen, val_gen, test_gen


if __name__ == "__main__":
    tr, va, te = get_generators()
    print("class_indices:", tr.class_indices)
    print("train samples:", tr.samples, "val samples:", va.samples, "test samples:", te.samples)
