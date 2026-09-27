"""
model.py
--------
Defines the CNN architecture used to classify facial expressions into
7 emotion classes (angry, disgust, fear, happy, neutral, sad, surprise).
"""

from tensorflow.keras import layers, models, regularizers


def build_cnn(input_shape=(48, 48, 1), num_classes=7):
    """A compact VGG-style CNN, well suited to small 48x48 grayscale faces."""
    reg = regularizers.l2(1e-4)

    model = models.Sequential(name="Emojify_CNN")
    model.add(layers.Input(shape=input_shape))

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(0.25))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(0.25))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(0.25))

    # Block 4
    model.add(layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(0.25))

    # Classifier head
    model.add(layers.Flatten())
    model.add(layers.Dense(256, kernel_regularizer=reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model


if __name__ == "__main__":
    m = build_cnn()
    m.summary()
