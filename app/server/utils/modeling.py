"""Keras model builders shared by training and prediction."""
from tensorflow import keras


@keras.utils.register_keras_serializable(package="Fashion")
class ModelMetadata(keras.layers.Layer):
    """Store label order, preprocessing and calibration inside the .keras file."""
    def __init__(self, metadata=None, **kwargs):
        super().__init__(**kwargs)
        self.metadata = dict(metadata or {})

    def call(self, inputs):
        return inputs

    def get_config(self):
        return {**super().get_config(), "metadata": dict(self.metadata)}


def FashionMLP(num_classes, dropout=0.2, image_size=(96, 128), deep=False):
    layers = [keras.Input(shape=(image_size[1], image_size[0], 3)),
              keras.layers.RandomFlip("horizontal"), keras.layers.Flatten()]
    for width in ((256, 128, 64) if deep else (256,)):
        layers.extend([keras.layers.Dense(width, activation="relu"), keras.layers.Dropout(dropout)])
    layers.extend([keras.layers.Dense(num_classes), ModelMetadata(name="metadata")])
    return keras.Sequential(layers, name="deeper_mlp" if deep else "shallow_mlp")


def _cnn(num_classes, channels, hidden, dropout, image_size):
    layers = [keras.Input(shape=(image_size[1], image_size[0], 3)),
              keras.layers.RandomFlip("horizontal")]
    for width in channels:
        layers.extend([keras.layers.Conv2D(width, 3, padding="same"),
                       keras.layers.BatchNormalization(), keras.layers.Activation("relu"),
                       keras.layers.MaxPooling2D(2)])
    # Fixed input dimensions permit ordinary Keras pooling to a 2 x 2 grid.
    height, width = image_size[1] // (2 ** len(channels)), image_size[0] // (2 ** len(channels))
    if height % 2 or width % 2 or min(height, width) < 2:
        raise ValueError("CNN input dimensions must support a 2 x 2 pooled grid")
    layers.extend([keras.layers.AveragePooling2D((height // 2, width // 2)),
                   keras.layers.Flatten(), keras.layers.Dense(hidden, activation="relu"),
                   keras.layers.Dropout(dropout), keras.layers.Dense(num_classes),
                   ModelMetadata(name="metadata")])
    return keras.Sequential(layers, name="tuned_cnn" if len(channels) == 4 else "simple_cnn")


def SimpleCNN(num_classes, dropout=0.2, image_size=(96, 128)):
    return _cnn(num_classes, (32, 64, 128), 128, dropout, image_size)


def TunedCNN(num_classes, dropout=0.2, image_size=(96, 128)):
    return _cnn(num_classes, (32, 64, 128, 256), 256, dropout, image_size)
