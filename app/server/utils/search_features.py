"""Keras hidden-layer embeddings shared by retrieval evaluation and the API."""
import hashlib
import numpy as np
from tensorflow import keras
from app.server.utils.artifacts import load_checkpoint
from app.server.utils.image_preprocessor import image_batch


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class NeuralImageEncoder:
    def __init__(self, path):
        checkpoint = load_checkpoint(path)
        model = checkpoint["model"]
        # All project classifiers end with hidden Dense, Dropout, logits, metadata.
        self.encoder = keras.Model(model.inputs, model.layers[-3].output)
        self.metadata = checkpoint

    def encode(self, images):
        inputs = np.concatenate([image_batch(image, self.metadata["image_size"],
                    self.metadata["mean"], self.metadata["std"]) for image in images])
        values = np.asarray(self.encoder(inputs, training=False), dtype=np.float32)
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
