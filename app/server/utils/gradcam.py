"""Grad-CAM of class logits for the project's sequential CNN classifiers.

Method reference: https://keras.io/examples/vision/grad_cam/
"""
import base64
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from app.server.utils.classifier import temperature_scale
from app.server.utils.image_preprocessor import image_batch


def gradcam(classifier, image, label=None):
    model = classifier.model
    if not isinstance(model, tf.keras.Sequential):
        raise ValueError('Heatmaps currently support sequential CNN models only')
    conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    if not conv_layers:
        raise ValueError('This classifier has no convolutional layer for Grad-CAM')
    if label is not None and label not in classifier.labels:
        raise ValueError('Unknown class label for this classifier')
    conv = conv_layers[-1]
    inputs = tf.convert_to_tensor(image_batch(image, classifier.image_size, classifier.mean, classifier.std))
    # Explicit layer traversal supports saved Sequential models without rebuilding
    # their symbolic graph; inference mode disables augmentation and dropout.
    with tf.device(classifier.device):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            values = inputs
            for layer in model.layers:
                values = layer(values, training=False)
                if layer is conv:
                    activations = values
            index = classifier.labels.index(label) if label is not None else int(tf.argmax(values[0]))
            score = values[:, index]
        gradients = tape.gradient(score, activations)
    if gradients is None:
        raise ValueError('The class output is not connected to the convolutional layer')
    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.maximum(tf.reduce_sum(activations[0] * weights, axis=-1), 0).numpy()
    if not np.isfinite(heatmap).all():
        raise ValueError('The model produced non-finite heatmap values')
    maximum = float(heatmap.max())
    has_signal = maximum > 0
    if has_signal:
        heatmap /= maximum
    # Match the uploaded image's aspect ratio, with a bounded response size.
    display = image.copy()
    display.thumbnail((768, 768))
    heat = np.asarray(Image.fromarray(heatmap).resize(display.size, Image.Resampling.BILINEAR))
    rgba = np.zeros((*heat.shape, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = np.asarray(210 * (1 - heat), dtype=np.uint8)
    rgba[..., 3] = np.asarray(255 * heat, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgba).save(buffer, format='PNG')
    probabilities = temperature_scale(tf.nn.softmax(values, axis=-1).numpy(), classifier.temperature)[0]
    return {'target': classifier.target, 'label': classifier.labels[index],
            'confidence': float(probabilities[index]), 'layer': conv.name,
            'has_signal': has_signal, 'method': 'Grad-CAM',
            'heatmap': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')}
