"""Shared, ordinary PyTorch training steps for the Tasks 1-3 notebooks.

The notebooks choose and compare models. This file only handles image loading,
one training loop, and probability prediction; it never opens the test split.
"""

from copy import deepcopy
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from app.server.utils.handcrafted import handcrafted_feature
from scripts.evaluation import expected_calibration_error, supported_macro_f1
from scripts.preprocessing import IMAGE_SIZE, SEED


class FashionDataset(Dataset):
    def __init__(self, frame, target, labels, normalisation, training=False):
        self.paths = frame.image_path.tolist()
        self.targets = [labels.index(label) for label in frame[target]]
        steps = [transforms.Resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))]
        if training:
            steps.append(transforms.RandomHorizontalFlip())
        steps.extend([
            transforms.ToTensor(),
            transforms.Normalize(normalisation['mean'], normalisation['std']),
        ])
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert('RGB'))
        return tensor, self.targets[index]


def make_loader(frame, target, labels, normalisation, training=False):
    dataset = FashionDataset(frame, target, labels, normalisation, training)
    return DataLoader(
        dataset, batch_size=64, shuffle=training, num_workers=0,
        generator=torch.Generator().manual_seed(SEED),
    )


def extract_features(frame):
    """No fitting here: one fixed HOG + HSV vector per image."""
    features = []
    for path in frame.image_path:
        with Image.open(path) as image:
            features.append(handcrafted_feature(image))
    return np.vstack(features)


def comparison_row(method, model_type, truth, probabilities, parameters, epochs=0, eligible=True):
    """Use the same validation metrics for classical and neural candidates."""
    return {
        'method': method,
        'model_type': model_type,
        'validation_accuracy': accuracy_score(truth, probabilities.argmax(1)),
        'validation_macro_f1': supported_macro_f1(truth, probabilities.argmax(1)),
        'validation_ece': expected_calibration_error(truth, probabilities),
        'complexity_parameters': parameters,
        'epochs_run': epochs,
        'eligible_for_selection': eligible,
    }


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    truth, predictions = [], []
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(targets)
        truth.extend(targets.cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
    return {
        'loss': total_loss / len(loader.dataset),
        'accuracy': accuracy_score(truth, predictions),
        'macro_f1': supported_macro_f1(truth, predictions),
    }


def train_cnn(model, train_loader, validation_loader, device, max_epochs=30, patience=5):
    """Ordinary cross-entropy + Adam; restore the best validation macro-F1 epoch."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    best_f1 = -1.0
    best_state = None
    stale_epochs = 0
    for epoch in range(1, max_epochs + 1):
        started = time.perf_counter()
        train = run_epoch(model, train_loader, criterion, device, optimizer)
        validation = run_epoch(model, validation_loader, criterion, device)
        row = {'epoch': epoch, 'loss_mode': 'ordinary'}
        for name, value in train.items():
            row[f'train_{name}'] = value
        for name, value in validation.items():
            row[f'validation_{name}'] = value
        history.append(row)
        print(f"Epoch {epoch:02d}: train F1={train['macro_f1']:.4f}, "
              f"validation F1={validation['macro_f1']:.4f}, "
              f"{time.perf_counter() - started:.1f}s", flush=True)
        if validation['macro_f1'] > best_f1:
            best_f1 = validation['macro_f1']
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


@torch.inference_mode()
def predict_cnn(model, loader, device):
    model.to(device).eval()
    probabilities = []
    for images, _ in loader:
        probabilities.append(model(images.to(device)).softmax(1).cpu().numpy())
    return np.vstack(probabilities)
