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
from scripts.utils.evaluation import expected_calibration_error, supported_macro_f1
from scripts.preprocessing import IMAGE_SIZE, SEED


class FashionDataset(Dataset):
    def __init__(self, frame, target, labels, normalisation, training=False, lighting=False):
        self.paths = frame.image_path.tolist()
        self.targets = [labels.index(label) for label in frame[target]]
        steps = [transforms.Resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))]
        if training:
            steps.append(transforms.RandomHorizontalFlip())
            if lighting:
                steps.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
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


def make_loader(frame, target, labels, normalisation, training=False, lighting=False):
    dataset = FashionDataset(frame, target, labels, normalisation, training, lighting)
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


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
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
            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
        total_loss += loss.item() * len(targets)
        truth.extend(targets.cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
    return {
        'loss': total_loss / len(loader.dataset),
        'accuracy': accuracy_score(truth, predictions),
        'macro_f1': supported_macro_f1(truth, predictions),
    }


def train_cnn(model, train_loader, validation_loader, device, max_epochs=30, patience=5, schedule=False,
              learning_rate=0.001, class_weights=None):
    """Ordinary cross-entropy + Adam; restore the best validation macro-F1 epoch."""
    model = model.to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2,
    ) if schedule else None
    history = []
    best_f1 = -1.0
    best_state = None
    stale_epochs = 0
    for epoch in range(1, max_epochs + 1):
        started = time.perf_counter()
        train = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        validation = run_epoch(model, validation_loader, criterion, device)
        row = {'epoch': epoch, 'loss_mode': 'weighted' if class_weights is not None else 'ordinary',
               'lr': optimizer.param_groups[0]['lr']}
        for name, value in train.items():
            row[f'train_{name}'] = value
        for name, value in validation.items():
            row[f'validation_{name}'] = value
        history.append(row)
        if scheduler is not None:
            scheduler.step(validation['macro_f1'])
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


def fit_final_classifier(target, train_frame, validation_frame, normalisation, device):
    """Train the selected recipe from scratch, using validation only for tuning.

    Returns a checkpoint and history in memory. The notebook evaluates it on
    test and writes it to models/. No feature caches or checkpoint backups.
    """
    from PIL import ImageEnhance
    from scipy.optimize import minimize_scalar
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from app.server.utils.classifier import temperature_scale
    from app.server.utils.handcrafted import DEFAULT_FEATURE_CONFIG
    from app.server.utils.modeling import SimpleCNN
    from scripts.utils.evaluation import evaluate_saved_checkpoint
    from scripts.preprocessing import seed_everything

    if target not in {'articleType', 'season', 'gender', 'usage'}:
        raise ValueError(f'Unsupported target: {target}')
    seed_everything(SEED)
    torch.manual_seed(SEED)
    labels = sorted(train_frame[target].unique())
    selection_ids, rest_ids = next(GroupShuffleSplit(
        n_splits=1, train_size=0.5, random_state=SEED,
    ).split(validation_frame, groups=validation_frame.group_key))
    selection = validation_frame.iloc[selection_ids]
    rest = validation_frame.iloc[rest_ids]
    calibration_ids, policy_ids = next(GroupShuffleSplit(
        n_splits=1, train_size=0.5, random_state=SEED + 1,
    ).split(rest, groups=rest.group_key))
    calibration, policy = rest.iloc[calibration_ids], rest.iloc[policy_ids]
    checkpoint = {'target': target, 'labels': labels, 'temperature': 1.0,
                  'seed': SEED, 'experiment': 'selected_recipe_notebook_training',
                  'validation_scope': 'selection_view_only',
                  'selection_metric': 'fixed_recipe_from_prior_validation_experiments'}
    if target in {'articleType', 'usage'}:
        factors = (0.9, 1.0, 1.1) if target == 'articleType' else (1.0,)
        features = []
        for factor in factors:
            for path in train_frame.image_path:
                with Image.open(path) as source:
                    image = ImageEnhance.Brightness(source.convert('RGB')).enhance(factor)
                    features.append(handcrafted_feature(image, DEFAULT_FEATURE_CONFIG))
        features = np.vstack(features)
        counts = train_frame[target].value_counts()
        inverse_sqrt = 1 / np.sqrt(counts)
        weights = (inverse_sqrt / inverse_sqrt.mean()).to_dict()
        estimator = make_pipeline(StandardScaler(), LogisticRegression(
            C=0.1, class_weight=weights, max_iter=1500, random_state=SEED, solver='lbfgs',
        ))
        estimator.fit(features, np.tile(train_frame[target].to_numpy(), len(factors)),
                      logisticregression__sample_weight=np.full(len(features), 1 / len(factors)))
        checkpoint.update(model_type='hog_hsv_logistic_regression', estimator=estimator,
                          feature_config=dict(DEFAULT_FEATURE_CONFIG),
                          training_config={'C': 0.1, 'weights': 'sqrt', 'brightness_factors': list(factors)})
        history = None
    else:
        model, history = train_cnn(
            SimpleCNN(len(labels)),
            make_loader(train_frame, target, labels, normalisation, training=True),
            make_loader(selection, target, labels, normalisation),
            device, max_epochs=30, patience=7, schedule=target == 'gender',
        )
        checkpoint.update(model_type='simple_cnn', state_dict=model.cpu().state_dict(),
                          mean=normalisation['mean'], std=normalisation['std'],
                          image_size=list(IMAGE_SIZE), dropout=0.2,
                          training_config={'schedule': target == 'gender', 'max_epochs': 30, 'patience': 7})
    # Fit temperature only on calibration groups; retain it only if policy NLL
    # improves without worsening ECE. Policy groups also determine review flags.
    cal = evaluate_saved_checkpoint(checkpoint, calibration)
    policy_raw = evaluate_saved_checkpoint(checkpoint, policy)
    fitted = minimize_scalar(
        lambda log_t: log_loss(cal['truth'], temperature_scale(cal['probabilities'], np.exp(log_t)),
                               labels=np.arange(len(labels))),
        bounds=(np.log(0.25), np.log(10)), method='bounded',
    )
    temperature = float(np.exp(fitted.x))
    scaled = temperature_scale(policy_raw['probabilities'], temperature)
    if (log_loss(policy_raw['truth'], scaled, labels=np.arange(len(labels))) < policy_raw['metrics']['nll']
            and expected_calibration_error(policy_raw['truth'], scaled) <= policy_raw['metrics']['ece']):
        checkpoint['temperature'] = temperature
    else:
        scaled = policy_raw['probabilities']
    threshold = None
    for value in np.round(np.arange(0.50, 1.00, 0.01), 2):
        accepted = scaled.max(1) >= value
        if accepted.sum() >= 100 and np.mean(scaled.argmax(1)[accepted] == policy_raw['truth'][accepted]) >= 0.90:
            threshold = float(value)
            break
    checkpoint['review_policy'] = {'threshold': threshold, 'target_accuracy': 0.90,
                                   'minimum_policy_samples': 100,
                                   'brightness_stability': target == 'articleType'}
    result = evaluate_saved_checkpoint(checkpoint, selection)
    checkpoint['validation_metrics'] = {f'validation_{key}': value for key, value in result['metrics'].items()}
    checkpoint['best_validation_macro_f1'] = result['metrics']['macro_f1']
    if history is None:
        history = pd.DataFrame([{'method': 'selected_hog_hsv_recipe', **checkpoint['validation_metrics']}])
    return checkpoint, history


@torch.inference_mode()
def predict_cnn(model, loader, device):
    model.to(device).eval()
    probabilities = []
    for images, _ in loader:
        probabilities.append(model(images.to(device)).softmax(1).cpu().numpy())
    return np.vstack(probabilities)
