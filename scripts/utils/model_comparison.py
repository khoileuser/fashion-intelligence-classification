"""Reproducible MLP/CNN experiments using frozen, disjoint validation groups.



Run as ``python -m scripts.utils.model_comparison``. Completed candidates are resumed

only when their data, implementation and experiment settings match this run.

"""

from __future__ import annotations



import argparse

from concurrent.futures import ThreadPoolExecutor

from copy import deepcopy

import hashlib

import json

from pathlib import Path



import numpy as np

import pandas as pd

from PIL import Image

from scipy.optimize import minimize_scalar

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import log_loss

from sklearn.model_selection import GroupShuffleSplit

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

import torch



from app.server.utils.classifier import temperature_scale

from app.server.utils.handcrafted import DEFAULT_FEATURE_CONFIG

from app.server.utils.modeling import FashionMLP, SimpleCNN, TunedCNN

from scripts.utils.classification import comparison_row, extract_features, predict_cnn, train_cnn

from scripts.utils.evaluation import evaluate_saved_checkpoint, expected_calibration_error

from scripts.preprocessing import (IMAGE_SIZE, NORMALISATION_PATH, ROOT, SEED,

                                   SPLIT_PATH, seed_everything, select_torch_device, task_frame)



EXPERIMENTS = {

    'shallow_mlp': ('shallow_mlp', False),

    'deeper_mlp': ('deeper_mlp', False),

    'cnn_ordinary': ('simple_cnn', False),

    'cnn_scheduled': ('simple_cnn', True),

    'cnn_four_blocks_scheduled': ('tuned_cnn', True),

}





class CachedBatches:

    """Decode once to uint8 RAM; normalize and flip batches on the device.



    PIL bilinear resize is identical to application preprocessing. The cache

    contains only deterministic pixels; random flips happen during training.

    """

    def __init__(self, frame, target, labels, normalisation, device, training=False, batch_size=64):

        self.dataset = frame

        self.training, self.device, self.batch_size = training, device, batch_size

        def read(path):

            with Image.open(path) as image:

                return np.asarray(image.convert('RGB').resize(IMAGE_SIZE, Image.Resampling.BILINEAR)).copy()

        with ThreadPoolExecutor(max_workers=4) as pool:

            self.images = torch.from_numpy(np.stack(list(pool.map(read, frame.image_path)))).permute(0, 3, 1, 2)

        self.targets = torch.tensor([labels.index(label) for label in frame[target]])

        self.mean = torch.tensor(normalisation['mean'], device=device).view(1, 3, 1, 1)

        self.std = torch.tensor(normalisation['std'], device=device).view(1, 3, 1, 1)



    def __iter__(self):

        indices = torch.randperm(len(self.dataset)) if self.training else torch.arange(len(self.dataset))

        for chunk in indices.split(self.batch_size):

            images = self.images[chunk].to(self.device, dtype=torch.float32) / 255

            if self.training:

                flips = torch.rand(len(chunk), device=self.device) < 0.5

                images[flips] = images[flips].flip(-1)

            yield (images - self.mean) / self.std, self.targets[chunk].to(self.device)





def validation_views(frame):

    selection_ids, rest_ids = next(GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=SEED)

                                   .split(frame, groups=frame.group_key))

    selection, rest = frame.iloc[selection_ids], frame.iloc[rest_ids]

    cal_ids, policy_ids = next(GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=SEED + 1)

                              .split(rest, groups=rest.group_key))

    return selection, rest.iloc[cal_ids], rest.iloc[policy_ids]





def ranked(table):

    """Predeclared: macro F1, then accuracy, then fewer parameters."""

    return table.sort_values(['validation_macro_f1', 'validation_accuracy', 'complexity_parameters'],

                             ascending=[False, False, True], kind='stable')





def calibrate(checkpoint, calibration, policy, device):

    checkpoint = deepcopy(checkpoint)

    cal = evaluate_saved_checkpoint(checkpoint, calibration, device=str(device))

    raw = evaluate_saved_checkpoint(checkpoint, policy, device=str(device))

    fit = minimize_scalar(lambda log_t: log_loss(

        cal['truth'], temperature_scale(cal['probabilities'], np.exp(log_t)),

        labels=np.arange(len(checkpoint['labels']))),

        bounds=(np.log(0.25), np.log(10)), method='bounded')

    temperature = float(np.exp(fit.x))

    scaled = temperature_scale(raw['probabilities'], temperature)

    if (log_loss(raw['truth'], scaled, labels=np.arange(len(checkpoint['labels']))) < raw['metrics']['nll']

            and expected_calibration_error(raw['truth'], scaled) <= raw['metrics']['ece']):

        checkpoint['temperature'] = temperature

    else:

        scaled = raw['probabilities']

    threshold = None

    for value in np.round(np.arange(0.50, 1.00, 0.01), 2):

        accepted = scaled.max(1) >= value

        if accepted.sum() >= 100 and np.mean(scaled.argmax(1)[accepted] == raw['truth'][accepted]) >= 0.90:

            threshold = float(value)

            break

    checkpoint['review_policy'] = {'threshold': threshold, 'target_accuracy': 0.90,

                                   'minimum_policy_samples': 100,

                                   'brightness_stability': checkpoint['target'] == 'articleType'}

    return checkpoint





def run_comparison(target, max_epochs=30, output_dir=None, methods=None, finalize=True, results_dir=None, cache_dir=None):

    """Train/resume requested methods; optionally select and evaluate the winner.



    Notebook model sections pass methods and finalize=False. Their final

    comparison section resumes every candidate and performs selection once.

    """

    requested = list(methods) if methods is not None else [*EXPERIMENTS, 'hog_hsv_logistic_regression']

    if not requested or set(requested) - {*EXPERIMENTS, 'hog_hsv_logistic_regression'}:

        raise ValueError('Choose at least one supported classification method')

    if finalize and set(requested) != {*EXPERIMENTS, 'hog_hsv_logistic_regression'}:

        raise ValueError('Final selection requires all classification candidates')

    stem = 'article_type' if target == 'articleType' else target

    directory = Path(output_dir or ROOT / 'models')

    directory.mkdir(parents=True, exist_ok=True)
    results = Path(results_dir or ROOT / 'results')
    cache = Path(cache_dir or ROOT / '.cache/training')
    results.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(4)

    device = select_torch_device()

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

    train = task_frame(target, 'train')

    selection, calibration, policy = validation_views(task_frame(target, 'validation'))

    for other in (selection, calibration, policy):

        assert set(train.group_key).isdisjoint(other.group_key)

    labels = sorted(train[target].unique())

    normalisation = json.loads(NORMALISATION_PATH.read_text())

    fingerprint = hashlib.sha256()

    for path in (Path(__file__), ROOT / 'scripts/utils/classification.py',

                 ROOT / 'app/server/utils/modeling.py', ROOT / 'app/server/utils/classifier.py',

                 ROOT / 'app/server/utils/handcrafted.py', ROOT / 'scripts/utils/evaluation.py',

                 SPLIT_PATH, NORMALISATION_PATH):

        fingerprint.update(path.read_bytes())

    fingerprint.update(train[['id', target, 'group_key']].to_csv(index=False).encode())

    fingerprint.update(json.dumps({'target': target, 'epochs': max_epochs, 'seed': SEED,

                                   'experiments': EXPERIMENTS}).encode())

    signature = fingerprint.hexdigest()

    base = {'target': target, 'labels': labels, 'temperature': 1.0, 'seed': SEED,

            'experiment': 'mlp_cnn_comparison', 'run_signature': signature,

            'selection_metric': 'validation_macro_f1_then_accuracy_then_parameters',

            'validation_scope': 'selection_view_only',

            'split_sizes': dict(train=len(train), selection=len(selection),

                                calibration=len(calibration), policy=len(policy))}

    truth = np.asarray([labels.index(label) for label in selection[target]])

    rows = []

    checkpoints = {}

    train_loader = validation_loader = None

    for method in requested:

        artifact = cache / f'{stem}_{method}.pt'

        cached = torch.load(artifact, map_location='cpu', weights_only=False) if artifact.exists() else None

        if cached is not None and cached.get('run_signature') == signature:

            print(f'{target}: resume {method}', flush=True)

            checkpoint = cached

            row = checkpoint['comparison_row']

        else:

            seed_everything(SEED)

            torch.manual_seed(SEED)

            print(f'{target}: training {method} on {device}', flush=True)

            checkpoint = dict(base)

            if method == 'hog_hsv_logistic_regression':

                estimator = make_pipeline(StandardScaler(), LogisticRegression(

                    class_weight='balanced', max_iter=1500, solver='lbfgs', random_state=SEED))

                estimator.fit(extract_features(train), train[target])

                probabilities = estimator.predict_proba(extract_features(selection))

                probabilities = probabilities[:, [list(estimator.classes_).index(label) for label in labels]]

                checkpoint.update(model_type=method, estimator=estimator, feature_config=DEFAULT_FEATURE_CONFIG)

                parameters = estimator[-1].coef_.size + estimator[-1].intercept_.size

                history = pd.DataFrame()

            else:

                if train_loader is None:

                    print('Decoding training and selection images once into RAM.', flush=True)

                    train_loader = CachedBatches(train, target, labels, normalisation, device, training=True)

                    validation_loader = CachedBatches(selection, target, labels, normalisation, device)

                model_type, schedule = EXPERIMENTS[method]

                if model_type.endswith('mlp'):

                    model = FashionMLP(len(labels), deep=model_type == 'deeper_mlp')

                else:

                    model = (TunedCNN if model_type == 'tuned_cnn' else SimpleCNN)(len(labels))

                parameters = sum(p.numel() for p in model.parameters())

                model, history = train_cnn(model, train_loader, validation_loader, device,

                                           max_epochs=max_epochs, patience=7, schedule=schedule)

                probabilities = predict_cnn(model, validation_loader, device)

                checkpoint.update(model_type=model_type, state_dict=deepcopy(model.cpu().state_dict()),

                                  mean=normalisation['mean'], std=normalisation['std'],

                                  image_size=list(IMAGE_SIZE), dropout=0.2,

                                  training_config={'schedule': schedule, 'learning_rate': 0.001,

                                                   'max_epochs': max_epochs, 'patience': 7,

                                                   'augmentation': 'horizontal_flip', 'batch_size': 64})

                del model

            row = comparison_row(method, checkpoint['model_type'], truth, probabilities, parameters, len(history))

            checkpoint['comparison_row'] = row

            checkpoint['validation_metrics'] = {k: v for k, v in row.items() if k.startswith('validation_')}

            checkpoint['best_validation_macro_f1'] = row['validation_macro_f1']

            torch.save(checkpoint, artifact)

            history.to_csv(results / f'{stem}_{method}_history.csv', index=False)

        checkpoints[method] = checkpoint

        rows.append(row)

        if finalize:

            pd.DataFrame(rows).to_csv(results / f'{stem}_experiments.csv', index=False)

        print(pd.DataFrame(rows)[['method', 'validation_accuracy', 'validation_macro_f1']].to_string(index=False), flush=True)



    experiments = pd.DataFrame(rows).set_index('method')

    if not finalize:

        return experiments

    cnn = experiments.loc[list(EXPERIMENTS)[2:]].copy()

    cnn['accuracy_change_vs_ordinary'] = cnn.validation_accuracy - cnn.loc['cnn_ordinary', 'validation_accuracy']

    cnn['macro_f1_change_vs_ordinary'] = cnn.validation_macro_f1 - cnn.loc['cnn_ordinary', 'validation_macro_f1']

    cnn.to_csv(results / f'{stem}_cnn_tuning.csv')

    cnn_winner = ranked(cnn).index[0]

    comparison = experiments.loc[['shallow_mlp', 'deeper_mlp', cnn_winner, 'hog_hsv_logistic_regression']].copy()

    comparison['family'] = ['shallow_mlp', 'deeper_mlp', 'cnn', 'hog_hsv']

    selected = ranked(comparison.loc[comparison.eligible_for_selection]).index[0]

    comparison['selected'] = comparison.index == selected

    comparison.to_csv(results / f'{stem}_comparison.csv')

    print(f'{target}: selected {selected}; calibrating on separate groups.', flush=True)

    checkpoint = calibrate(checkpoints[selected], calibration, policy, device)

    # The internal test already had prior development exposure; it never selects candidates here.

    result = evaluate_saved_checkpoint(checkpoint, task_frame(target, 'test'), device=str(device))

    checkpoint['test_metrics'] = result['metrics']

    torch.save(checkpoint, directory / f'{stem}_model.pt')

    selected_history = results / f'{stem}_{selected}_history.csv'

    if selected_history.stat().st_size > 2:

        pd.read_csv(selected_history).to_csv(results / f'{stem}_history.csv', index=False)

    else:

        pd.DataFrame([checkpoint['comparison_row']]).to_csv(results / f'{stem}_history.csv', index=False)

    pd.Series(result['metrics']).to_csv(results / f'{stem}_test_metrics.csv')

    (results / f'{stem}_summary.json').write_text(json.dumps({

        'target': target, 'selected': selected, 'cnn_selected': cnn_winner,

        'highest_validation_accuracy': experiments.loc[experiments.eligible_for_selection].validation_accuracy.idxmax(),

        'test_metrics': result['metrics'], 'run_signature': signature,

        'split_sizes': base['split_sizes'],

        'test_scope': 'internal_test_with_prior_development_exposure',

    }, indent=2) + '\n')

    return comparison, cnn





def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--targets', nargs='+', default=['articleType', 'season', 'gender', 'usage'],

                        choices=['articleType', 'season', 'gender', 'usage'])

    parser.add_argument('--max-epochs', type=int, default=30)

    parser.add_argument('--output-dir', type=Path)

    args = parser.parse_args()

    if args.max_epochs < 1:

        parser.error('--max-epochs must be positive')

    for target in args.targets:

        run_comparison(target, args.max_epochs, args.output_dir)





if __name__ == '__main__':

    main()

