"""Build a compact report from completed experiments, without retraining."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from scripts.preprocessing import ROOT

DIRECTORY = ROOT / 'results'
TASKS = {'article_type': 'Article type', 'season': 'Season', 'gender': 'Gender', 'usage': 'Occasion'}
FAMILIES = {'shallow_mlp': 'Shallow MLP (baseline)', 'deeper_mlp': 'Deeper MLP',
            'cnn': 'Best CNN', 'hog_hsv': 'HOG + HSV logistic regression'}


def markdown_table(frame):
    # Avoid adding a tabulate dependency just for report generation.
    columns = [str(frame.index.name or 'Model'), *map(str, frame.columns)]
    rows = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    rows.extend('| ' + ' | '.join([str(index), *map(str, row)]) + ' |' for index, row in frame.iterrows())
    return '\n'.join(rows)


def build_report(directory=DIRECTORY, figures_dir=None):
    directory = Path(directory)
    figures_dir = Path(figures_dir) if figures_dir is not None else ROOT / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    comparisons = {task: pd.read_csv(directory / f'{task}_comparison.csv').set_index('family') for task in TASKS}
    summaries = {task: json.loads((directory / f'{task}_summary.json').read_text()) for task in TASKS}
    content = ['# MLP and CNN comparison',
               'These are measured results on the selection half of the frozen validation groups. '
               'All candidates in a task use the same images. Shallow MLP is the baseline; '
               'the four rows are trained model families. The CNN row is the best of three CNN configurations.',
               'Selection prioritizes macro-F1, then accuracy, then fewer parameters. '
               'Calibration and review policy use separate validation groups. Historical full-validation scores '
               'and results from the rice dataset are not directly comparable.']
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for axis, metric, title in zip(axes, ['validation_accuracy', 'validation_macro_f1'], ['Validation accuracy', 'Validation macro-F1']):
        table = pd.DataFrame({TASKS[task]: frame.loc[list(FAMILIES), metric].to_numpy() for task, frame in comparisons.items()},
                             index=list(FAMILIES.values()))
        content.extend([f'## {title}', markdown_table(table.map(lambda value: f'{value:.3f}'))])
        table.T.plot.bar(ax=axis, rot=0, ylim=(0, 1), title=title, legend=False)
        axis.set_ylabel(title)
        axis.grid(axis='y', alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(figures_dir / 'classification_comparison.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    content.extend(['## CNN tuning',
                    'Changes below compare saved best-macro-F1 checkpoints on the same selection images. '
                    'Positive accuracy changes are percentage-point gains; macro-F1 changes are absolute.'])
    for task, title in TASKS.items():
        tuning = pd.read_csv(directory / f'{task}_cnn_tuning.csv').set_index('method')
        table = pd.DataFrame({
            'Accuracy': tuning.validation_accuracy.map(lambda x: f'{x:.3f}'),
            'Macro-F1': tuning.validation_macro_f1.map(lambda x: f'{x:.3f}'),
            'Accuracy change (pp)': tuning.accuracy_change_vs_ordinary.map(lambda x: f'{100*x:+.2f}'),
            'Macro-F1 change': tuning.macro_f1_change_vs_ordinary.map(lambda x: f'{x:+.3f}'),
        })
        content.extend([f'### {title}', markdown_table(table)])
    selected = pd.DataFrame({TASKS[task]: {
        'Selected method': summary['selected'],
        'Internal-test accuracy': f"{summary['test_metrics']['accuracy']:.3f}",
        'Internal-test macro-F1': f"{summary['test_metrics']['macro_f1']:.3f}",
    } for task, summary in summaries.items()}).T
    selected.index.name = 'Task'
    content.extend(['## Selected-model internal test', markdown_table(selected),
                    'The internal test has prior development exposure. It did not select models or hyperparameters '
                    'in this experiment. These results are not independent real-world performance estimates.',
                    'The results directory contains target-prefixed comparison and CNN tuning tables, '
                    'and per-candidate learning histories. '
                    'The web application loads the selected checkpoints directly from models/<target>_model.pt.'])
    (directory / 'classification_report.md').write_text('\n\n'.join(content) + '\n', encoding='utf-8')
    print(directory / 'classification_report.md')


if __name__ == '__main__':
    build_report()
