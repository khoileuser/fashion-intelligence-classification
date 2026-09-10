# MLP and CNN comparison

These are measured results on the selection half of the frozen validation groups. All candidates in a task use the same images. Shallow MLP is the baseline; the four rows are trained model families. The CNN row is the best of three CNN configurations.

Selection prioritizes macro-F1, then accuracy, then fewer parameters. Calibration and review policy use separate validation groups. Historical full-validation scores and results from the rice dataset are not directly comparable.

## Validation accuracy

| Model | Article type | Season | Gender | Occasion |
| --- | --- | --- | --- | --- |
| Shallow MLP (baseline) | 0.692 | 0.683 | 0.819 | 0.850 |
| Deeper MLP | 0.727 | 0.705 | 0.851 | 0.865 |
| Best CNN | 0.870 | 0.752 | 0.897 | 0.907 |
| HOG + HSV logistic regression | 0.791 | 0.626 | 0.788 | 0.773 |

## Validation macro-F1

| Model | Article type | Season | Gender | Occasion |
| --- | --- | --- | --- | --- |
| Shallow MLP (baseline) | 0.332 | 0.626 | 0.654 | 0.358 |
| Deeper MLP | 0.369 | 0.675 | 0.728 | 0.372 |
| Best CNN | 0.672 | 0.740 | 0.774 | 0.459 |
| HOG + HSV logistic regression | 0.614 | 0.585 | 0.620 | 0.426 |

## CNN tuning

Changes below compare saved best-macro-F1 checkpoints on the same selection images. Positive accuracy changes are percentage-point gains; macro-F1 changes are absolute.

### Article type

| method | Accuracy | Macro-F1 | Accuracy change (pp) | Macro-F1 change |
| --- | --- | --- | --- | --- |
| cnn_ordinary | 0.821 | 0.578 | +0.00 | +0.000 |
| cnn_scheduled | 0.831 | 0.598 | +1.00 | +0.020 |
| cnn_four_blocks_scheduled | 0.870 | 0.672 | +4.83 | +0.094 |

### Season

| method | Accuracy | Macro-F1 | Accuracy change (pp) | Macro-F1 change |
| --- | --- | --- | --- | --- |
| cnn_ordinary | 0.723 | 0.700 | +0.00 | +0.000 |
| cnn_scheduled | 0.707 | 0.697 | -1.55 | -0.004 |
| cnn_four_blocks_scheduled | 0.752 | 0.740 | +2.94 | +0.040 |

### Gender

| method | Accuracy | Macro-F1 | Accuracy change (pp) | Macro-F1 change |
| --- | --- | --- | --- | --- |
| cnn_ordinary | 0.852 | 0.735 | +0.00 | +0.000 |
| cnn_scheduled | 0.873 | 0.740 | +2.07 | +0.004 |
| cnn_four_blocks_scheduled | 0.897 | 0.774 | +4.48 | +0.039 |

### Occasion

| method | Accuracy | Macro-F1 | Accuracy change (pp) | Macro-F1 change |
| --- | --- | --- | --- | --- |
| cnn_ordinary | 0.888 | 0.400 | +0.00 | +0.000 |
| cnn_scheduled | 0.894 | 0.407 | +0.62 | +0.007 |
| cnn_four_blocks_scheduled | 0.907 | 0.459 | +1.86 | +0.058 |

## Selected-model internal test

| Task | Selected method | Internal-test accuracy | Internal-test macro-F1 |
| --- | --- | --- | --- |
| Article type | cnn_four_blocks_scheduled | 0.870 | 0.633 |
| Season | cnn_four_blocks_scheduled | 0.739 | 0.718 |
| Gender | cnn_four_blocks_scheduled | 0.883 | 0.713 |
| Occasion | cnn_four_blocks_scheduled | 0.899 | 0.412 |

The internal test has prior development exposure. It did not select models or hyperparameters in this experiment. These results are not independent real-world performance estimates.

The results directory contains target-prefixed comparison and CNN tuning tables, and per-candidate learning histories. The web application loads the selected checkpoints directly from models/<target>_model.pt.
