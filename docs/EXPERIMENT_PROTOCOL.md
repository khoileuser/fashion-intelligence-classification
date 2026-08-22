# Experiment Protocol

## Before training

1. Confirm `scripts/data/audit.json`, `splits.csv`, and `normalization.json` exist.
2. Record the notebook version, seed, device, hypothesis, and changed parameters.
3. Verify that feature fitting, weighting, and augmentation use training rows only.
4. Run a simple baseline before the neural candidate.

## Selection

- Classifiers select primarily on validation macro F1. The versioned starter notebooks treat candidates within 0.01 macro F1 as practically near-tied, then prefer lower validation ECE and lower parameter count. Change this rule only before viewing internal-test results and document the reason.
- Macro F1 is computed over the fixed set of ground-truth labels supported by that evaluation partition. Labels absent from a partition are reported as absent rather than conditionally entering the metric only when predicted.
- Visual search selects on validation contrastive loss and frozen retrieval criteria.
- Accuracy, weighted behavior, calibration, latency, size, robustness, and qualitative evidence support the decision.
- The saved comparison CSV must contain the majority reference, classical candidate, every neural candidate, validation calibration, complexity, and selection eligibility; it must not contain only the winning loss mode.
- Change one justified factor per comparison.
- If candidates are practically tied, prefer the smaller or better-calibrated model.

## Internal test rule

The internal test set is evaluated once after the method is frozen. It is not another validation set. Any model change after seeing test results must be disclosed as test-set tuning.

## Required reporting

- Dataset support and excluded blanks
- Baseline and candidate results
- Learning curves and selected epoch
- Per-class or retrieval metrics
- Confusion/retrieval examples, including failures
- Calibration and mild-corruption robustness
- Runtime, model/index size, and reproducibility details
- Limitations and an evidence-based ultimate judgement
- Independent comparison with relevant literature or genuinely external data, including dataset and metric comparability caveats

Never write performance claims before the corresponding notebook output exists.
