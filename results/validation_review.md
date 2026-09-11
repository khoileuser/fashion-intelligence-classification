# Classification validation and tuning review

Executed only setup, added parameter-tuning, comparison, calibration, selected-model evaluation and export cells in Tasks 1–3. Original fitting cells were skipped. Task 4 and the submission were not run, as requested.

## Validation checks

- Frozen split IDs are unique. No product group or exact-image hash crosses splits.
- Per-class weighted recall reproduces accuracy, and supported-class F1 reproduces the selection metric.
- Selected scores agree with the restored best history epoch.
- Selected model metadata agrees with the summaries and model-selection tables.
- Validation selects models; subsequent internal-test results have prior development exposure.

## Results

| Target | Initial CNN validation F1 | Final validation F1 | Selected method | Internal-test accuracy | Internal-test F1 |
|---|---:|---:|---|---:|---:|
| article_type | 0.6664 | 0.6954 | cnn_continue_weighted | 86.34% | 0.6373 |
| season | 0.7363 | 0.7396 | cnn_continue | 73.95% | 0.7235 |
| gender | 0.7901 | 0.7901 | cnn_four_blocks_scheduled | 88.91% | 0.7398 |
| usage | 0.4187 | 0.4522 | shallow_mlp_lower_lr | 85.69% | 0.3745 |

## Interpretation limits

Selection macro-F1 covers only ground-truth-supported classes: 97/124 article types, 4/4 seasons, 5/5 gender categories and 8/9 occasion labels. The standard macro-average in the full per-class report includes all output labels and can be lower. Rare-class support and zero recall are discussed in the notebooks.

The MLP trials use lower learning rates, at most 20 epochs and patience 4. CNN trials use identical starting weights and at most eight additional epochs, with versus without training class weights. These are one-seed configuration comparisons; no repeated-run significance claim is made.

Sampled GPU checks observed batch/single-image probability differences, including approximately 0.0011 for article type and 0.0023 for season, without changed labels in those samples. These diagnostics do not prove agreement for every image. Float32 softmax values are explicitly renormalized after conversion to float64 in the runtime and notebooks.

## Remaining work

Task 4 retrieval evaluation, gallery regeneration and submission generation remain pending by user choice. Check final report pagination after copying into the document editor (five main pages and up to two appendix pages). Public web deployment has not been validated.
