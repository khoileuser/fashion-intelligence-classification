# Figures for the report

These files are exported by the refreshed notebooks. Classification figures use fresh CPU inference from the current saved checkpoints and the unchanged internal test split. They do not describe the older baseline checkpoints.

| File | Source | Suggested use |
|---|---|---|
| `target_distributions.png` | Task 0, Section 5 | Appendix A: four target distributions; the article panel shows only the 30 largest classes, with logarithmic count axes. |
| `gender_confusion.png` | Task 3, Section 23 | Appendix B: current audience confusion matrix. |
| `usage_confusion.png` | Task 3, Section 23 | Appendix B: current usage confusion matrix. The Home row has no test support. |
| `retrieval_examples.png` | Task 4, Section 5 | Query images and five training-gallery neighbours; includes both successes and failures. |
| `article_type_confusion.png` | Task 1, Section 20 | Optional detailed article analysis; too dense for a small appendix panel. |
| `season_confusion.png` | Task 2, Section 17 | Optional substitute for the audience matrix if discussing seasonal errors. |
| `*_calibration.png` | Current checkpoint sections in Tasks 1–3 | Optional reliability diagrams; these use the saved temperature and ten confidence bins. |

For the two-page appendix limit, prioritize the target distributions on page one, then audience/usage matrices side by side and a short retrieval example panel on page two. Check labels at the final Word/PDF size; use fewer panels if they become unreadable. Existing numerical appendix tables would need to be reduced or replaced to make room.

## Recommended appendix titles and captions

**Page 1 — Appendix A. Target-Class Distributions**

Insert `target_distributions.png` as **Fig. A1. Target-label distributions in the usable dataset. Article types are limited to the 30 largest categories; counts use logarithmic axes.**

**Page 2 — Appendix B. Classification Errors and Visual Search Examples**

Place `gender_confusion.png` and `usage_confusion.png` side by side as **Fig. B1. Row-normalized internal-test confusion matrices for the final audience (left) and usage (right) classifiers. The Home usage class has no test examples.**

Below them, use three rows from `retrieval_examples.png`: a successful handbag example, a footwear confusion, and a swimwear failure. The full six-row image is too tall for this layout. Caption the selection **Fig. B2. Selected internal-test queries (first column) and their five nearest training-gallery images, illustrating successful and unsuccessful retrieval.**

Use about 16 cm total width for each figure group and check readability at the final page size. This is a proposed Word layout, not a verified two-page PDF. If labels are too small, retain one confusion matrix at a larger size and describe the omitted target's results in the main report.

The `*_metrics.csv` files contain freshly evaluated metrics. Each `*_evaluation.npz` stores test IDs, label order, true labels, probabilities and checkpoint SHA-256, allowing the plots to be traced to a particular model. These numerical files are supporting evidence, not figures to paste into Word.

The notebooks retain historical experiment analysis. Use their explicitly named **Current checkpoint evaluation and report figures** sections for final classification figures. Source-model metrics and fresh results are checked in those cells; the current evaluation does not retrain or overwrite models.
