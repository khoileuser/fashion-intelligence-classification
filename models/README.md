# Model artifact status

The classifier checkpoints and histories currently in this directory are prototype/mock artifacts created to test notebook execution, inference helpers, the API, and the web interface. They are not accepted final assignment models.

Classifier owners must replace the files for their assigned targets only after controlled validation-based comparison, one-time internal-test evaluation, completed failure analysis, and an evidence-backed ultimate judgement. Each task also returns its selected history CSV and a validation comparison CSV containing every baseline/candidate. Final checkpoints must follow `docs/CLASSIFIER_CONTRACT.md`; both compact-CNN and HOG+HSV logistic-regression checkpoints are supported.

The visual-search artifacts are owned separately by the foundation/visual-search member and must not be overwritten by classifier work.
