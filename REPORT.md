# Fashion Intelligence: Image Classification and Visual Search

**COSC2753–3013 Machine Learning — Assignment 2, Semester 2026B**

**Group:** [Insert group number]  
**Lecturer / tutor:** [Insert name]  
**Submission date:** [Insert date]

| Student name | Student ID |
|---|---|
| [Student 1] | [ID] |
| [Student 2] | [ID] |
| [Student 3] | [ID] |
| [Student 4] | [ID] |
| [Student 5, if applicable] | [ID] |

<!-- Insert a page break after the cover. Main report begins below. -->

## 1. Introduction

This project develops an image-based fashion assistant for article-type classification (Task 1), season classification (Task 2), intended audience and usage classification (Task 3), and visual search (Task 4). The system combines four separately trained classifiers with a catalogue retrieval index and a web interface. All classifiers learn from the lecturer-provided dataset without externally pretrained weights. The original dataset remains unchanged; preprocessing, training augmentation and derived artifacts are maintained separately.

The objective is useful classification with manageable implementation complexity. Model selection therefore considers accuracy, minority-class performance and confidence calibration together. A simpler classifier is preferred when its results justify it, while retrieval is assessed separately from classification.

## 2. Data exploration and preprocessing: Task 0

### 2.1. Exploratory Data Analysis

The labelled CSV contains 38,617 unique IDs. Five images are missing and one available image cannot be decoded, leaving 38,611 usable examples. The unlabelled prediction set contains 5,829 rows. Image auditing identifies 636 exact-duplicate groups involving 1,399 rows. A random image-level split could therefore place identical products in both training and evaluation sets, inflating performance.

The usable data contain 124 article categories, four seasons, five audience labels and nine usage labels. Image exclusion removes all usable examples of `Suits`, so the article classifier cannot learn that original category. Twenty season values and one usage value are blank.

Class imbalance is substantial: on the initial validation split, majority-only accuracy is 17.75% for article type, 49.98% for season, 54.06% for audience and 77.22% for usage. Consequently, high usage accuracy alone provides weak evidence of broad recognition. Sparse categories also make per-class estimates unstable.

### 2.2. Data Preprocessing

Examples with blank target values are omitted only from the corresponding target's supervised fitting and evaluation. The literal usage label `NA` is preserved rather than mistaken for a missing value. Malformed product-description fields are handled during parsing without rewriting the source CSV.

Examples are grouped using normalized product names and exact image hashes before splitting into 27,051 training, 5,842 validation and 5,718 test images. Group membership is kept within one partition. Available target classes are represented in training, although some rare classes have no test examples. Product names support grouping only; classifiers receive image features, not catalogue descriptions or target metadata.

Images are converted to RGB. CNN inputs are resized to 96 × 128 pixels and normalized using training-only channel statistics. Handcrafted features use 48 × 64 images. Horizontal flips and selected brightness augmentation are applied during training, leaving source images and evaluation inputs unchanged.

## 3. Model approaches, experiments and results

Each classification target compares a majority baseline, logistic regression using handcrafted image features, and a small CNN. The following subsections present each target's approach, model comparison and final results separately.

Accuracy measures overall correctness; macro F1 gives each ground-truth-supported class equal weight. Expected calibration error (ECE) compares confidence and accuracy in ten confidence bins; lower is better. Negative log-likelihood (NLL) and multiclass Brier score provide additional probability-quality measurements. Macro F1 excludes classes absent from the evaluated ground truth, so it does not measure competence across every original label.

Initial comparisons use the full validation split. Later experiments divide validation groups into selection, calibration and review-policy subsets in approximate 50:25:25 proportions. Candidate decisions use validation evidence before evaluating selected replacements on test. However, these partitions had prior development exposure, and test results were inspected across earlier rounds. The reported test scores are internal evaluation evidence, not a pristine final independent assessment.

In Tables 1–4, validation rows describe the initial full-validation comparisons, while the final row reports the current checkpoint on internal test data. These different partitions must not be compared as a direct before–after experiment. F1 uses the 0–1 scale.

### 3.1. Article Type Classification (Task 1)

The article classifier predicts 124 categories using multinomial logistic regression. Histogram of oriented gradients (HOG) captures edge structure, while HSV histograms describe colour distributions. Their concatenation contains 1,284 features. The selected pipeline standardizes these features and uses regularization parameter C = 0.1 with square-root inverse-frequency class weights to moderate imbalance.

**Table 1. Article-type model comparison and final result.**

| Model | Evaluation split | Accuracy | Macro F1 |
|---|---|---:|---:|
| Majority baseline | Validation | 17.75% | 0.0029 |
| HOG + HSV logistic regression | Validation | 78.88% | 0.6096 |
| Small CNN | Validation | 82.45% | 0.5347 |
| Final logistic regression; brightness training | Test | 80.27% | 0.6302 |

In the initial comparison, logistic regression sacrifices accuracy relative to the CNN for substantially stronger macro F1. This trade-off supports the simpler model for a target with many imbalanced categories. Separately normalized shape/colour features and inference-time brightness averaging were also investigated but rejected after validation.

Article training uses brightness factors 0.9, 1.0 and 1.1, with each view assigned one-third weight. Compared with the immediately preceding checkpoint, accuracy increases from 80.12% to 80.27% and macro F1 from 0.6241 to 0.6302, while ECE worsens from 0.0105 to 0.0160. This modest calibration cost was accepted because classification was prioritized. Nevertheless, group-bootstrap intervals for the accuracy and F1 changes include zero: the observed gain is uncertain.

### 3.2. Season Classification (Task 2)

Season prediction uses a four-output CNN with three convolutional blocks containing 32, 64 and 128 channels, batch normalization, ReLU and max pooling. Adaptive pooling feeds a 128-unit dense layer and dropout of 0.2. Training uses Adam with initial learning rate 0.001, batch size 64, at most 30 epochs and early stopping after seven unimproved epochs. This approximately 160,000-parameter architecture learns spatial features without external pretraining.

**Table 2. Season model comparison and final result.**

| Model | Evaluation split | Accuracy | Macro F1 |
|---|---|---:|---:|
| Majority baseline | Validation | 49.98% | 0.1666 |
| HOG + HSV logistic regression | Validation | 62.90% | 0.5877 |
| Small CNN | Validation | 71.79% | 0.7009 |
| Final ordinary CNN | Test | 71.29% | 0.6934 |

The CNN improves both initial validation metrics over logistic regression. A fresh ordinary CNN run supplies the retained model; brightness-augmentation trials did not justify replacement. Its final ECE is 0.0282. Season remains the least accurate classification target, and visually similar products can belong to different seasons, limiting image-only prediction.

### 3.3. Gender / Audience Classification (Task 3)

Audience prediction uses a separately trained CNN with the architecture and basic training settings in Section 3.2, but a five-class output layer. The selected model additionally reduces the learning rate by a factor of 0.5 after two unimproved validation macro-F1 epochs. This allows smaller optimization steps when progress stalls without enlarging the network.

**Table 3. Gender / audience model comparison and final result.**

| Model | Evaluation split | Accuracy | Macro F1 |
|---|---|---:|---:|
| Majority baseline | Validation | 54.06% | 0.1404 |
| HOG + HSV logistic regression | Validation | 79.27% | 0.6253 |
| Small CNN | Validation | 86.75% | 0.7053 |
| Final CNN; learning-rate scheduling | Test | 87.25% | 0.6704 |

The CNN outperforms logistic regression on both initial validation metrics. Compared with the preceding checkpoint on the same test partition, the selected replacement raises accuracy from 85.38% to 87.25%, while macro F1 slightly decreases from 0.6725 to 0.6704. Its final ECE is 0.0196. Mild brightness augmentation did not justify a further replacement. Higher average correctness therefore does not establish uniformly better recognition across all audience labels.

### 3.4. Occasion / Usage Classification (Task 3)

Usage is treated as a separate nine-class prediction problem rather than combined with audience, avoiding a larger set of sparse joint labels. The selected model uses the standardized HOG + HSV features, C = 0.1 and square-root class weights described in Section 3.1, but is trained separately without the article model's brightness-view augmentation.

**Table 4. Occasion / usage model comparison and final result.**

| Model | Evaluation split | Accuracy | Macro F1 |
|---|---|---:|---:|
| Majority baseline | Validation | 77.22% | 0.1089 |
| HOG + HSV logistic regression | Validation | 76.91% | 0.4185 |
| Small CNN | Validation | 87.54% | 0.3887 |
| Final logistic regression | Test | 84.42% | 0.4134 |

Usage shows the clearest imbalance trade-off: the initial CNN achieves higher accuracy, but logistic regression has better macro F1. The final model's temperature is approximately 0.922, giving test ECE of 0.0191; the other classifiers retain temperature 1.0. Temperature scaling adjusts probabilities without changing the highest-scoring class (Guo et al., 2017). Low aggregate ECE does not guarantee trustworthy individual predictions or calibration under new photographic conditions. [Source: Guo et al.](https://proceedings.mlr.press/v70/guo17a.html)

Usage remains the weakest target by macro F1 despite its high accuracy. Several uncommon occasions have extremely limited evaluation support, and intended occasion may not be uniquely visible from a photograph.

### 3.5. Visual Search (Task 4)

#### 3.5.1. Retrieval approach and evaluation

The final search system ranks images by cosine similarity between fixed HOG and HSV features. Each feature block is independently unit-normalized, then combined using square-root weights. This prevents raw feature magnitudes from unintentionally determining the shape–colour balance. It is a fixed-feature nearest-neighbour system, with validation-selected settings; it does not train a retrieval neural network.

Evaluation uses only training images as the gallery, with validation and test images as separate queries. Matching article type defines relevance. This reproducible proxy measures category agreement, although it does not capture all aspects of perceived similarity or exact product identity.

**Table 5. Retrieval configuration comparison on validation queries.**

| Representation | Precision@5 | MRR |
|---|---:|---:|
| RGB thumbnail | 66.92% | 0.8022 |
| Raw HOG + HSV | 72.26% | 0.8381 |
| 25% shape / 75% colour | 72.63% | 0.8394 |
| 50% shape / 50% colour | 74.58% | 0.8534 |
| 75% shape / 25% colour | 75.14% | 0.8587 |

The selected 75% shape configuration improves validation precision@5 by 2.88 percentage points over raw HOG + HSV. Test precision@1 is 79.78%, precision@5 is 74.83%, hit rate@5 is 93.30%, and mean reciprocal rank is 0.8581. However, class-macro precision@5 falls to 51.45%, exposing weaker retrieval for less common categories. Recall@5 is only 1.07% because its denominator includes every relevant gallery item; it is not interchangeable with hit rate.

Qualitative results retrieve consistent handbags and briefs but confuse footwear styles and return tops or vests for swimwear. Colour agreement among the top five results is only 25.63%. Optimizing category precision therefore does not guarantee matching colour or fine-grained style.

#### 3.5.2. Real-world Application

The web application demonstrates how the models could support fashion catalogue management and product discovery. Catalogue staff can upload a product image and review suggested article, season, audience and usage labels before accepting them. Shoppers could use the same image-upload workflow to discover visually similar catalogue items without knowing the exact product name. Confidence and review indicators support human checking, while image-based retrieval provides alternatives to inspect. These are intended uses of a locally tested educational prototype; public deployment and evaluation on independently collected real-world photographs have not been completed.

The Next.js interface calls a FastAPI backend to display four classifications, confidence and review indicators, and five similar catalogue images. It includes persistent local history and a mobile layout. After evaluation, the deployment index includes all 38,611 usable labelled images; its 1,284-dimensional embeddings occupy approximately 189 MiB. Ten warm local CPU searches on one query have a median of 73.1 ms, which is a smoke measurement rather than a load benchmark.

Eight browser uploads verify agreement with direct inference, image loading, history persistence, mobile layout and invalid-file rejection. They also reveal classification failures. With the latest article model, a T-shirt is labelled Sweatshirts at approximately 65% confidence; its darkened version remains incorrect at approximately 82%. Review indicators identify some low-confidence or lighting-sensitive outputs but do not repair their labels. These catalogue-image checks, including retrieval self-matches, demonstrate integration rather than independent generalization.

## 4. Comparison with research and ultimate judgment

DeepFashion investigates the same broad goals of clothing recognition and retrieval. FashionNet jointly learns attributes and landmarks, and reports 82.58% top-three category accuracy across 50 categories (Liu et al., 2016, Table 2). Our article score is 80.27% top-one accuracy with 124 usable categories. Different datasets, supervision and ranking metrics prevent a valid numerical superiority claim. The methodological contrast is useful: our compact image-only features require fewer annotations but lack explicit garment localization. [Source: Liu et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.pdf)

DeepFashion2 combines clothing detection, landmarks, segmentation and consumer-to-shop retrieval (Ge et al., 2019). Its product-matching setting differs from our same-category relevance proxy. Consequently, our 93.30% hit rate@5 should not be interpreted as equivalent consumer-to-shop retrieval performance. These differences suggest that clutter, occlusion and viewpoint changes deserve separate evaluation before broader use. [Source: Ge et al.](https://arxiv.org/abs/1901.07973)

The implemented system is a useful educational prototype and catalogue-assistance tool. Separate model selection improves the balance between simplicity and performance, and the web interface makes errors inspectable. It is not yet supported as an unattended production classifier: usage has weak macro F1, rare categories remain difficult, and confident browser mistakes persist. Audience predictions represent catalogue labels rather than a determination of a person's identity.

The literature comparison provides external context, but no independently collected image set has been evaluated. A next assessment should freeze the models and test separately sourced photographs with documented label mapping, reporting per-class results and uncertainty. For improvement, targeted training-only augmentation and limited regularization or scheduler changes are preferable first steps to increasing architecture complexity. The dataset's educational-use restriction also limits deployment beyond this assignment.

<!-- Main report ends here. Insert a page break before references. -->

## References

Ge, Y., Zhang, R., Wang, X., Tang, X. and Luo, P. (2019). *DeepFashion2: A Versatile Benchmark for Detection, Pose Estimation, Segmentation and Re-Identification of Clothing Images*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. [Paper](https://arxiv.org/abs/1901.07973).

Guo, C., Pleiss, G., Sun, Y. and Weinberger, K.Q. (2017). *On Calibration of Modern Neural Networks*. Proceedings of the 34th International Conference on Machine Learning, 70, pp. 1321–1330. [Paper](https://proceedings.mlr.press/v70/guo17a.html).

Liu, Z., Luo, P., Qiu, S., Wang, X. and Tang, X. (2016). *DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1096–1104. [Paper](https://openaccess.thecvf.com/content_cvpr_2016/html/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.html).

RMIT University (2026). *COSC2753–3013 Machine Learning: Assignment 2 — 2026B*. Assignment specification and supplied fashion dataset. Course materials.

<!-- Supporting appendices begin below. Keep the combined appendices within two pages. -->

## Appendix A. Supporting data and probability metrics

**Table A1. Dataset audit and evaluation support.**

| Item | Count |
|---|---:|
| Original labelled CSV rows | 38,617 |
| Missing images / undecodable available images | 5 / 1 |
| Usable labelled images | 38,611 |
| Exact-duplicate groups / involved rows | 636 / 1,399 |
| Training / validation / test images | 27,051 / 5,842 / 5,718 |
| Article / audience / usage test examples | 5,718 each |
| Season test examples with valid labels | 5,715 |
| Article classes in training / supported in test | 124 / 101 |
| Unlabelled prediction rows | 5,829 |

**Table A2. Current classifier probability metrics. Lower is better for both columns.**

| Target | NLL | Brier score |
|---|---:|---:|
| Article type | 0.6258 | 0.2755 |
| Season | 0.7050 | 0.3891 |
| Audience | 0.3401 | 0.1808 |
| Usage | 0.4364 | 0.2371 |

**Table A3. Group-bootstrap intervals for the latest article replacement minus its predecessor, in absolute 0–1 metric units.**

| Metric change | Recorded interval |
|---|---:|
| Accuracy | [−0.00243, 0.00571] |
| Macro F1 | [−0.00747, 0.02107] |
| ECE | [−0.00553, 0.01085] |

## Appendix B. Supporting retrieval results

**Table B1. Test retrieval metrics with the training-only gallery.**

| Metric | Value |
|---|---:|
| Precision@1 | 0.7978 |
| Precision@5 | 0.7483 |
| Precision@10 | 0.7192 |
| Hit rate@5 | 0.9330 |
| Hit rate@10 | 0.9591 |
| Recall@5 | 0.0107 |
| Mean reciprocal rank | 0.8581 |
| Class-macro precision@5 | 0.5145 |

**Table B2. Agreement between query attributes and the five retrieved items.**

| Attribute | Agreement |
|---|---:|
| Article type | 74.83% |
| Subcategory | 92.16% |
| Audience | 83.92% |
| Usage | 84.72% |
| Base colour | 25.63% |

These tables summarize category-based retrieval evaluation; attribute agreement is not classifier accuracy.
