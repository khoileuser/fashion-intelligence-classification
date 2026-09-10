# Fashion Intelligence: Image Classification and Visual Search

**COSC2753–3013 Machine Learning — Assignment 2, Semester 2026B**

**Group:** [Insert group number]  
**Lecturer / tutor:** [Insert name]  
**Submission date:** [Insert date]

| Student name               | Student ID |
| -------------------------- | ---------- |
| [Student 1]                | [ID]       |
| [Student 2]                | [ID]       |
| [Student 3]                | [ID]       |
| [Student 4]                | [ID]       |
| [Student 5, if applicable] | [ID]       |

<!-- Insert a page break after the cover. Main report begins below. -->

## 1. Introduction

Fashion catalogue management and product discovery depend on describing clothing items and identifying relevant alternatives. Image-based classification can support product labelling, while visual search allows users to find similar items without knowing their exact names. However, differences in garment appearance, pose and photographic conditions make automated recognition challenging. DeepFashion identifies deformation, occlusion and differences between catalogue and consumer photographs as obstacles to clothing recognition [1].

Machine learning offers approaches for learning clothing characteristics from image data. FashionNet jointly predicts clothing attributes and landmarks to support recognition and retrieval [1]. DeepFashion2 extends this research to clothing detection, pose estimation, segmentation and consumer-to-shop retrieval [2]. These studies provide a foundation for fashion-image analysis, although their datasets and annotation requirements differ from this assignment. Our project investigates whether simpler handcrafted-feature classifiers and compact convolutional neural networks can provide useful results using only the supplied training data.

This project develops an end-to-end fashion assistant addressing four tasks: article-type classification, season classification, audience and usage classification, and visual search [3]. The lecturer-provided dataset supplies 38,611 usable labelled images and 5,829 unlabelled prediction examples. The workflow covers exploratory analysis, preprocessing, model comparison, tuning and evaluation, followed by integration into a web application. Four classifiers are trained without externally pretrained weights and combined with a fixed-feature retrieval index. The objective is to balance classification accuracy, minority-class performance, confidence calibration and implementation simplicity while preserving the original dataset.

## 2. Data exploration and preprocessing: Task 0

### 2.1. Exploratory Data Analysis

The labelled CSV contains 38,617 unique IDs. Five images are missing and one available image cannot be decoded, leaving 38,611 usable examples. The unlabelled prediction set contains 5,829 rows. Image auditing identifies 636 exact-duplicate groups involving 1,399 rows. A random image-level split could therefore place identical products in both training and evaluation sets, inflating performance.

The usable data contain 124 article categories, four seasons, five audience labels and nine usage labels. Image exclusion removes all usable examples of `Suits`, so the article classifier cannot learn that original category. Twenty season values and one usage value are blank.

Class imbalance is substantial: on the initial validation split, majority-only accuracy is 17.75% for article type, 49.98% for season, 54.06% for audience and 77.22% for usage. Consequently, high usage accuracy alone provides weak evidence of broad recognition. Sparse categories also make per-class estimates unstable. We therefore report the mean of per-class F1 scores, distinguishing it from alternative definitions of macro F1 discussed by Opitz and Burst [4]. The target-label distributions in Fig. A1 illustrate the imbalance across the four classification targets.

### 2.2. Data Preprocessing

Examples with blank target values are omitted only from the corresponding target's supervised fitting and evaluation. The literal usage label `NA` is preserved rather than mistaken for a missing value. Malformed product-description fields are handled during parsing without rewriting the source CSV.

Examples are grouped using normalized product names and exact image hashes before splitting into 27,051 training, 5,842 validation and 5,718 test images. Group membership is kept within one partition. Available target classes are represented in training, although some rare classes have no test examples. Product names support grouping only; classifiers receive image features, not catalogue descriptions or target metadata.

Images are converted to RGB. CNN inputs are resized to 96 × 128 pixels and normalized using training-only channel statistics. Handcrafted features use 48 × 64 images. Horizontal flips and selected brightness augmentation are applied during training, leaving source images and evaluation inputs unchanged. Fitting preprocessing statistics only on training data follows established leakage-prevention guidance [5].

## 3. Model approaches, experiments and results

Each classification target compares a majority baseline, logistic regression using handcrafted image features, and a small CNN. The following subsections present each target's approach, model comparison and final results separately.

Accuracy measures overall correctness; macro F1 gives each ground-truth-supported class equal weight. Expected calibration error (ECE) compares confidence and accuracy in ten confidence bins; lower is better. This follows the binned calibration framework discussed by Guo et al. [6]. Negative log-likelihood (NLL) and multiclass Brier score provide additional probability-quality measurements. Macro F1 excludes classes absent from the evaluated ground truth, so it does not measure competence across every original label.

Initial comparisons use the full validation split. Later experiments divide validation groups into selection, calibration and review-policy subsets in approximate 50:25:25 proportions. Candidate decisions use validation evidence before evaluating selected replacements on test. However, these partitions had prior development exposure, and test results were inspected across earlier rounds. The reported test scores are internal evaluation evidence, not a pristine final independent assessment.

In Tables 1–4, validation rows describe the initial full-validation comparisons, while the final row reports the current checkpoint on internal test data. These different partitions must not be compared as a direct before–after experiment. F1 uses the 0–1 scale.

### 3.1. Article Type Classification (Task 1)

The article classifier predicts 124 categories using multinomial logistic regression. Histogram of oriented gradients (HOG) captures local edge-orientation structure, following the descriptor introduced by Dalal and Triggs for human detection; here it is applied to fashion images [7]. HSV histograms describe colour distributions. Their concatenation contains 1,284 features. The selected pipeline standardizes these features and uses regularization parameter C = 0.1 with square-root inverse-frequency class weights to moderate imbalance.

**Table 1. Article-type model comparison and final result.**

| Model                                          | Evaluation split | Accuracy | Macro F1 |
| ---------------------------------------------- | ---------------- | -------: | -------: |
| Majority baseline                              | Validation       |   17.75% |   0.0029 |
| HOG + HSV logistic regression                  | Validation       |   78.88% |   0.6096 |
| Small CNN                                      | Validation       |   82.45% |   0.5347 |
| Final logistic regression; brightness training | Test             |   80.27% |   0.6302 |

In the initial comparison, logistic regression sacrifices accuracy relative to the CNN for substantially stronger macro F1. This trade-off supports the simpler model for a target with many imbalanced categories. Separately normalized shape/colour features and inference-time brightness averaging were also investigated but rejected after validation. The current frequent-class confusion matrix in Fig. B1 shows errors among article categories, with predictions outside the displayed classes retained in an Other predicted class column.

Article training uses brightness factors 0.9, 1.0 and 1.1, with each view assigned one-third weight. Compared with the immediately preceding checkpoint, accuracy increases from 80.12% to 80.27% and macro F1 from 0.6241 to 0.6302, while ECE worsens from 0.0105 to 0.0160. This modest calibration cost was accepted because classification was prioritized. Nevertheless, group-bootstrap intervals for the accuracy and F1 changes include zero: the observed gain is uncertain.

### 3.2. Season Classification (Task 2)

Season prediction uses a four-output CNN with three convolutional blocks containing 32, 64 and 128 channels, batch normalization, ReLU and max pooling. Adaptive pooling feeds a 128-unit dense layer and dropout of 0.2. Training uses Adam, an adaptive gradient-based optimizer [8]. Our settings are initial learning rate 0.001, batch size 64, at most 30 epochs and early stopping after seven unimproved epochs. This approximately 160,000-parameter architecture learns spatial features without external pretraining.

**Table 2. Season model comparison and final result.**

| Model                         | Evaluation split | Accuracy | Macro F1 |
| ----------------------------- | ---------------- | -------: | -------: |
| Majority baseline             | Validation       |   49.98% |   0.1666 |
| HOG + HSV logistic regression | Validation       |   62.90% |   0.5877 |
| Small CNN                     | Validation       |   71.79% |   0.7009 |
| Final ordinary CNN            | Test             |   71.29% |   0.6934 |

The CNN improves both initial validation metrics over logistic regression. A fresh ordinary CNN run supplies the retained model; brightness-augmentation trials did not justify replacement. Its final ECE is 0.0282. Season remains the least accurate classification target, and visually similar products can belong to different seasons, limiting image-only prediction. The confusion matrix in Fig. B2 shows that Fall, Spring and Winter items are frequently classified as Summer.

### 3.3. Gender / Audience Classification (Task 3)

Audience prediction uses a separately trained CNN with the architecture and basic training settings in Section 3.2, but a five-class output layer. The selected model additionally reduces the learning rate by a factor of 0.5 after two unimproved validation macro-F1 epochs. This uses the metric-monitoring behaviour of PyTorch’s ReduceLROnPlateau scheduler; the factor, patience and monitored metric are project settings [9]. It allows smaller optimization steps when progress stalls without enlarging the network.

**Table 3. Gender / audience model comparison and final result.**

| Model                               | Evaluation split | Accuracy | Macro F1 |
| ----------------------------------- | ---------------- | -------: | -------: |
| Majority baseline                   | Validation       |   54.06% |   0.1404 |
| HOG + HSV logistic regression       | Validation       |   79.27% |   0.6253 |
| Small CNN                           | Validation       |   86.75% |   0.7053 |
| Final CNN; learning-rate scheduling | Test             |   87.25% |   0.6704 |

The CNN outperforms logistic regression on both initial validation metrics. Compared with the preceding checkpoint on the same test partition, the selected replacement raises accuracy from 85.38% to 87.25%, while macro F1 slightly decreases from 0.6725 to 0.6704. Its final ECE is 0.0196. Mild brightness augmentation did not justify a further replacement. Higher average correctness therefore does not establish uniformly better recognition across all audience labels. The audience confusion matrix in Fig. B3 illustrates the remaining errors across adult, children's and Unisex labels.

### 3.4. Occasion / Usage Classification (Task 3)

Usage is treated as a separate nine-class prediction problem rather than combined with audience, avoiding a larger set of sparse joint labels. The selected model uses the standardized HOG + HSV features, C = 0.1 and square-root class weights described in Section 3.1, but is trained separately without the article model's brightness-view augmentation.

**Table 4. Occasion / usage model comparison and final result.**

| Model                         | Evaluation split | Accuracy | Macro F1 |
| ----------------------------- | ---------------- | -------: | -------: |
| Majority baseline             | Validation       |   77.22% |   0.1089 |
| HOG + HSV logistic regression | Validation       |   76.91% |   0.4185 |
| Small CNN                     | Validation       |   87.54% |   0.3887 |
| Final logistic regression     | Test             |   84.42% |   0.4134 |

Usage shows the clearest imbalance trade-off: the initial CNN achieves higher accuracy, but logistic regression has better macro F1. The final model's temperature is approximately 0.922, giving test ECE of 0.0191; the other classifiers retain temperature 1.0. Temperature scaling adjusts probabilities without changing the highest-scoring class [6]. Low aggregate ECE does not guarantee trustworthy individual predictions or calibration under new photographic conditions.

Usage remains the weakest target by macro F1 despite its high accuracy. Several uncommon occasions have extremely limited evaluation support, and intended occasion may not be uniquely visible from a photograph. The usage confusion matrix in Fig. B4 shows the tendency to predict Casual for several uncommon usage labels; Home has no test examples.

### 3.5. Visual Search (Task 4)

#### 3.5.1. Retrieval approach and evaluation

The final search system ranks images by cosine similarity between fixed HOG and HSV features. Cosine ranking follows the vector-space retrieval principle described by Manning et al.; our implementation applies it to image descriptors rather than text vectors [10]. Each feature block is independently unit-normalized, then combined using square-root weights. This prevents raw feature magnitudes from unintentionally determining the shape–colour balance. It is a fixed-feature nearest-neighbour system, with validation-selected settings; it does not train a retrieval neural network.

Evaluation uses only training images as the gallery, with validation and test images as separate queries. Matching article type defines relevance. This reproducible proxy measures category agreement, although it does not capture all aspects of perceived similarity or exact product identity.

**Table 5. Retrieval configuration comparison on validation queries.**

| Representation         | Precision@5 |    MRR |
| ---------------------- | ----------: | -----: |
| RGB thumbnail          |      66.92% | 0.8022 |
| Raw HOG + HSV          |      72.26% | 0.8381 |
| 25% shape / 75% colour |      72.63% | 0.8394 |
| 50% shape / 50% colour |      74.58% | 0.8534 |
| 75% shape / 25% colour |      75.14% | 0.8587 |

The selected 75% shape configuration improves validation precision@5 by 2.88 percentage points over raw HOG + HSV. Test precision@1 is 79.78%, precision@5 is 74.83%, hit rate@5 is 93.30%, and mean reciprocal rank is 0.8581. However, class-macro precision@5 falls to 51.45%, exposing weaker retrieval for less common categories. Recall@5 is only 1.07% because its denominator includes every relevant gallery item; it is not interchangeable with hit rate.

Qualitative results retrieve consistent handbags and briefs but confuse footwear styles and return tops or vests for swimwear. Colour agreement among the top five results is only 25.63%. Optimizing category precision therefore does not guarantee matching colour or fine-grained style.

#### 3.5.2. Real-world Application

The web application demonstrates how the models could support fashion catalogue management and product discovery. Catalogue staff can upload a product image and review suggested article, season, audience and usage labels before accepting them. Shoppers could use the same image-upload workflow to discover visually similar catalogue items without knowing the exact product name. Confidence and review indicators support human checking, while image-based retrieval provides alternatives to inspect. This design is consistent with guidance to communicate AI capabilities and the possibility of mistakes [11]. These are intended uses of a locally tested educational prototype; public deployment and evaluation on independently collected real-world photographs have not been completed.

The Next.js interface calls a FastAPI backend to display four classifications, confidence and review indicators, and five similar catalogue images. It includes persistent local history and a mobile layout. Figure C1 shows the application interface presenting predicted attributes, confidence indicators and visually similar catalogue items. After evaluation, the deployment index includes all 38,611 usable labelled images; its 1,284-dimensional embeddings occupy approximately 189 MiB. Ten warm local CPU searches on one query have a median of 73.1 ms, which is a smoke measurement rather than a load benchmark.

Eight browser uploads verify agreement with direct inference, image loading, history persistence, mobile layout and invalid-file rejection. They also reveal classification failures. With the latest article model, a T-shirt is labelled Sweatshirts at approximately 65% confidence; its darkened version remains incorrect at approximately 82%. Review indicators identify some low-confidence or lighting-sensitive outputs but do not repair their labels. These catalogue-image checks, including retrieval self-matches, demonstrate integration rather than independent generalization.

## 4. Comparison with research and ultimate judgment

DeepFashion investigates the same broad goals of clothing recognition and retrieval. FashionNet jointly learns attributes and landmarks, and reports 82.58% top-three category accuracy across 50 categories [1, Table 2]. Our article score is 80.27% top-one accuracy with 124 usable categories. Different datasets, supervision and ranking metrics prevent a valid numerical superiority claim. The methodological contrast is useful: our compact image-only features require fewer annotations but lack explicit garment localization.

DeepFashion2 combines clothing detection, landmarks, segmentation and consumer-to-shop retrieval [2]. Its product-matching setting differs from our same-category relevance proxy. Consequently, our 93.30% hit rate@5 should not be interpreted as equivalent consumer-to-shop retrieval performance. These differences suggest that clutter, occlusion and viewpoint changes deserve separate evaluation before broader use.

The implemented system is a useful educational prototype and catalogue-assistance tool. Separate model selection improves the balance between simplicity and performance, and the web interface makes errors inspectable. It is not yet supported as an unattended production classifier: usage has weak macro F1, rare categories remain difficult, and confident browser mistakes persist. Audience predictions represent catalogue labels rather than a determination of a person's identity.

The final choices reflect different error priorities. For article type and usage, stronger baseline macro F1 justified logistic regression despite the CNN's higher baseline accuracy. Season and audience benefited from learned CNN features on both initial validation metrics. The latest article update accepts a small ECE increase for observed accuracy and F1 gains, but its bootstrap intervals include zero; it should be treated as a modest candidate improvement rather than a proven advance. Similarly, the audience accuracy gain does not establish better minority-class recognition because test macro F1 decreases slightly. These outcomes support selecting models against explicit priorities rather than declaring one algorithm universally best.

The literature comparison provides external context, but no independently collected image set has been evaluated. Grouped splitting reduces duplicate leakage, yet repeated inspection during development limits the independence of the reported test results. Classes absent from test cannot be assessed, and estimates based on only a few examples are unstable. Catalogue self-matches in the application also provide no evidence of performance on unseen consumer photographs.

A next assessment should freeze all checkpoints, temperatures and review thresholds before collecting a separate evaluation set. It should cover common and uncommon categories, different backgrounds, lighting and viewpoints, with documented label mapping and ambiguous labels recorded explicitly. Report class support, per-class F1, accuracy and calibration, together with the proportion of predictions flagged for review and accuracy among unflagged predictions. Retrieval should additionally receive human similarity judgments, distinguishing category agreement from colour, style and product-identity matches. This assessment remains proposed work, not a completed experiment.

For improvement, targeted training-only augmentation and limited regularization or scheduler changes are preferable first steps to increasing architecture complexity. The dataset's educational-use restriction also limits deployment beyond this assignment.

<!-- Main report ends here. Insert a page break before references. -->

## References

[1] Z. Liu, P. Luo, S. Qiu, X. Wang, and X. Tang, "DeepFashion: Powering robust clothes recognition and retrieval with rich annotations," in _Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)_, 2016, pp. 1096–1104. [Online]. Available: [Paper](https://openaccess.thecvf.com/content_cvpr_2016/html/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.html).

[2] Y. Ge, R. Zhang, X. Wang, X. Tang, and P. Luo, "DeepFashion2: A versatile benchmark for detection, pose estimation, segmentation and re-identification of clothing images," in _Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)_, 2019. [Online]. Available: [Paper](https://arxiv.org/abs/1901.07973).

[3] RMIT University, "COSC2753–3013 Machine Learning: Assignment 2—2026B," assignment specification and supplied fashion dataset, course materials, 2026.

[4] J. Opitz and S. Burst, "Macro F1 and macro F1," _arXiv preprint arXiv:1911.03347_, 2019. [Online]. Available: [Paper](https://arxiv.org/abs/1911.03347).

[5] scikit-learn developers, "Common pitfalls and recommended practices," _scikit-learn documentation_, sec. 12.2, "Data leakage." Accessed: Sep. 9, 2026. [Online]. Available: [Documentation](https://scikit-learn.org/stable/common_pitfalls.html).

[6] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in _Proc. 34th Int. Conf. Machine Learning_, vol. 70, 2017, pp. 1321–1330. [Online]. Available: [Paper](https://proceedings.mlr.press/v70/guo17a.html).

[7] N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection," in _Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)_, vol. 1, 2005, pp. 886–893. [Online]. Available: [Paper](https://www.cs.princeton.edu/courses/archive/fall13/cos429/papers/Dalal05.pdf).

[8] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in _Proc. Int. Conf. Learning Representations (ICLR)_, 2015. [Online]. Available: [Paper](https://arxiv.org/abs/1412.6980).

[9] PyTorch contributors, "ReduceLROnPlateau," _PyTorch 2.8 documentation_. Accessed: Sep. 9, 2026. [Online]. Available: [Documentation](https://docs.pytorch.org/docs/2.8/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html).

[10] C. D. Manning, P. Raghavan, and H. Schütze, _Introduction to Information Retrieval_. Cambridge, U.K.: Cambridge University Press, 2008. [Online]. Available: [Book](https://nlp.stanford.edu/IR-book/information-retrieval-book.html).

[11] S. Amershi _et al._, "Guidelines for human-AI interaction," in _Proc. CHI Conf. Human Factors in Computing Systems_, 2019, pp. 1–13. [Online]. Available: [Paper](https://www.microsoft.com/en-us/research/publication/guidelines-for-human-ai-interaction/).

<!-- Supporting appendices begin below. Keep the combined appendices within two pages. -->

<!-- Figure references in the main text follow the proposed Word layout: A1 = target distributions; B1 = article confusion; B2 = season confusion; B3 = audience confusion; B4 = usage confusion; C1 = web application screenshot. Match these labels to your final Word captions. The supporting tables below predate your figure layout and can be replaced with the figures you have already arranged; appendix letters do not require separate pages. -->

## Appendix A. Supporting data and probability metrics

**Table A1. Dataset audit and evaluation support.**

| Item                                            |                  Count |
| ----------------------------------------------- | ---------------------: |
| Original labelled CSV rows                      |                 38,617 |
| Missing images / undecodable available images   |                  5 / 1 |
| Usable labelled images                          |                 38,611 |
| Exact-duplicate groups / involved rows          |            636 / 1,399 |
| Training / validation / test images             | 27,051 / 5,842 / 5,718 |
| Article / audience / usage test examples        |             5,718 each |
| Season test examples with valid labels          |                  5,715 |
| Article classes in training / supported in test |              124 / 101 |
| Unlabelled prediction rows                      |                  5,829 |

**Table A2. Current classifier probability metrics. Lower is better for both columns.**

| Target       |    NLL | Brier score |
| ------------ | -----: | ----------: |
| Article type | 0.6258 |      0.2755 |
| Season       | 0.7050 |      0.3891 |
| Audience     | 0.3401 |      0.1808 |
| Usage        | 0.4364 |      0.2371 |

**Table A3. Group-bootstrap intervals for the latest article replacement minus its predecessor, in absolute 0–1 metric units.**

| Metric change |   Recorded interval |
| ------------- | ------------------: |
| Accuracy      | [−0.00243, 0.00571] |
| Macro F1      | [−0.00747, 0.02107] |
| ECE           | [−0.00553, 0.01085] |

## Appendix B. Supporting retrieval results

**Table B1. Test retrieval metrics with the training-only gallery.**

| Metric                  |  Value |
| ----------------------- | -----: |
| Precision@1             | 0.7978 |
| Precision@5             | 0.7483 |
| Precision@10            | 0.7192 |
| Hit rate@5              | 0.9330 |
| Hit rate@10             | 0.9591 |
| Recall@5                | 0.0107 |
| Mean reciprocal rank    | 0.8581 |
| Class-macro precision@5 | 0.5145 |

**Table B2. Agreement between query attributes and the five retrieved items.**

| Attribute    | Agreement |
| ------------ | --------: |
| Article type |    74.83% |
| Subcategory  |    92.16% |
| Audience     |    83.92% |
| Usage        |    84.72% |
| Base colour  |    25.63% |

These tables summarize category-based retrieval evaluation; attribute agreement is not classifier accuracy.
