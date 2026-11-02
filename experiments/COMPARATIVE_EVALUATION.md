# Multi-Domain Comparative Experiment Analysis

**Project Title:** Fake Review Detection and Review Reliability Analysis System  
**Project Team:** Himanshi (SID: 24106002), Samarth (SID: 24106046) | **Branch:** CSE (Data Science)  
**Document:** Multi-Domain Cross-Experiment Academic Comparative Evaluation  
**Status:** All 3 Domain Experiments Completed & Evaluated on Held-Out Test Partitions  

---

## 1. Cross-Domain Comparative Summary Table

The table below summarizes the architectural configurations, dataset provenance, and empirical test metrics across all three independent task experts.

| Feature / Metric | Experiment A: Hotel Reviews | Experiment B: Product Reviews | Experiment C: Restaurant Reviews |
| :--- | :--- | :--- | :--- |
| **Domain** | Hotel | E-Commerce Products (10 Categories) | Restaurants |
| **Corpus / Source** | Ott Deceptive Opinion Spam Corpus (Ott et al., 2011, 2013) | Amazon Reviews 40k (Salminen et al., 2022) | YelpChi Restaurant Dataset (Rayana & Akoglu, KDD 2015) |
| **Prediction Task** | Human Truthful vs Human Crowdsourced Deceptive | Human-Written (`OR`) vs GPT-2 Generated (`CG`) | Platform Recommended (`N`) vs Filtered (`Y`) |
| **Total Corpus Size** | 1,600 reviews | 40,432 reviews | 26,955 reviews |
| **Train Samples (80%)** | 1,280 (640 Truthful / 640 Deceptive) | 32,345 (16,172 OR / 16,173 CG) | 21,565 (16,601 N / 4,964 Y) |
| **Test Samples (20%)** | 320 (160 Truthful / 160 Deceptive) | 8,087 (4,044 OR / 4,043 CG) | 5,390 (4,150 N / 1,240 Y) |
| **Class Distribution** | Balanced (50.0% / 50.0%) | Balanced (50.0% / 50.0%) | Natural Imbalance (76.98% N / 23.02% Y) |
| **Feature Extractor** | TF-IDF (1-2 ngrams, min_df=2, sublinear TF) | TF-IDF (1-2 ngrams, min_df=2, sublinear TF) | TF-IDF (1-2 ngrams, min_df=2, sublinear TF) |
| **Vocabulary Size** | 23,047 features | 139,582 features | 196,654 features |
| **Classifier** | Logistic Regression (L2, $C=1.0$, seed 42) | Logistic Regression (L2, $C=1.0$, seed 42) | Logistic Regression (L2, $C=1.0$, seed 42) |
| **Overall Accuracy** | **89.69%** | **93.36%** | **77.79%** |
| **Balanced Accuracy** | **89.69%** | **93.36%** | **53.88%** |
| **Macro Precision** | **89.70%** | **93.41%** | **69.72%** |
| **Macro Recall** | **89.69%** | **93.36%** | **53.88%** |
| **Macro F1-Score** | **89.69%** | **93.36%** | **51.89%** |
| **Weighted F1-Score** | **89.69%** | **93.36%** | **70.95%** |
| **Primary Positive Class Metric** | Deceptive F1: **0.8959** | CG (AI) F1: **0.9325** | Filtered (Y) F1: **0.1659** |
| **Artifact Path** | `experiments/ott_hotel/ott_hotel_model.joblib` | `experiments/amazon_product/amazon_product_model.joblib` | `experiments/yelp_restaurant/yelp_restaurant_model.joblib` |

---

## 2. Confusion Matrices Comparison

### Experiment A: Ott Hotel ($N=320$)
*Positive Class: Deceptive | Negative Class: Truthful*
```
                       Predicted Truthful      Predicted Deceptive
Actual Truthful              145 (TN)                15 (FP)
Actual Deceptive              18 (FN)               142 (TP)
```
- Specificity (Truthful Recall): 90.62%
- Sensitivity (Deceptive Recall): 88.75%

### Experiment B: Amazon Product ($N=8,087$)
*Positive Class: GPT-2 (`CG`) | Negative Class: Human Original (`OR`)*
```
                       Predicted Human (OR)      Predicted GPT-2 (CG)
Actual Human (OR)            3,842 (TN)                  202 (FP)
Actual GPT-2 (CG)              335 (FN)                3,708 (TP)
```
- Specificity (Human Recall): 95.00%
- Sensitivity (AI Recall): 91.71%

### Experiment C: Yelp Restaurant ($N=5,390$)
*Positive Class: Filtered (`Y`) | Negative Class: Recommended (`N`)*
```
                           Predicted Recommended (N)      Predicted Filtered (Y)
Actual Recommended (N)             4,074 (TN)                     76 (FP)
Actual Filtered (Y)                1,121 (FN)                    119 (TP)
```
- Specificity (Recommended Recall): 98.17%
- Sensitivity (Filtered Recall): 9.60% (due to natural class imbalance and text-only limitation)

---

## 3. Why the Three Tasks are Fundamentally Different

A frequent misconception in commercial fraud detection is treating "fake reviews" as a single homogeneous problem. Our experimental findings demonstrate that the underlying ground-truth constructs in these three domains represent distinct classification tasks:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       THREE DISTINCT LEARNING CONSTRUCTS                    │
├────────────────────────┬──────────────────────────┬─────────────────────────┤
│    Ott Hotel Corpus    │      Amazon 40K Corpus   │    YelpChi Restaurant   │
│ (Psychological Deception) (Machine Generation)    │  (Algorithmic Curation) │
├────────────────────────┼──────────────────────────┼─────────────────────────┤
│ Human crowdsourced     │ Human original reviews   │ Platform automated      │
│ imaginative fabrications│ vs fine-tuned GPT-2 text │ recommendation filter   │
│ vs truthful TripAdvisor│                          │ decision                │
│ reviews                │                          │                         │
│                        │                          │                         │
│ Target: Deceptive lies │ Target: AI vs Human text │ Target: Platform action │
└────────────────────────┴──────────────────────────┴─────────────────────────┘
```

1. **Ott Hotel Measures Human Psychological Deception:**
   - The fake reviews were commissioned through Amazon Mechanical Turk under controlled conditions where workers were explicitly instructed to write fake reviews for hotels they never visited.
   - The hotel model evaluates this benchmark using the prediction labels **Truthful** and **Deceptive**.
   - Benchmark limitations apply: it is trained on 1,600 reviews across 20 luxury Chicago hotels (Ott et al., 2011, 2013) and evaluates crowdsourced deception in hotel reviews rather than general product or restaurant reviews.

2. **Amazon 40k Measures Machine-Generated vs Human Text:**
   - The model is implemented solely as a TF-IDF Vectorizer + Logistic Regression pipeline, distinguishing human-written original reviews (`OR`) from fine-tuned GPT-2-generated synthetic text (`CG`) (Salminen et al., 2022).
   - The implementation does not calculate transformer perplexity, does not use a transformer at inference, and does not independently run a repetitive n-gram detection algorithm.
   - It distinguishes human writing style from early GPT-2 generation patterns in the dataset. It does **not** evaluate whether the author had malicious commercial intent: `OR` indicates human authorship (not verified honesty), and `CG` indicates synthetic text (not confirmed fraud). A customer using AI to summarize a real purchase triggers `CG`, while a paid human fabricator will be classified as `OR`.

3. **Yelp Restaurant Measures Platform Filtering Decisions:**
   - Labels represent Yelp's platform recommendation filter: Recommended (`N`) vs Filtered / Not Recommended (`Y`).
   - The dataset exhibits natural class imbalance (76.98% Recommended vs 23.02% Filtered). The linear text model achieves 77.79% overall accuracy and 98.17% recall on Recommended reviews, but catches only 9.60% of Filtered reviews (yielding Balanced Accuracy: 53.88% and Macro F1: 51.89%).
   - Crucially, neither label confirms genuine or fake reviews: Filtered (`Y`) does not confirm fake or fraudulent reviews, and Recommended (`N`) does not confirm genuine reviews. Platform filtering is an automated curation action, not a verification of authenticity.

---

## 4. Methodological Guidelines: Avoiding Construct Conflation

> [!WARNING]
> **Strict Academic Rule Against Monolithic Merging:**  
> These three datasets must **NEVER** be merged into a single generic 0/1 binary classifier. Merging them would create severe construct conflation, corrupting the feature weights and invalidating the statistical guarantees of the classifier.

- **Do not compare accuracies directly:** Amazon's 93.36% accuracy does not mean it is "better at detecting fake reviews" than Ott's 89.69% or Yelp's 77.79%. Each score reflects the separability of that specific dataset's classes under a linear boundary.
- **Independent Task Experts:** The system architecture maintains three segregated, dedicated pipelines:
  - `Hotel Review Expert` $\rightarrow$ Evaluates psychological deception.
  - `Product Review Expert` $\rightarrow$ Evaluates machine-generated synthetic text.
  - `Restaurant Review Expert` $\rightarrow$ Evaluates platform recommendation filter propensity.
