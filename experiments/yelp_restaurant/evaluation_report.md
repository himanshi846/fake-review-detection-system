# Phase 3: Yelp Restaurant Recommendation Filter Classification Report

**Model Name:** Yelp Restaurant Recommendation Filter Model  
**Artifact Directory:** `experiments/yelp_restaurant/`  
**Model Path:** `experiments/yelp_restaurant/yelp_restaurant_model.joblib`  
**Evaluation Date & Time:** 2026-10-30T13:27:36.685184  
**Reproducibility:** Seed 42, deterministic TF-IDF + Logistic Regression  

> [!IMPORTANT]
> **Construct & Label Provenance Notice:**  
> This experiment models Yelp's **algorithmic recommendation filter decision**:  
> - **`N`** = **Recommended Review** (Displayed prominently by Yelp)  
> - **`Y`** = **Filtered / Not Recommended Review** (Segregated into Yelp's hidden filtered section)  
> **Yelp's filtered label is an automated platform decision and does NOT prove human deception.**  
> Similarly, a recommended review (`N`) is not guaranteed to be truthful. This model classifies **platform filtering tendencies**, not confirmed fraud.

---

## 1. Executive Performance Summary

The model was fitted strictly on `yelp_restaurant_train.csv` (21,565 reviews) and evaluated once on the untouched held-out `yelp_restaurant_test.csv` (5,390 reviews).

| Metric | Score | Detail |
| :--- | :---: | :--- |
| **Accuracy** | **77.79%** (0.7779) | 4,193 out of 5,390 correct classifications |
| **Balanced Accuracy** | **53.88%** (0.5388) | Unweighted average of per-class recall |
| **Macro Precision** | **69.72%** (0.6972) | Average precision across classes |
| **Macro Recall** | **53.88%** (0.5388) | Average recall across classes |
| **Macro F1-Score** | **51.89%** (0.5189) | Harmonic mean of precision and recall |
| **Weighted F1-Score** | **70.95%** (0.7095) | Support-weighted F1 reflecting natural class imbalance |

---

## 2. Dataset & Training Specifications

- **Total Corpus Size:** 26,955 restaurant reviews (Rayana & Akoglu, KDD 2015)
- **Train Partition:** 21,565 reviews (80.0%), exactly 16,601 N (76.98%) / 4,964 Y (23.02%)
- **Held-Out Test Partition:** 5,390 reviews (20.0%), exactly 4,150 N (76.99%) / 1,240 Y (23.01%)
- **Exact Text-Hash Overlap:** **0 reviews** (Guaranteed zero data leakage)
- **Duplicate Clustering:** All duplicate text groups strictly co-located inside the same partition.

### Pipeline Configuration
1. **Feature Extractor:** `TfidfVectorizer`
   - `ngram_range`: (1, 2) [unigrams and bigrams]
   - `min_df`: 2
   - `sublinear_tf`: True ($1 + \log(\text{tf})$)
   - `strip_accents`: 'unicode'
   - `lowercase`: True
   - **Vocabulary Size:** 196,654 features
2. **Classifier:** `LogisticRegression`
   - `C`: 1.0 (L2 regularisation)
   - `max_iter`: 1000
   - `random_state`: 42
   - `solver`: 'lbfgs'

---

## 3. Confusion Matrix Breakdown

*Evaluation conducted on N=5,390 held-out test samples*

```
                           Predicted Recommended (N)      Predicted Filtered (Y)
Actual Recommended (N)             4,074 (TN)                    76 (FP)
Actual Filtered (Y)                1,121 (FN)                    119 (TP)
```

- **True Negatives (TN):** **4,074** / 4,150 recommended reviews correctly identified as `N` (98.17% specificity)
- **False Positives (FP):** **76** / 4,150 recommended reviews incorrectly classified as `Y` (1.83% false alarm rate)
- **False Negatives (FN):** **1,121** / 1,240 filtered reviews incorrectly classified as `N` (90.40% miss rate)
- **True Positives (TP):** **119** / 1,240 filtered reviews correctly identified as `Y` (9.60% sensitivity)

---

## 4. Per-Class Classification Report

| Class | Label Meaning | Precision | Recall | F1-Score | Test Support |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **N** | Recommended review | 0.7842 | 0.9817 | 0.8719 | 4,150 |
| **Y** | Filtered / Not Recommended review | 0.6103 | 0.0960 | 0.1659 | 1,240 |
| **Macro Avg** | Unweighted mean | 0.6972 | 0.5388 | 0.5189 | 5,390 |
| **Weighted Avg** | Support-weighted mean | 0.7442 | 0.7779 | 0.7095 | 5,390 |

---

## 5. Class Imbalance and Methodological Limitations

1. **Natural Real-World Class Imbalance:** Unlike Ott and Amazon (which are artificially balanced 50/50), Yelp data exhibits a real-world ~77% recommended vs ~23% filtered distribution. Standard logistic regression without threshold adjustment prioritizes overall accuracy (77.79%), leading to high recall on the majority class `N` (98.17%) but more conservative recall on the minority class `Y` (9.60%).
2. **Text-Only Modeling of Multi-Modal Filtering:** Yelp's proprietary recommendation software uses user behavioral metadata (reviewer activity, account age, friend network, IP history, review velocity) alongside text. A pure n-gram NLP model can only capture linguistic cues associated with filtered text (e.g. extreme brevity, generic praise, aggressive complaints), explaining why text-alone performance is lower than on crowdsourced or synthetic corpora.
3. **No Direct Ground Truth:** Many filtered Yelp reviews are genuine customer opinions from infrequent users, while some recommended reviews may be subtle paid promotions.

---

## 6. Artifacts Inventory

- [x] Trained Pipeline: `experiments/yelp_restaurant/yelp_restaurant_model.joblib`
- [x] Predictions CSV: `experiments/yelp_restaurant/test_predictions.csv`
- [x] High-Res Visualization: `experiments/yelp_restaurant/confusion_matrix.png`
- [x] Configuration JSON: `experiments/yelp_restaurant/training_config.json`
- [x] Evaluation Report: `experiments/yelp_restaurant/evaluation_report.md`
