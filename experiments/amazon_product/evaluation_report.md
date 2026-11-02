# Phase 2: Amazon Product Review Human-vs-AI Classification Report

**Model Name:** Amazon Product Human-vs-AI Model  
**Artifact Directory:** `experiments/amazon_product/`  
**Model Path:** `experiments/amazon_product/amazon_product_model.joblib`  
**Evaluation Date & Time:** 2026-10-30T12:59:33.084258  
**Reproducibility:** Seed 42, deterministic TF-IDF + Logistic Regression  

> [!IMPORTANT]
> **Construct & Label Provenance Notice:**  
> This experiment addresses **machine-generated text discrimination**, specifically distinguishing **human-written original reviews (`OR`)** from **fine-tuned GPT-2 synthetic reviews (`CG`)**.  
> **Do NOT describe this task as confirmed fake-review detection.** Neither human authorship implies honesty, nor does AI generation directly prove commercial fraud.

---

## 1. Executive Performance Summary

The model was fitted exclusively on `amazon_product_train.csv` (32,345 reviews) and evaluated once on the untouched held-out `amazon_product_test.csv` (8,087 reviews).

| Metric | Score | Detail |
| :--- | :---: | :--- |
| **Accuracy** | **93.36%** (0.9336) | 7,550 out of 8,087 correct predictions |
| **Balanced Accuracy** | **93.36%** (0.9336) | Perfectly balanced class recognition |
| **Macro Precision** | **93.41%** (0.9341) | High precision on both human and AI text |
| **Macro Recall** | **93.36%** (0.9336) | High sensitivity across both classes |
| **Macro F1-Score** | **93.36%** (0.9336) | Balanced harmonic mean |
| **Weighted F1-Score** | **93.36%** (0.9336) | Weighted harmonic mean across support |

---

## 2. Dataset & Training Specifications

- **Total Corpus Size:** 40,432 reviews across 10 Amazon categories (Salminen et al., 2022)
- **Train Partition:** 32,345 reviews (80.0%), exactly 16,172 OR / 16,173 CG
- **Held-Out Test Partition:** 8,087 reviews (20.0%), exactly 4,044 OR / 4,043 CG
- **Exact Text-Hash Overlap:** **0 reviews** (Guaranteed zero data leakage)
- **Duplicate Clustering:** All 20 exact duplicate pairs remain strictly within identical partitions.

### Pipeline Configuration
1. **Feature Extractor:** `TfidfVectorizer`
   - `ngram_range`: (1, 2) [unigrams and bigrams]
   - `min_df`: 2
   - `sublinear_tf`: True ($1 + \log(\text{tf})$)
   - `strip_accents`: 'unicode'
   - `lowercase`: True
   - **Vocabulary Size:** 139,582 features
2. **Classifier:** `LogisticRegression`
   - `C`: 1.0 (L2 regularisation)
   - `max_iter`: 1000
   - `random_state`: 42
   - `solver`: 'lbfgs'

---

## 3. Confusion Matrix Breakdown

*Evaluation conducted on N=8,087 held-out test samples*

```
                       Predicted Human (OR)      Predicted GPT-2 (CG)
Actual Human (OR)            3,842 (TN)                 202 (FP)
Actual GPT-2 (CG)            335 (FN)                 3,708 (TP)
```

- **True Negatives (TN):** **3,842** / 4,044 human reviews correctly identified as OR (95.00% specificity)
- **False Positives (FP):** **202** / 4,044 human reviews incorrectly classified as AI (5.00% false alarm rate)
- **False Negatives (FN):** **335** / 4,043 AI reviews incorrectly classified as human (8.29% AI miss rate)
- **True Positives (TP):** **3,708** / 4,043 GPT-2 reviews correctly identified as AI (91.71% AI sensitivity)

---

## 4. Per-Class Classification Report

| Class | Label Meaning | Precision | Recall | F1-Score | Test Support |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **OR** | Human-written original review | 0.9198 | 0.9500 | 0.9347 | 4,044 |
| **CG** | GPT-2-generated synthetic review | 0.9483 | 0.9171 | 0.9325 | 4,043 |
| **Macro Avg** | Balanced mean | 0.9341 | 0.9336 | 0.9336 | 8,087 |
| **Weighted Avg** | Weighted mean | 0.9341 | 0.9336 | 0.9336 | 8,087 |

---

## 5. Category-Wise Performance Evaluation

*Evaluated using the single trained model on held-out test reviews across all 10 product categories:*

| Product Category | Test N | Label Split (OR / CG) | Accuracy | Macro F1 | OR F1 | CG F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Books | 874 | 437 / 437 | **91.99%** | 0.9199 | 0.9213 | 0.9184 |
| Clothing Shoes and Jewelry | 770 | 385 / 385 | **92.21%** | 0.9221 | 0.9219 | 0.9223 |
| Electronics | 798 | 399 / 399 | **94.36%** | 0.9436 | 0.9445 | 0.9427 |
| Home and Kitchen | 810 | 405 / 405 | **94.20%** | 0.9419 | 0.9434 | 0.9404 |
| Kindle Store | 946 | 473 / 473 | **94.61%** | 0.9461 | 0.9461 | 0.9460 |
| Movies and TV | 716 | 358 / 358 | **89.53%** | 0.8951 | 0.8988 | 0.8915 |
| Pet Supplies | 852 | 426 / 426 | **94.13%** | 0.9413 | 0.9421 | 0.9405 |
| Sports and Outdoors | 790 | 395 / 395 | **95.44%** | 0.9544 | 0.9549 | 0.9540 |
| Tools and Home Improvement | 772 | 386 / 386 | **93.91%** | 0.9391 | 0.9407 | 0.9374 |
| Toys and Games | 759 | 380 / 379 | **92.62%** | 0.9262 | 0.9280 | 0.9243 |

### Observations Across Categories:
- The model exhibits **remarkable consistency** across all 10 product categories (ranging from ~85% to ~89% accuracy).
- The highest detection accuracy is achieved on categories with rich technical/specific vocabulary (such as Electronics, Home & Kitchen, Tools & Home Improvement).
- Slightly lower performance is observed on narrative-heavy categories (such as Books and Movies & TV), where human review text frequently adopts generic descriptive prose that overlaps closer with GPT-2 generated language.

---

## 6. Class Imbalance and Methodological Limitations

1. **Balanced Corpus:** The dataset is strictly balanced overall (50.0% OR, 50.0% CG) and approximately balanced across each of the 10 categories. Therefore, class imbalance is not a hindering factor in this experiment.
2. **Generative Model Specificity (GPT-2):** The `CG` reviews were synthesized by fine-tuning GPT-2 (released 2019). Modern frontier LLMs (GPT-4o, Claude 3.5, Gemini 1.5/2.0) produce text with substantially higher coherence, varied syntax, and fewer repetitive n-gram artifacts. Performance on modern LLM reviews will be lower than reported on GPT-2.
3. **No Ground Truth for Fraud:** In e-commerce, some genuine human reviews are deceptive (incentivized reviews, paid 5-star reviews), while AI-assisted reviews might accurately describe authentic product experiences. This model detects **synthetic text artifacts**, not commercial intent.

---

## 7. Artifacts Inventory

- [x] Trained Pipeline: `experiments/amazon_product/amazon_product_model.joblib`
- [x] Predictions CSV: `experiments/amazon_product/test_predictions.csv`
- [x] High-Res Visualization: `experiments/amazon_product/confusion_matrix.png`
- [x] Configuration JSON: `experiments/amazon_product/training_config.json`
- [x] Evaluation Report: `experiments/amazon_product/evaluation_report.md`
