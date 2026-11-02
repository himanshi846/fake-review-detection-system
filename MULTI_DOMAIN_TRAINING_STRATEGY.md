# Multi-Domain Training & Evaluation Strategy

**Project:** Fake Review Detection and Review Reliability Analysis System  
**Document:** `MULTI_DOMAIN_TRAINING_STRATEGY.md`  
**Phase:** Dataset Preparation & Experimental Strategy Audit  
**Date:** October 2026  
**Status:** Pre-Training Specification (Audited & Verified)

---

## 1. Executive Summary & Audit State

This document establishes the scientific training and evaluation framework for expanding the **ReviewTrust** system across multiple review domains:
1. **Hotels & Hospitality**
2. **Products & E-Commerce** (10 product categories)
3. **Restaurants & Dining**
4. **Consumer Services**
5. **Mobile & Web Applications**
6. **Other Commercial Businesses**

### Explicit System State Confirmation:
- **Zero Models Trained:** No new models have been fitted or serialized.
- **Application Files Preserved:** [`app.py`](file:///c:/Users/himan/Downloads/Fake_Review_Detection_System/Fake_Review_Detection_System/app.py), [`model_utils.py`](file:///c:/Users/himan/Downloads/Fake_Review_Detection_System/Fake_Review_Detection_System/model_utils.py), and the active baseline [`review_model.joblib`](file:///c:/Users/himan/Downloads/Fake_Review_Detection_System/Fake_Review_Detection_System/review_model.joblib) are completely unchanged.
- **Original Data Preserved:** All raw source files in `data/` and `candidate_datasets/` remain intact.
- **Isolated Staging:** Prepared experimental datasets and partition files reside exclusively in [`prepared_datasets/`](file:///c:/Users/himan/Downloads/Fake_Review_Detection_System/Fake_Review_Detection_System/prepared_datasets/).

---

## 2. Prepared Dataset Inventory & Exact Audit Metrics

Every staged dataset has been verified directly against disk:

| Prepared Dataset | Physical File Path | File Size | Record Count | Available Columns | Label Distribution | Missing Texts | Exact Duplicate Copies | Total Affected Rows |
| :--- | :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: |
| **Ott Hotel** | [`prepared_datasets/prepared_ott_hotel.csv`](file:///c:/Users/himan/Downloads/Fake_Review_Detection_System/Fake_Review_Detection_System/prepared_datasets/prepared_ott_hotel.csv) | 1.46 MB | **1,600** | `review_text`, `original_label`, `domain`, `dataset_source`, `label_nature`, `sentiment_polarity`, `entity_name`, `group_id` | `truthful`: 800 (50.0%)<br>`deceptive`: 800 (50.0%) | 0 | 4 extra copies (4 unique pairs) | 8 rows |
| **Amazon Product** | [`prepared_datasets/prepared_amazon_product.csv`](file:///c:/Users/himan/Downloads/Fake_Review_Detection_System/Fake_Review_Detection_System/prepared_datasets/prepared_amazon_product.csv) | 17.94 MB | **40,432** | `review_text`, `original_label`, `domain`, `dataset_source`, `label_nature`, `product_category`, `star_rating`, `group_id` | `CG`: 20,216 (50.0%)<br>`OR`: 20,216 (50.0%) | 0 | 20 extra copies (20 unique pairs) | 40 rows |
| **Yelp Restaurant** | [`prepared_datasets/prepared_yelp_restaurant.csv`](file:///c:/Users/himan/Downloads/Fake_Review_Detection_System/Fake_Review_Detection_System/prepared_datasets/prepared_yelp_restaurant.csv) | 16.47 MB | **26,955** | `review_text`, `original_label`, `domain`, `dataset_source`, `label_nature`, `star_rating`, `entity_name`, `review_id`, `group_id` | `N`: 20,751 (76.98%)<br>`Y`: 6,204 (23.02%) | 0 (3 raw nulls dropped) | 36 extra copies (23 unique groups) | 59 rows |

---

## 3. Duplicate-Count Reconciliation Analysis

An audit was conducted to resolve duplicate reporting discrepancies across raw and prepared files:

### A. Yelp Duplicate Count Reconciliation (38 vs. 36)
- **Raw File Audit:** In `candidate_datasets/yelp_restaurant_reviews.csv` (26,958 rows), `df.duplicated(subset=['reviewContent']).sum()` reported **38**.
- **Root Cause:** Exactly 3 rows contained `NaN` (missing) reviewContent. In pandas, `df.duplicated()` treats `NaN == NaN` as matching values. The first `NaN` is marked as unique, and the 2nd and 3rd `NaN` are flagged as duplicate occurrences ($+2$).
- **Reconciliation:** The non-null duplicate count in raw Yelp is exactly $38 - 2 = \mathbf{36}$.
- **Prepared File Audit:** In `prepared_datasets/prepared_yelp_restaurant.csv`, the 3 null rows were dropped (leaving 26,955 rows). With nulls eliminated, the clean duplicate record count is **36**.
- **Group Breakdown:** These 36 extra copies belong to **23 unique text groups**, affecting **59 total rows** in the dataset.

### B. Amazon 40k Duplicate Clarification (20 Duplicates)
- In `prepared_amazon_product.csv`, `df.duplicated(subset=['review_text'], keep='first').sum()` equals **20 duplicate records** (extra copies).
- Inspection confirms these correspond to exactly **20 unique duplicate text groups**, each appearing exactly twice (i.e. **20 duplicate pairs**).
- Total affected rows = **40 rows** ($20 \times 2 = 40$). No duplicate text appears more than twice.

### C. Ott Hotel Duplicate Clarification (4 Duplicates)
- In `prepared_ott_hotel.csv`, `df.duplicated(subset=['review_text'], keep='first').sum()` equals **4 duplicate records** (extra copies).
- These correspond to exactly **4 unique duplicate text groups**, each appearing exactly twice (i.e. **4 duplicate pairs**).
- Total affected rows = **8 rows** ($4 \times 2 = 8$). All 4 pairs are truthful negative reviews scraped from TripAdvisor/Web.

---

## 4. Train / Test Split Verification (Zero Leakage Confirmed)

All train and test partitions have been generated and saved to disk using reproducible **Group-Aware Stratified Splitting (`StratifiedGroupKFold`)** with fixed seed `random_state=42`. All identical review texts are guaranteed to reside inside the same partition.

### Split Generation Algorithm:
```python
from sklearn.model_selection import StratifiedGroupKFold

# text_to_group maps identical review texts to an identical integer group_id
text_to_group = {text: idx for idx, text in enumerate(df["review_text"].unique())}
df["group_id"] = df["review_text"].map(text_to_group)

# StratifiedGroupKFold enforces 80/20 partition without cross-partition group leakage
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(sgkf.split(df, df["strat_group"], df["group_id"]))
```

### Empirical Verification of Saved Partition Files:

| Dataset | Partition File | Actual Rows | Share | Label Distribution (%) | Exact Text-Hash Overlap | Audit Status |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: |
| **Ott Hotel** | [`prepared_datasets/ott_hotel_train.csv`](file:///c:/Users/himan/Downloads/Fake_Review_Detection_System/Fake_Review_Detection_System/prepared_datasets/ott_hotel_train.csv) | **1,280** | 80.0% | `truthful`: 50.0% \| `deceptive`: 50.0% | **0** | **VERIFIED (PASSED)** |
| | [`prepared_datasets/ott_hotel_test.csv`](file:///c:/Users/himan/Downloads/Fake_Review_Detection_System/Fake_Review_Detection_System/prepared_datasets/ott_hotel_test.csv) | **320** | 20.0% | `truthful`: 50.0% \| `deceptive`: 50.0% | | |
| **Amazon Product** | [`prepared_datasets/amazon_product_train.csv`](file:///c:/Users/himan/Downloads/Fake_Review_Detection_System/Fake_Review_Detection_System/prepared_datasets/amazon_product_train.csv) | **32,345** | 80.0% | `CG`: 50.0% \| `OR`: 50.0% | **0** | **VERIFIED (PASSED)** |
| | [`prepared_datasets/amazon_product_test.csv`](file:///c:/Users/himan/Downloads/Fake_Review_Detection_System/Fake_Review_Detection_System/prepared_datasets/amazon_product_test.csv) | **8,087** | 20.0% | `CG`: 49.99% \| `OR`: 50.01% | | |
| **Yelp Restaurant** | [`prepared_datasets/yelp_restaurant_train.csv`](file:///c:/Users/himan/Downloads/Fake_Review_Detection_System/Fake_Review_Detection_System/prepared_datasets/yelp_restaurant_train.csv) | **21,565** | 80.0% | `N`: 76.98% \| `Y`: 23.02% | **0** | **VERIFIED (PASSED)** |
| | [`prepared_datasets/yelp_restaurant_test.csv`](file:///c:/Users/himan/Downloads/Fake_Review_Detection_System/Fake_Review_Detection_System/prepared_datasets/yelp_restaurant_test.csv) | **5,390** | 20.0% | `N`: 76.99% \| `Y`: 23.01% | | |

*Summary:* The saved files on disk confirm **0 overlapping review texts** between train and test across all three datasets, with exact label preservation.

---

## 5. Label Provenance & Exact Operational Meaning

To prevent methodological errors, the exact origin and semantics of each label are documented:

### 1. Ott Deceptive Opinion Spam Corpus
* **Original Column:** `deceptive`
* **Values:** `truthful` (800) and `deceptive` (800).
* **Provenance:** Sourced from Ott et al. (ACL 2011, NAACL 2013). Truthful reviews were crawled from verified public listings (TripAdvisor, Expedia, Hotels.com, Orbitz, Priceline, Yelp). Deceptive reviews were created under controlled experimental conditions where Amazon Mechanical Turk crowdworkers were paid to write realistic positive and negative reviews for 20 Chicago hotels they had never visited.

### 2. Amazon Labeled Fake Reviews Dataset (40k)
* **Original Column:** `label`
* **Values:** `OR` (Original Review) and `CG` (Computer-Generated Review).
* **Provenance:** Sourced from Salminen et al. (2022). `OR` reviews are authentic human customer reviews sampled from Amazon across 10 product categories. `CG` reviews were synthetically generated using a fine-tuned GPT-2 language model prompted with product metadata, category, and target rating.
* **Semantic Constraint:** `OR` and `CG` represent **Human-Written vs. Machine-Generated text**, NOT human truth vs. human lies.

### 3. Yelp Restaurant Dataset
* **Original Column:** `flagged`
* **Values:** `N` (Not Flagged) and `Y` (Flagged).
* **Provenance:** Sourced from Rayana & Akoglu (KDD 2015), crawled from Yelp.
* **Source Definition of Filtered Reviews:** Yelp operates proprietary automated recommendation software designed to protect consumers from suspicious or biased content.
  * Reviews tagged as `N` are **Recommended Reviews** displayed on the business profile.
  * Reviews tagged as `Y` are **Filtered / Not Recommended Reviews** relegated to a secondary page.
  * **Critical Distinction:** As Yelp's documentation and academic literature explicitly state, Yelp's filter is an algorithmic heuristic based on user activity, posting velocity, and network signals. **Filtered reviews are NOT confirmed fakes**; legitimate reviews from infrequent or non-networked users are frequently filtered, and sophisticated fake reviews occasionally pass. Therefore, `flagged` represents an **automated platform filter decision**, not factual ground-truth deception.

---

## 6. Training Strategy & Cross-Domain Constraints

### Principle 1: Keep Datasets Strictly Independent
The three datasets must be trained as separate classifiers because their target labels measure fundamentally different phenomena:
1. **Model A (Ott Hotel Expert):** Evaluates **Human Deceptive Opinion Spam** (Experimental crowdworker deception).
2. **Model B (Amazon Product Expert):** Evaluates **AI-Generated vs. Human-Written Text** (Synthetic GPT-2 detection).
3. **Model C (Yelp Restaurant Expert):** Evaluates **Platform Recommendation Status** (Predicting Yelp filter decisions under class imbalance).

### Principle 2: Cross-Domain Prediction vs. Cross-Domain Accuracy
* **Exploratory Predictions Are Valid:** It is scientifically valid to run a model trained on Amazon products on hotel reviews to observe which features trigger prediction shifts.
* **Calculating "Accuracy" Across Incompatible Labels Is Invalid:**
  > [!WARNING]  
  > If Model B (trained on `OR` vs `CG`) is evaluated on the Ott test set (`truthful` vs `deceptive`), one **cannot** calculate accuracy, precision, or recall. A human deceptive review written by an MTurk crowdworker is human-authored (it belongs to the `OR` category in distribution, not `CG`). Scoring human lies as "True Positives" or "False Positives" against a synthetic generator detector is mathematically meaningless. Cross-domain transfer can only be analyzed qualitatively (score distributions and feature attribution shifts), not as standard classification accuracy.

---

## 7. Domain Coverage & Explicit Project Limitations

| Domain | Best Available Benchmark in Project | Staged Status | Label Nature | Project Coverage Status |
| :--- | :--- | :---: | :--- | :--- |
| **1. Hotels** | Ott Deceptive Opinion Spam | `prepared_ott_hotel.csv` | Crowdsourced Experimental | **Fully Covered** (Benchmark standard) |
| **2. Products** | Amazon 40k Product Dataset | `prepared_amazon_product.csv` | Synthetic AI vs Human | **Fully Covered** (10 categories, 40k rows) |
| **3. Restaurants** | Yelp Restaurant Dataset | `prepared_yelp_restaurant.csv` | Platform Filter Proxy | **Fully Covered** (26.9k rows) |
| **4. Services** | YelpZip Services Subset | Documented | Platform Filter Proxy | **Partially Covered** (Yelp filter proxy) |
| **5. Mobile Apps** | None acquired | None | None | **LACKS SUITABLE DATASET:** No suitable verified labelled app-review dataset has been acquired for this project. |
| **6. Other Businesses** | Yelp Venue/Business Crawl | Documented | Platform Filter Proxy | **Partially Covered** (Yelp filter proxy) |

---

## 8. Current System State & Stop Confirmation

- **No models have been trained on the prepared datasets.**
- **Existing single-domain model ([`review_model.joblib`](file:///c:/Users/himan/Downloads/Fake_Review_Detection_System/Fake_Review_Detection_System/review_model.joblib)) is active and intact.**
- **Existing frontend ([`app.py`](file:///c:/Users/himan/Downloads/Fake_Review_Detection_System/Fake_Review_Detection_System/app.py)) and backend ([`model_utils.py`](file:///c:/Users/himan/Downloads/Fake_Review_Detection_System/Fake_Review_Detection_System/model_utils.py)) are unchanged.**
- All persistent train/test split files are generated and ready in [`prepared_datasets/`](file:///c:/Users/himan/Downloads/Fake_Review_Detection_System/Fake_Review_Detection_System/prepared_datasets/).

*Awaiting your approval before executing any training scripts.*
