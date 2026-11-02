# Academic Minor Project Report

# Fake Review Detection and Review Reliability Analysis System

**Institution:** Punjab Engineering College (PEC), Chandigarh  
**Course:** B.Tech Minor Project &bull; 5th / 6th Semester  
**Project Team / Authors:**  
1. **Himanshi** — SID: 24106002  
2. **Samarth** — SID: 24106046  
**Branch:** Computer Science & Engineering (Data Science)  
**Academic Year:** 2026  

---

## Abstract

Online reviews significantly sway consumer purchasing decisions across e-commerce, hospitality, and dining services. However, consumer trust is heavily degraded by deceptive opinion spam, AI-generated synthetic reviews, and platform filtering ambiguities. Conventional commercial fraud detection tools often treat "fake reviews" as a monolithic construct, leading to flawed models and high error rates when applied across heterogeneous domains.

This academic project presents **ReviewTrust**, an end-to-end, multi-domain Natural Language Processing (NLP) system designed to analyze online reviews across three distinct domains: Hotels, E-Commerce Products, and Restaurants. Rather than creating an invalid universal classifier, ReviewTrust deploys three dedicated, leak-free task experts trained on canonical benchmark datasets: the Ott Deceptive Opinion Spam Corpus (Hotel Reviews, 89.69% test accuracy), the Amazon Reviews 40K Corpus (Product Reviews, 93.36% test accuracy), and the YelpChi Corpus (Restaurant Reviews, 77.79% test accuracy). 

The system incorporates three-tier probabilistic classification to model uncertainty, transparent Explainable AI (XAI) feature attribution, batch CSV analysis, a responsive Streamlit interface, and parameterized SQLite relational database persistence. This report details the problem formulation, theoretical foundations, data curation, empirical evaluations, software architecture, and critical methodological boundaries of the project.

---

## 1. Introduction

Consumer reliance on online reviews has grown exponentially over the past decade. Research across digital commerce indicates that over 88% of consumers consult online reviews prior to making reservations, purchasing products, or choosing dining establishments. However, this economic reliance creates strong financial incentives for malicious actors to fabricate reviews—either to artificially boost a business's reputation (positive deception) or defame commercial rivals (negative deception).

Detecting deceptive text is an inherently challenging computational problem. Unlike traditional spam emails, which often feature bizarre characters, obvious grammatical anomalies, or suspicious hyperlinks, deceptive opinion spam is deliberately written to resemble authentic, high-quality human feedback. Furthermore, the rapid emergence of generative Large Language Models (LLMs) enables malicious syndicates to produce thousands of contextually fluent synthetic reviews at near-zero marginal cost.

This minor project addresses these challenges by developing a robust, scientifically honest review analysis system implemented in Python, Scikit-learn, and Streamlit.

---

## 2. Problem Statement

Current approaches to fake review detection suffer from three fundamental limitations:

1. **Construct Conflation:** Existing commercial detectors frequently aggregate datasets with incompatible label definitions—combining human crowdsourced lies, GPT-generated synthetic text, and platform recommendation heuristics into a single generic 0/1 target. This conflation corrupts feature weights and invalidates statistical guarantees.
2. **Overconfidence & False Certainty:** Standard binary classification forces a definitive 0 or 1 prediction even on reviews where the model's posterior probability hovers around 0.50. Borderline reviews are flagged as fake without communicating statistical uncertainty.
3. **Black-Box Opacity:** Deep learning and proprietary tools fail to provide interpretable explanations of why a review was flagged, eroding user trust during live demonstrations and operational audits.

---

## 3. Project Objectives

The project accomplishes the following engineering and academic objectives:

- **Objective 1 — Multi-Domain Expert Specialization:** Build and validate separate, dedicated machine learning pipelines for hotel reviews, product reviews, and restaurant reviews, keeping incompatible datasets strictly segregated.
- **Objective 2 — Leakage-Free Stratified Group Partitioning:** Enforce rigorous train/test splitting using `StratifiedGroupKFold` so that duplicate review texts never cross between partitions, guaranteeing zero data leakage.
- **Objective 3 — Probabilistic Uncertainty Modeling:** Calibrate three-tier classification boundaries (e.g., *Truthful*, *Uncertain (Borderline)*, *Deceptive*) to communicate model confidence transparently.
- **Objective 4 — Explainable AI (XAI) Attribution:** Implement mathematically grounded feature-attribution algorithms based on TF-IDF weights and Logistic Regression coefficients to display influential tokens in real time.
- **Objective 5 — Full Application Lifecycle:** Build an interactive Streamlit web dashboard supporting both single-review evaluation and high-throughput bulk CSV processing.
- **Objective 6 — Relational Database Persistence:** Integrate a local SQLite database (`reviews.db`) using parameterized SQL queries to persist analysis logs and support analytics.

---

## 4. Background & Theoretical Foundations

### 4.1 Opinion Spam Typology
In academic literature, opinion spam is categorized into three primary types (Jindal & Liu, 2008):
- **Type 1 (Deceptive Reviews):** Untruthful reviews deliberately written to mislead consumers.
- **Type 2 (Reviews on Brands/Sellers Only):** Comments focusing on the manufacturer rather than product functionality.
- **Type 3 (Non-Reviews):** Advertisements, gibberish, or irrelevant questions.

This project focuses on **Type 1** deception, machine-generated synthetic text, and platform filtering behavior.

### 4.2 Psycholinguistic Cues in Deception
Foundational research by Ott et al. (2011, 2013) demonstrated that human deceivers leave distinct psycholinguistic footprints. Deceptive authors cannot rely on actual sensory memories of physical presence. Consequently, deceptive reviews exhibit:
- Significantly higher emotional exaggeration (superlatives).
- Fewer concrete spatial, temporal, and physical details.
- Increased self-referential pronouns (I, my) as deceivers unconsciously attempt to build personal credibility.

---

## 5. Dataset Descriptions

The system utilizes three academically documented benchmark datasets:

| Characteristic | Ott Hotel Corpus | Amazon Reviews 40K | YelpChi Restaurant Corpus |
| :--- | :--- | :--- | :--- |
| **Domain** | Hotel Accommodations | E-Commerce Goods (10 Categories) | Restaurants |
| **Academic Source** | Ott et al. (ACL 2011, NAACL 2013) | Salminen et al. (Expert Systems 2022) | Rayana & Akoglu (KDD 2015) |
| **Target Task** | Human Truthful vs Deceptive | Human Original (`OR`) vs GPT-2 (`CG`) | Platform Recommended (`N`) vs Filtered (`Y`) |
| **Total Size** | 1,600 reviews | 40,432 reviews | 26,955 reviews |
| **Training Records** | 1,280 reviews (80%) | 32,345 reviews (80%) | 21,565 reviews (80%) |
| **Testing Records** | 320 reviews (20%) | 8,087 reviews (20%) | 5,390 reviews (20%) |
| **Label Balance** | 50% Truthful / 50% Deceptive | 50% OR / 50% CG | 76.98% Recommended / 23.02% Filtered |

---

## 6. Dataset Provenance & Ground-Truth Constructs

### 6.1 Hotel Reviews (Ott Corpus)
- **Truthful Reviews (800 reviews):** Mined from TripAdvisor for the 20 most popular 4-star hotels in the Chicago area.
- **Deceptive Reviews (800 reviews):** Commissioned through Amazon Mechanical Turk. Crowdworkers were given the hotel website and instructed to assume the persona of a customer who stayed at the property and write realistic positive or negative reviews.

### 6.2 Product Reviews (Amazon 40K Corpus)
- **Human Original Reviews (`OR`, 20,216 reviews):** Human-written customer reviews across 10 Amazon categories (Books, Clothing, Electronics, Home, Kindle, Movies, Pet Supplies, Sports, Tools, Toys).
- **Computer-Generated Reviews (`CG`, 20,216 reviews):** Generated by fine-tuning GPT-2 on the Amazon review corpus to produce synthetic review text matching the product category and star rating.

### 6.3 Restaurant Reviews (YelpChi Corpus)
- **Recommended Reviews (`N`, 20,751 reviews):** Recommended reviews displayed on Yelp business pages.
- **Filtered Reviews (`Y`, 6,204 reviews):** Filtered / Not Recommended reviews placed in Yelp's filtered section by Yelp's automated filtering algorithm. Crucially, neither label confirms genuine or fake reviews.

---

## 7. Data Preprocessing & Leakage Prevention

Data leakage occurs when information from outside the training dataset is inadvertently used to train the model, resulting in overly optimistic evaluation metrics that fail to generalize.

To guarantee zero data leakage:
1. **Deduplication Audit:** Identical reviews appearing across the raw datasets were grouped into unique clusters (`group_id`).
2. **Stratified Group Splitting:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` was employed. This mathematically ensures that all identical review copies reside in the exact same partition.
3. **Cryptographic Validation:** Pre-training audits verified **0 matching SHA-256 text hashes** between train and test partitions across all three datasets.
4. **Isolated Feature Fitting:** The TF-IDF vectorizers were fitted strictly on the training partitions (`*_train.csv`). The held-out test sets (`*_test.csv`) were transformed strictly during inference.

---

## 8. Methodology

The machine learning architecture follows a supervised classification pipeline:

```
[Raw Review Text] 
       │
       ▼
[Text Preprocessing] (Lowercasing, Accent Stripping, Tokenization)
       │
       ▼
[TF-IDF Feature Extraction] (Unigrams + Bigrams, min_df=2, Sublinear Scaling)
       │
       ▼
[L2 Regularized Logistic Regression] (C=1.0, max_iter=1000)
       │
       ▼
[Class Probabilities P(c0), P(c1)]
       │
       ├──► [3-Tier Uncertainty Stratification] ──► [Predicted Category]
       ├──► [Mathematical Feature Attribution]  ──► [Influential N-Gram Drivers]
       └──► [Parameterized SQLite Persistence]   ──► [Historical Records]
```

---

## 9. TF-IDF Feature Extraction

Term Frequency-Inverse Document Frequency (TF-IDF) reflects how important a word or n-gram is to a document within a corpus:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

1. **Sublinear Term Frequency Scaling:**
   $$\text{TF}(t, d) = 1 + \log(\text{tf}(t, d)) \quad \text{for } \text{tf}(t, d) > 0$$
   This prevents frequent terms from dominating the feature space linearly.
2. **Inverse Document Frequency:**
   $$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + \text{DF}(t, D)}\right) + 1$$
3. **N-Gram Tokenization:** Extracting both unigrams and bigrams (`ngram_range=(1, 2)`) preserves critical local phrases (e.g., *"front desk"*, *"highly recommend"*, *"poor service"*).
4. **Vocabulary Sizes:** Hotel ($V=23,047$), Product ($V=139,582$), Restaurant ($V=196,654$).

---

## 10. Logistic Regression Classification

Logistic Regression models the posterior probability of the positive class using the sigmoid function:

$$P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

The optimization objective minimizes the L2-regularized binary cross-entropy loss:

$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|_2^2 + C \sum_{i=1}^{N} \log\left(1 + e^{-y_i (\mathbf{w}^T \mathbf{x}_i + b)}\right)$$

- **Hyperparameters:** $C = 1.0$, `max_iter = 1000`, `solver = 'lbfgs'`, `random_state = 42`.
- **Interpretability:** Unlike deep neural networks, logistic regression coefficients $\mathbf{w}$ provide direct, linear explainability.

---

## 11. Training/Test Methodology

An 80/20 train/test split was maintained across all experiments:
- **Ott Hotel:** 1,280 train / 320 test.
- **Amazon Product:** 32,345 train / 8,087 test.
- **Yelp Restaurant:** 21,565 train / 5,390 test.

The held-out test partitions remained completely isolated and were evaluated only once following model serialization.

---

## 12. Evaluation Metrics

Model performance is evaluated across six standard statistical metrics:

1. **Accuracy:** Overall proportion of correct predictions:
   $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
2. **Balanced Accuracy:** Average recall obtained on each class, critical for imbalanced corpora:
   $$\text{Balanced Accuracy} = \frac{1}{2} \left(\frac{\text{TP}}{\text{TP} + \text{FN}} + \frac{\text{TN}}{\text{TN} + \text{FP}}\right)$$
3. **Precision:** Ratio of correctly predicted positive observations to total predicted positive observations:
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
4. **Recall (Sensitivity):** Ratio of correctly predicted positive observations to all observations in actual class:
   $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
5. **Macro F1-Score:** Unweighted harmonic mean of precision and recall:
   $$\text{Macro F1} = \frac{2 \times \text{Precision}_{\text{macro}} \times \text{Recall}_{\text{macro}}}{\text{Precision}_{\text{macro}} + \text{Recall}_{\text{macro}}}$$
6. **Weighted F1-Score:** Support-weighted harmonic mean.

---

## 13. Empirical Results

### 13.1 Hotel Model (Ott Benchmark, N=320)
- **Accuracy:** **89.69%** (287 / 320 correct)
- **Balanced Accuracy:** **89.69%**
- **Macro Precision:** 89.70% &bull; **Macro Recall:** 89.69% &bull; **Macro F1:** **0.8969**
- **Confusion Matrix:** $\text{TN}=145$, $\text{FP}=15$, $\text{FN}=18$, $\text{TP}=142$
- **Class Performance:**
  - Truthful: Precision 88.96%, Recall 90.62%, F1 0.8978
  - Deceptive: Precision 90.45%, Recall 88.75%, F1 0.8959

### 13.2 Product Model (Amazon 40K, N=8,087)
- **Accuracy:** **93.36%** (7,550 / 8,087 correct)
- **Balanced Accuracy:** **93.36%**
- **Macro Precision:** 93.41% &bull; **Macro Recall:** 93.36% &bull; **Macro F1:** **0.9336**
- **Confusion Matrix:** $\text{TN}=3,842$, $\text{FP}=202$, $\text{FN}=335$, $\text{TP}=3,708$
- **Class Performance:**
  - Human Original (`OR`): Precision 91.98%, Recall 95.00%, F1 0.9347
  - GPT-2 Generated (`CG`): Precision 94.83%, Recall 91.71%, F1 0.9325

### 13.3 Restaurant Model (YelpChi Benchmark, N=5,390)
- **Accuracy:** **77.79%** (4,193 / 5,390 correct)
- **Balanced Accuracy:** **53.88%**
- **Macro Precision:** 69.72% &bull; **Macro Recall:** 53.88% &bull; **Macro F1:** **0.5189**
- **Confusion Matrix:** $\text{TN}=4,074$, $\text{FP}=76$, $\text{FN}=1,121$, $\text{TP}=119$
- **Class Performance:**
  - Recommended (`N`): Precision 78.42%, Recall 98.17%, F1 0.8719
  - Filtered (`Y`): Precision 61.03%, Recall 9.60%, F1 0.1659

---

## 14. Comparative Analysis & Domain Incompatibility

A core contribution of this project is demonstrating why multi-domain review models cannot be combined into a monolithic binary classifier:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       THREE DISTINCT LEARNING CONSTRUCTS                    │
├────────────────────────┬──────────────────────────┬─────────────────────────┤
│    Hotel Reviews       │      Product Reviews     │    Restaurant Reviews   │
├────────────────────────┼──────────────────────────┼─────────────────────────┤
│ Psycholinguistic       │ Machine-Generated        │ Algorithmic Platform    │
│ Deception Markers      │ Stylistic Tokens         │ Recommendation Filter   │
│                        │                          │                         │
│ Truthful vs Deceptive  │ Human (OR) vs GPT-2 (CG) │ Recommended vs Filtered │
└────────────────────────┴──────────────────────────┴─────────────────────────┘
```

1. **Ott Hotel Model** uses Truthful and Deceptive prediction labels to evaluate crowdsourced deceptive reviews versus truthful TripAdvisor reviews (Ott et al., 2011, 2013).
2. **Amazon Product Model** is implemented solely as TF-IDF Vectorizer + Logistic Regression, distinguishing human-written original reviews (`OR`) from fine-tuned GPT-2-generated synthetic text (`CG`). It does not calculate transformer perplexity, does not use a transformer at inference, and does not independently run a repetitive n-gram detection algorithm.
3. **Yelp Restaurant Model** evaluates platform recommendation filter decisions: Recommended (`N`) vs Filtered / Not Recommended (`Y`). Neither label confirms genuine or fake reviews.

Merging these datasets would cause catastrophic construct conflation. Each domain must be deployed as a dedicated task expert.

---

## 15. System Architecture

The ReviewTrust software architecture is decoupled into three primary layers:

1. **Presentation Layer (Streamlit Frontend):** Interactive web dashboard supporting multi-page navigation, domain switching, single-review inputs, file uploads, and analytics displays.
2. **Inference & Explainability Engine (`model_utils.py`):** Loads serialized pipelines (`.joblib`), executes vectorization and logistic inference, computes probabilistic confidence, and derives token-level attribution scores.
3. **Data Persistence Layer (`db_utils.py`):** Interfaces with a local SQLite database (`reviews.db`) using parameterized SQL queries.

---

## 16. Application Modules

- **Module A — Home Page:** Comprehensive academic introduction, student identification, problem statement, objectives, and technology stack.
- **Module B — Domain Selection:** Dropdown selector enabling switching between Hotel, Product, and Restaurant task experts, showing exact citations and label meanings.
- **Module C — Single Review Analysis:** Text area with 1-click sample review buttons, prediction badges, probability distributions, confidence meters, and interactive text highlighting.
- **Module D — Bulk CSV Analysis:** Multi-column CSV file uploader with dynamic column detection, batch scoring, summary statistics, and downloadable CSV exports.
- **Module E — Review Analytics Dashboard:** Real-time KPI summary cards, domain breakdown bar charts, and category distribution histograms.
- **Module F — SQL Database & History:** Searchable, filterable audit log displaying previous analysis runs with CSV export and clearing capabilities.

---

## 17. Database Design

The relational database (`reviews.db`) contains the `review_predictions` table:

```sql
CREATE TABLE IF NOT EXISTS review_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_text TEXT NOT NULL,
    domain TEXT DEFAULT 'Hotel Reviews',
    model_name TEXT DEFAULT 'Ott Hotel Model',
    predicted_label TEXT NOT NULL,
    raw_prediction TEXT DEFAULT '',
    confidence REAL NOT NULL,
    prob_baseline REAL DEFAULT 0.0,
    prob_target REAL DEFAULT 0.0,
    p_deceptive REAL DEFAULT 0.0,
    p_truthful REAL DEFAULT 0.0,
    reliability_score REAL DEFAULT 0.0,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

All database queries utilize parameterized placeholders (`?`) to prevent SQL injection vulnerabilities.

---

## 18. Testing & Validation

The complete application was verified using an automated 25-point integration test suite (`scratch/test_app_integration.py`):

| Test ID | Test Name | Execution Status |
| :---: | :--- | :---: |
| 1 | Application Syntax Compilation | **PASS** |
| 2 | Home Page Metadata & Team Info | **PASS** |
| 3 | Domain Selection Configuration | **PASS** |
| 4 | Hotel Model Serialization & Load | **PASS** |
| 5 | Product Model Serialization & Load | **PASS** |
| 6 | Restaurant Model Serialization & Load | **PASS** |
| 7 | Single Hotel Review Inference & XAI | **PASS** |
| 8 | Single Product Review Inference & XAI | **PASS** |
| 9 | Single Restaurant Review Inference & XAI | **PASS** |
| 10 | Probability Normalization (Sum = 1.0) | **PASS** |
| 11 | Bulk CSV Upload & Parsing | **PASS** |
| 12 | Text-Column Dynamic Selection | **PASS** |
| 13 | Bulk Prediction Execution | **PASS** |
| 14 | CSV Export Serialization | **PASS** |
| 15 | Dashboard Metrics Calculation | **PASS** |
| 16 | Chart Rendering Pipeline | **PASS** |
| 17 | Parameterized SQLite Insertion | **PASS** |
| 18 | History Log Retrieval | **PASS** |
| 19 | Domain Filtering in SQLite | **PASS** |
| 20 | Empty / Whitespace Input Handling | **PASS** |
| 21 | Missing Column Error Handling | **PASS** |
| 22 | Invalid Domain Error Handling | **PASS** |
| 23 | Model Isolation & Weight Independence | **PASS** |
| 24 | Zero Retraining on App Startup | **PASS** |
| 25 | Preservation of Baseline Codebase | **PASS** |

**Final Verification Score:** **25 / 25 Tests Passed (100.0%)**.

---

## 19. Limitations

1. **Domain Lexical Boundaries:** Hotel models predict Truthful vs Deceptive reviews within the 20 Chicago luxury hotels benchmark (Ott et al., 2011, 2013) and cannot reliably evaluate electronics, as hotel vocabulary (*concierge, bellman, suite*) does not overlap with electronics vocabulary (*battery, firmware, charging port*).
2. **Generative Horizon Specificity:** The AI product detector was trained on fine-tuned GPT-2 text (Salminen et al., 2022). The model is implemented solely as a TF-IDF Vectorizer + Logistic Regression pipeline distinguishing human-written original reviews (`OR`) from fine-tuned GPT-2-generated synthetic text (`CG`). The implementation does not calculate transformer perplexity, does not use a transformer at inference, and does not independently run a repetitive n-gram detection algorithm. Furthermore, `OR` does not prove honesty and `CG` does not prove fraud. Performance will differ on modern frontier LLMs.
3. **Platform Filtering Scope & Natural Class Imbalance:** In the Yelp restaurant domain, labels represent platform recommendation decisions: Recommended (`N`) vs Filtered / Not Recommended (`Y`). The dataset exhibits natural class imbalance (76.98% Recommended vs 23.02% Filtered). A text-only model achieves 77.79% overall accuracy and 98.17% recall on Recommended reviews, but catches only 9.60% of Filtered reviews, yielding a Balanced Accuracy of 53.88% and Macro F1 of 51.89%. Crucially, neither label confirms genuine or fake reviews.
4. **Not Definitive Legal Proof:** Predictions indicate statistical model confidence, not transactional proof of fraud.

---

## 20. Future Scope

1. **Modern LLM Adaptation:** Incorporating few-shot prompting and fine-tuning with modern LLM benchmarks (Llama-3, Mistral, GPT-4).
2. **Review Metadata Integration:** Exploring additional contextual features alongside review text.
3. **Transformer Embeddings:** Benchmarking contextual transformer representations (RoBERTa, DeBERTa) against the linear TF-IDF baseline.
4. **Active Learning:** Enabling human-in-the-loop validation of borderline reviews.

---

## 21. Conclusion

This B.Tech minor project successfully establishes an end-to-end, multi-domain Fake Review Detection and Review Reliability Analysis System. By respecting the fundamental divergence between psychological deception (Hotels), machine-generated text (Products), and platform filtering (Restaurants), the system avoids construct conflation and provides scientifically honest predictions. 

With 100% automated test suite coverage, transparent mathematical explainability, local database auditing, and a responsive Streamlit web application, ReviewTrust represents a complete, academically sound software engineering project suitable for college examination and live demonstration.

---

## Academic References

1. Ott, M., Choi, Y., Cardie, C., & Hancock, J. T. (2011). Finding deceptive opinion spam by any stretch of the imagination. *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL)*, 309–319.
2. Ott, M., Cardie, C., & Hancock, J. T. (2013). Negative deceptive opinion spam. *Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)*, 497–501.
3. Salminen, J., Kandpal, C., Kamel, A. M., Jung, S. G., & Jansen, B. J. (2022). Creating and detecting fake reviews on e-commerce platforms. *ACM Transactions on the Web*, 16(4), 1–32.
4. Rayana, S., & Akoglu, L. (2015). Collective opinion spam detection: Bridging review networks and metadata. *Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 985–994.
5. Jindal, N., & Liu, B. (2008). Opinion spam and analysis. *Proceedings of the 2008 International Conference on Web Search and Data Mining (WSDM)*, 219–230.
