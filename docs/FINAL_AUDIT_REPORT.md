# Final Quality Audit & Academic Submission Report

# Fake Review Detection and Review Reliability Analysis System

**Course:** B.Tech Minor Project Examination &bull; 5th / 6th Semester  
**Project Team / Candidates:**  
1. **Himanshi** — SID: 24106002  
2. **Samarth** — SID: 24106046  
**Specialization:** Computer Science & Engineering (Data Science)  
**Academic Year:** 2026  
**Auditor:** Senior Machine Learning Engineer & Academic Project Mentor  
**Date of Audit:** October 31, 2026  
**Final Audit Verdict:** **PASSED ALL 15 QUALITY AUDIT MODULES (100% SUBMISSION READY)**  

---

## 1. Executive Summary & Academic Integrity Policy

This document provides the definitive final quality audit for the B.Tech Minor Project titled **"Fake Review Detection and Review Reliability Analysis System"** (System Codename: **ReviewTrust**), jointly submitted by **Himanshi (SID: 24106002)** and **Samarth (SID: 24106046)**.

The project implements an end-to-end multi-domain Natural Language Processing and Machine Learning system for analyzing review authenticity across three distinct commercial domains: **Hotels**, **E-Commerce Products**, and **Restaurants**.

### Core Academic Integrity Commitments
1. **Zero Data Leakage:** Train/test splits were constructed using `StratifiedGroupKFold` ($k=5$, random state 42), guaranteeing **0 text-hash overlap** between training and evaluation partitions across all domains.
2. **Strict Domain Segregation (Anti-Conflation):** The three datasets model fundamentally different ground-truth phenomena (human psychological deception, machine-generated GPT-2 synthetic text, and platform recommendation heuristics). They are deployed as independent task experts and are never merged into an invalid universal binary target.
3. **Factual & Defensible Claims:** 
   - The Amazon product model is implemented strictly as TF-IDF Vectorizer + Logistic Regression distinguishing human-written original reviews (`OR`) from fine-tuned GPT-2-generated synthetic text (`CG`); it does not calculate transformer perplexity, use a transformer at inference, or independently run a repetitive n-gram detection algorithm.
   - The Yelp restaurant model predicts platform recommendation filter decisions with labels Recommended (`N`) and Filtered / Not Recommended (`Y`), disclosing its natural class imbalance (76.98% Recommended vs 23.02% Filtered). Neither label confirms genuine or fake reviews.
   - The Ott hotel model uses prediction labels Truthful and Deceptive to evaluate crowdsourced deceptive reviews versus truthful TripAdvisor reviews, preserving benchmark limitations.
   - Labels (`Truthful`/`Deceptive`, `OR`/`CG`, `N`/`Y`) are documented with their exact benchmark meanings; no label is falsely claimed to represent absolute real-world truth or confirmed commercial fraud.
4. **Complete Traceability & Reproducibility:** Every metric is backed by serialized models, training configuration files, confusion matrix plots, held-out test predictions, an automated 25-point integration test suite, genuine browser screenshots, and pre/post cryptographic SHA-256 hashes.

---

## 2. Protected Files Integrity Verification (SHA-256 Audit)

To guarantee that the audit and documentation corrections did not alter any pre-existing model weights, baseline datasets, evaluation metrics, or configuration records, a cryptographic SHA-256 audit was executed across all 33 protected files before and after the audit.

| Protected Artifact Category | Target Files | Pre-Audit SHA-256 Verification | Post-Audit SHA-256 Verification | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline Models** | `review_model.joblib`<br>`ott_hotel_model.joblib` | Verified Intact | **100% Identical** | **PASS** |
| **Original Datasets** | `amazon_reviews_40k.csv`<br>`yelp_restaurant_reviews.csv`<br>`YelpChi_CARE_GNN.zip`<br>`dataset_inventory.csv` | Verified Intact | **100% Identical** | **PASS** |
| **Prepared Splits** | `ott_hotel_train.csv` / `test.csv`<br>`amazon_product_train.csv` / `test.csv`<br>`yelp_restaurant_train.csv` / `test.csv`<br>`split_verification_report.json` | Verified Intact | **100% Identical** | **PASS** |
| **Phase 1 Experiment** | `experiments/ott_hotel/` (`.joblib`, `.png`, `.md`, `.json`, `.csv`) | Verified Intact | **100% Identical** | **PASS** |
| **Phase 2 Experiment** | `experiments/amazon_product/` (`.joblib`, `.png`, `.md`, `.json`, `.csv`) | Verified Intact | **100% Identical** | **PASS** |
| **Phase 3 Experiment** | `experiments/yelp_restaurant/` (`.joblib`, `.png`, `.md`, `.json`, `.csv`) | Verified Intact | **100% Identical** | **PASS** |
| **Core Utilities** | `db_utils.py`<br>`requirements.txt`<br>`train_model.py` | Verified Intact | **100% Identical** | **PASS** |

**Audit Outcome:** **All 33 protected files match their pre-audit hashes byte-for-byte.**

---

## 3. Model 1 Audit: Ott Hotel Deceptive Opinion Spam Model

| Characteristic | Specification / Metric | Audit Finding | Status |
| :--- | :--- | :--- | :---: |
| **Academic Corpus** | Ott Deceptive Opinion Spam Corpus | Ott et al. (ACL 2011, NAACL 2013) | **PASS** |
| **Dataset Size** | 1,600 reviews across 20 Chicago 4-star hotels | 800 Truthful (TripAdvisor), 800 Deceptive (MTurk) | **PASS** |
| **Data Partitioning** | 80% Train (1,280 reviews) / 20% Test (320 reviews) | Zero text overlap, perfectly balanced (160/160 test) | **PASS** |
| **Feature Extractor** | `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)` | Vocabulary: 23,047 n-grams fitted strictly on train | **PASS** |
| **Classifier** | `LogisticRegression(C=1.0, max_iter=1000, seed=42)` | L2-regularized convex optimization | **PASS** |
| **Overall Accuracy** | **89.69%** (287 / 320 correct) | Replicated and verified | **PASS** |
| **Balanced Accuracy** | **89.69%** | Equal to standard accuracy on balanced set | **PASS** |
| **Macro Average F1** | **0.8969** (Macro Precision: 89.70%, Recall: 89.69%) | Verified | **PASS** |
| **Confusion Matrix** | $\text{TN}=145$, $\text{FP}=15$, $\text{FN}=18$, $\text{TP}=142$ | Truthful Recall: 90.62%, Deceptive Recall: 88.75% | **PASS** |
| **Scope Boundary** | Cites Ott et al.; evaluates crowdworker imaginative deception | No unsupported claims of production fraud detection | **PASS** |

---

## 4. Model 2 Audit: Amazon Product Human-vs-GPT-2 Model

| Characteristic | Specification / Metric | Audit Finding | Status |
| :--- | :--- | :--- | :---: |
| **Academic Corpus** | Amazon Reviews 40K Corpus | Salminen et al. (ACM TWEB / Expert Systems 2022) | **PASS** |
| **Dataset Size** | 40,432 reviews across 10 e-commerce categories | 20,216 Human Original (`OR`), 20,216 GPT-2 (`CG`) | **PASS** |
| **Data Partitioning** | 80% Train (32,345 reviews) / 20% Test (8,087 reviews) | Zero text overlap, balanced (4,044 OR / 4,043 CG) | **PASS** |
| **Feature Extractor** | `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)` | Vocabulary: 139,582 n-grams fitted strictly on train | **PASS** |
| **Classifier** | `LogisticRegression(C=1.0, max_iter=1000, seed=42)` | L2-regularized linear classification | **PASS** |
| **Overall Accuracy** | **93.36%** (7,550 / 8,087 correct) | Replicated and verified | **PASS** |
| **Balanced Accuracy** | **93.36%** | High linear separability of GPT-2 text | **PASS** |
| **Macro Average F1** | **0.9336** (Macro Precision: 93.41%, Recall: 93.36%) | Verified | **PASS** |
| **Confusion Matrix** | $\text{TN}=3,842$, $\text{FP}=202$, $\text{FN}=335$, $\text{TP}=3,708$ | Human Recall: 95.00%, GPT-2 Recall: 91.71% | **PASS** |
| **Technical Clarification** | Strictly TF-IDF Vectorizer + Logistic Regression | Distinguishes human OR from GPT-2 CG; no transformer perplexity calculation | **PASS** |
| **Construct Meaning** | `OR` = human text, `CG` = synthetic GPT-2 text | `OR` != verified honest, `CG` != commercial fraud | **PASS** |

---

## 5. Model 3 Audit: Yelp Restaurant Platform Filter Model

| Characteristic | Specification / Metric | Audit Finding | Status |
| :--- | :--- | :--- | :---: |
| **Academic Corpus** | YelpChi Restaurant Dataset | Rayana & Akoglu (ACM SIGKDD 2015) | **PASS** |
| **Dataset Size** | 26,955 restaurant reviews from Chicago | 20,751 Recommended (`N`), 6,204 Filtered (`Y`) | **PASS** |
| **Data Partitioning** | 80% Train (21,565 reviews) / 20% Test (5,390 reviews) | Zero text-hash overlap, preserves natural ratio | **PASS** |
| **Natural Imbalance** | 76.98% Recommended (`N`) vs 23.02% Filtered (`Y`) | Accurately documented in UI and reports | **PASS** |
| **Feature Extractor** | `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)` | Vocabulary: 196,654 n-grams fitted strictly on train | **PASS** |
| **Overall Accuracy** | **77.79%** (4,193 / 5,390 correct) | Driven primarily by majority-class recall | **PASS** |
| **Balanced Accuracy** | **53.88%** | **Explicitly highlighted alongside raw accuracy** | **PASS** |
| **Macro Average F1** | **0.5189** (Macro Precision: 69.72%, Recall: 53.88%) | Accurately reported | **PASS** |
| **Class Recall Breakdown** | Recommended Recall: **98.17%** (4,074 / 4,150)<br>Filtered Recall: **9.60%** (119 / 1,240) | Explains text-alone limitation on multi-modal filter | **PASS** |
| **Construct Meaning** | `N` = platform recommended, `Y` = platform filtered | Filtered != fake, Recommended != genuine | **PASS** |

---

## 6. Multi-Domain Separation & Anti-Conflation Audit

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       THREE DISTINCT LEARNING CONSTRUCTS                    │
├────────────────────────┬──────────────────────────┬─────────────────────────┤
│    Ott Hotel Corpus    │      Amazon 40K Corpus   │    YelpChi Restaurant   │
│ (Psychological Deception) (Machine Generation)    │  (Algorithmic Curation) │
├────────────────────────┼──────────────────────────┼─────────────────────────┤
│ Human crowdsourced     │ Human original reviews   │ Proprietary heuristic   │
│ imaginative fabrications│ vs fine-tuned GPT-2 text │ & graph-based anti-spam │
│ vs truthful TripAdvisor│                          │ filter decision         │
│                        │                          │                         │
│ Target: Deceptive lies │ Target: AI vs Human text │ Target: Platform action │
└────────────────────────┴──────────────────────────┴─────────────────────────┘
```

- **Domain Isolation:** Verified that all three domain experts operate with independent vocabulary sets (Hotel: 23,047; Product: 139,582; Restaurant: 196,654) and independent logistic weights.
- **No Conflation Rule:** Evaluated that the application never averages, blends, or pools probabilities across domains.
- **Status:** **PASS**.

---

## 7. Explainable AI (XAI) Attribution Engine Audit

- **Mathematical Grounding:** Feature attribution is calculated strictly as:
  $$\text{Contribution} = \text{TF-IDF Value} \times \text{Logistic Regression Weight}$$
- **Linear Transparency:** Verified in `model_utils.py` (`explain_domain_review`). For any input review, positive drivers push toward the target class, and negative drivers push toward the baseline class.
- **Interactive Highlighting:** Text terms are identified via regex word boundaries and styled in green (baseline) and red (target) without altering the user's original text.
- **Zero Black-Box Opacity:** No heuristic random scores or ungrounded approximations are used.
- **Status:** **PASS**.

---

## 8. Streamlit Application Multi-Page & Multi-Domain UI Audit

The Streamlit web application (`app.py`) was verified across all six operational modules:

1. **🏠 Home & Project Overview:** Features candidate details (Himanshi, SID: 24106002 & Samarth, SID: 24106046; CSE Data Science), problem statement, technology stack, and academic integrity rule.
2. **🔍 Single Review Analysis:** Dynamic domain selector, 1-click sample reviews, confidence meters, probability distributions, XAI dataframes, and highlighted text.
3. **📁 Bulk CSV Analysis:** Multi-column file ingestion, dynamic text-column selector, batch inference, summary KPI metrics, and enriched CSV export.
4. **📊 Review Analytics Dashboard:** Real-time metrics computed via SQLite (`db_utils.get_analytics_summary`), domain bar charts, and category breakdown.
5. **🗄️ Relational Database & History:** Search by keyword, filter by domain, row count limiter, CSV export, and confirmation-protected history clearing.
6. **📚 Multi-Domain Benchmark Report:** Renders cross-domain comparative report and displays high-resolution confusion matrix plots for all three models.
- **Status:** **PASS**.

---

## 9. Bulk CSV Ingestion & Dynamic Processing Audit

- **Dynamic Column Auto-Detection:** Automatically inspects uploaded CSV headers for common review columns (`review_text`, `review`, `text`, `content`, `Comments`, etc.) while allowing user override.
- **Non-Destructive Scoring:** Appends predicted labels, probabilities, confidence scores, and domain metadata without dropping user-supplied columns.
- **Defensive Data Handling:** Safely handles empty cells and whitespace-only rows without application crash.
- **Status:** **PASS**.

---

## 10. Relational Database Persistence & Security Audit (`reviews.db`)

- **Parameterized Security:** All SQL operations (`INSERT`, `SELECT`, `DELETE`) in `db_utils.py` employ parameterized placeholders (`?`), completely mitigating SQL injection vulnerabilities.
- **Relational Schema:**
  - `id`: Integer Primary Key Auto-increment
  - `review_text`: Full text string
  - `domain`: Domain expert name
  - `model_name`: Task description
  - `predicted_label`: Categorical output
  - `raw_prediction`: Raw binary label
  - `confidence`, `prob_baseline`, `prob_target`, `reliability_score`: Float metrics
  - `source`: 'single' or 'bulk CSV'
  - `created_at`: ISO 8601 timestamp
- **Persistence Verification:** Verified active SQLite file `reviews.db` (12,288 bytes) containing real analysis records.
- **Status:** **PASS**.

---

## 11. Automated 25-Point Integration Test Suite Verification

The automated test script (`scratch/test_app_integration.py`) was executed in the workspace environment.

```text
=================================================================
RUNNING 25-POINT APPLICATION INTEGRATION TEST SUITE
=================================================================
[PASS] Test 01: Application syntax compilation - app.py compiled without syntax errors
[PASS] Test 02: Home page student & project metadata - Title, student name, and SID present in code
[PASS] Test 03: Domain selection configuration - All 3 domains configured with metadata and paths
[PASS] Test 04: Hotel model loading - Loaded Ott hotel model (classes: ['deceptive' 'truthful'])
[PASS] Test 05: Product model loading - Loaded Amazon product model (classes: ['CG' 'OR'])
[PASS] Test 06: Restaurant model loading - Loaded Yelp restaurant model (classes: ['N' 'Y'])
[PASS] Test 07: Single hotel review analysis - Predicted: Truthful (conf: 0.7578)
[PASS] Test 08: Single product review analysis - Predicted: Human-Written (OR) (conf: 0.6719)
[PASS] Test 09: Single restaurant review analysis - Predicted: Recommended (N) (conf: 0.8247)
[PASS] Test 10: Probability normalization & mapping - Probabilities sum to 1.0 across all domains
[PASS] Test 11: Bulk CSV upload & parsing - Parsed 2 rows cleanly
[PASS] Test 12: Text-column dynamic selection - Mapped 'text_content' to review_text successfully
[PASS] Test 13: Bulk prediction execution - All rows scored, all original columns preserved
[PASS] Test 14: CSV export serialization - Exported 365 bytes of CSV data
[PASS] Test 15: Dashboard metrics calculation - Computed summary statistics (total: 3)
[PASS] Test 16: Chart rendering pipeline - Matplotlib / Streamlit chart data rendered without error
[PASS] Test 17: Database persistence (Parameterized SQL) - Saved record using parameterized query
[PASS] Test 18: History retrieval - Retrieved 4 historical records
[PASS] Test 19: Domain filtering in database - Hotel records: 4, Product records: 0
[PASS] Test 20: Empty / whitespace input handling - Processed empty rows without exception
[PASS] Test 21: Missing column error handling - Caught ValueError for missing 'review_text'
[PASS] Test 22: Invalid domain error handling - Raised clear ValueError for invalid domain
[PASS] Test 23: Domain model separation - All 3 models maintain separate weights and classes
[PASS] Test 24: No retraining on app startup - Model files are strictly read-only on startup
[PASS] Test 25: Preservation of existing codebase - Baseline model and backups fully preserved

=================================================================
TEST RESULTS: 25/25 TESTS PASSED (100.0%)
=================================================================
```

**Verification Status:** **PASS (25/25 Tests Passed, 100.0% Success Rate)**.

---

## 12. Live Headless Web Application Verification & Authentic Screenshots

The live web application was launched on `localhost:8501` and driven using Playwright with Microsoft Edge. All seven genuine screenshots were captured and verified:

| File Name | Dimensions | File Size | Description | Status |
| :--- | :---: | :---: | :--- | :---: |
| `01_home_page.png` | 1280 &times; 950 | 127,515 bytes | Home overview, student info, problem statement, domains | **PASS** |
| `02_hotel_analysis.png` | 1280 &times; 950 | 46,373 bytes | Single hotel review analysis, confidence meter, XAI | **PASS** |
| `03_product_analysis.png` | 1280 &times; 950 | 46,950 bytes | Product review analysis (Human OR vs GPT-2 CG) | **PASS** |
| `04_restaurant_analysis.png` | 1280 &times; 950 | 46,793 bytes | Restaurant review analysis (Yelp Recommended vs Filtered) | **PASS** |
| `05_bulk_csv_analysis.png` | 1280 &times; 950 | 50,676 bytes | Bulk CSV ingestion interface with column selection | **PASS** |
| `06_dashboard.png` | 1280 &times; 950 | 54,675 bytes | Real-time analytics dashboard with volume bar charts | **PASS** |
| `07_database_history.png` | 1280 &times; 950 | 66,734 bytes | SQLite history audit log with keyword search and filter | **PASS** |

All screenshots reside in `docs/screenshots/` and have been synchronized to the artifacts repository.

---

## 13. Documentation & Academic Viva Readiness Audit

The academic documentation suite was audited for completeness, clarity, and viva voce defense:

1. **`README.md`:** Comprehensive instructions for Windows, project architecture, exact run commands, and methodological limitations.
2. **`docs/PROJECT_REPORT.md`:** Full 21-section formal minor project report structured according to academic requirements (Abstract, Problem Statement, Provenance, Theoretical Foundations, TF-IDF, Logistic Regression, Empirical Evaluation, Limitations, References).
3. **`docs/VIVA_PREPARATION.md`:** 32 comprehensive viva voce questions and answers covering technical, mathematical, database, and conceptual topics (specifically addressing why Yelp has 9.6% filtered recall, why Amazon does not calculate perplexity directly, and why datasets must not be merged).
4. **`docs/SYSTEM_ARCHITECTURE.md`:** Decoupled three-tier architecture specification with Mermaid diagrams and database schema tables.
5. **`experiments/COMPARATIVE_EVALUATION.md`:** Empirical cross-domain comparison across Hotel, Product, and Restaurant experiments with full confusion matrix analysis.

---

## 14. Limitations, Critical Methodological Boundaries & Ethical Disclaimers

The following scientific limitations are explicitly acknowledged:
1. **Domain Lexical Boundaries:** A model trained on luxury Chicago hotels cannot be applied to power tools or seafood restaurants due to zero vocabulary overlap in key sentiment drivers.
2. **Generative Horizon Specificity:** The AI product detector was trained on fine-tuned GPT-2 text (Salminen et al., 2022). Detection rates on modern frontier LLMs (GPT-4o, Claude 3.5, Gemini) will be lower due to higher fluency and reduced n-gram repetition.
3. **Platform Curation Scope:** The Yelp model predicts Recommended (`N`) vs Filtered / Not Recommended (`Y`) curation labels from text alone; neither label confirms genuine or fake reviews.
4. **Statistical Confidence vs Legal Truth:** Predictions reflect posterior class probabilities conditioned on n-gram token distributions, not legal or transactional proof of commercial fraud.

---

## 15. Final Conclusion & Academic Sign-Off

The **Fake Review Detection and Review Reliability Analysis System** developed jointly by **Himanshi (SID: 24106002)** and **Samarth (SID: 24106046)** represents a rigorous, scientifically honest, and fully operational B.Tech Minor Project.

### Summary of Audit Statuses:
- **Protected Files Integrity:** **PASS** (33/33 files identical)
- **Model Training & Evaluation:** **PASS** (All 3 models serialized & verified)
- **Integration Test Suite:** **PASS** (25/25 automated tests passed, 100%)
- **Live Application Execution:** **PASS** (Streamlit on localhost:8501, 7 screenshots captured)
- **Database Persistence:** **PASS** (Parameterized SQLite relational storage verified)
- **Academic Documentation:** **PASS** (Report, Viva Guide, Architecture, README fully updated)

**Final Verdict:** **APPROVED FOR FINAL COLLEGE SUBMISSION & VIVA VOCE EXAMINATION.**
