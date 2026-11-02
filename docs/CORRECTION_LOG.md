# Academic Consistency Correction Log

**Project Title:** Fake Review Detection and Review Reliability Analysis System (ReviewTrust)  
**Project Team:** Himanshi (SID: 24106002), Samarth (SID: 24106046) &nbsp;|&nbsp; **Institution:** Punjab Engineering College &nbsp;|&nbsp; **Branch:** B.Tech CSE (Data Science)  
**Date of Corrections:** November 2, 2026  
**Status:** Completed, Verified, and Tested  

---

## 1. Summary of Changes

A limited academic consistency correction was applied across project documentation and application UI definitions to ensure rigorous alignment with original benchmark dataset literature and implementation reality.

### A. Amazon Product Domain
- **Action:** Described the model solely as **TF-IDF Vectorizer + Logistic Regression**, distinguishing human-written original reviews (`OR`) from fine-tuned GPT-2-generated synthetic text (`CG`).
- **Eliminated:** Removed every claim that the implementation calculates transformer perplexity, uses a transformer model at inference, or independently runs a repetitive n-gram detection algorithm.
- **Interpretation:** Clarified that `OR` indicates human authorship (not verified honesty) and `CG` indicates synthetic text (not confirmed commercial fraud).

### B. Yelp Restaurant Domain
- **Action:** Described labels strictly as **Recommended (`N`)** and **Filtered / Not Recommended (`Y`)**.
- **Eliminated:** Removed unsupported claims that the dataset's filtering labels rely on account age, IP clustering, or social graphs.
- **Interpretation:** Clearly stated that neither label confirms genuine or fake reviews; platform filtering represents an automated curation action rather than an adjudication of authenticity.

### C. Ott Hotel Domain
- **Action:** Standardized prediction labels to **Truthful** and **Deceptive**.
- **Eliminated:** Removed "Likely Genuine" as a substitute for Truthful.
- **Interpretation:** Preserved the benchmark limitations (1,600 reviews across 20 luxury Chicago hotels; crowdworker imaginative deception versus truthful TripAdvisor reviews; Ott et al., 2011, 2013).

---

## 2. Inventory of Files Changed

| File Path | Nature of Modifications |
| :--- | :--- |
| `model_utils.py` | Updated hotel labels to `["Truthful", "Uncertain (Borderline)", "Deceptive"]`; aligned Amazon and Yelp limitation strings. |
| `app.py` | Updated Academic Integrity banner and probability caption to reference Truthful/Deceptive and Recommended/Filtered labels. |
| `README.md` | Standardized key features table, uncertainty categorization examples, and Section 7 limitation boundaries. |
| `docs/PROJECT_REPORT.md` | Aligned Objective 3, Section 6 dataset provenance, Section 14 comparative analysis, Section 19 limitations, and Section 20 future scope. |
| `docs/VIVA_PREPARATION.md` | Updated Q10 (Confusion Matrix), Q18 (Ott benchmark), Q19 (Amazon OR/CG), Q20 (Authorship vs Honesty), Q21-Q23 (Yelp platform filter), Q30 (Limitations), and Q31 (Future Scope). |
| `docs/SYSTEM_ARCHITECTURE.md` | Aligned database schema table categorical output examples. |
| `experiments/COMPARATIVE_EVALUATION.md` | Aligned cross-domain comparison table and Section 3 learning constructs. |
| `docs/FINAL_AUDIT_REPORT.md` | Aligned Executive Summary, Section 4 (Amazon), Section 5 (Yelp), Section 6 (Anti-conflation), Section 11 (Test 07), and Section 14 (Limitations). |
| `test_system.py` | Updated test expectations to assert Truthful and Deceptive labels (7/7 tests passed). |
| `scratch/test_app_integration.py` | Updated test expectations for hotel prediction to assert Truthful (25/25 tests passed). |

---

## 3. Preservation of Protected Artifacts

All model files, original candidate datasets, and prepared train/test splits were cryptographically verified using SHA-256 hashes before and after corrections:
- `review_model.joblib`: **Unchanged** (SHA-256: `1075ad49c159...`)
- `ott_hotel_model.joblib`: **Unchanged** (SHA-256: `134654835444...`)
- `amazon_product_model.joblib`: **Unchanged** (SHA-256: `ac08e390760d...`)
- `yelp_restaurant_model.joblib`: **Unchanged** (SHA-256: `1febd26240f7...`)
- All 14 candidate and prepared dataset CSV/zip files: **Unchanged**
- SQLite database (`reviews.db`): **Preserved**

---

## 4. Test Suite Execution Results

1. **`scratch/test_app_integration.py`:**
   - **25 / 25 Tests Passed (100.0%)**
   - Verified syntax compilation, model loading, single review analysis, bulk CSV processing, database persistence, and domain model separation.
2. **`test_system.py`:**
   - **7 / 7 Tests Passed (100.0%)**
   - Verified single review predictions, feature explainability, edge case handling, and model reloading.

---

## 5. Final SRS Compliance Correction (Suspicion Score & Review Reliability Score)

**Date:** November 2, 2026 (Pre-Submission Audit)  
**Objective:** Fully align the Streamlit application and database analytics with the project Software Requirements Specification (SRS).

### A. Single Review Analysis (Hotel Domain):
- Added clearly labelled **Suspicion Score as a percentage**:
  $$\text{Suspicion Score} = P(\text{Deceptive}) \times 100\%$$
- Added **Individual Review Reliability Score as a percentage**:
  $$\text{Individual Review Reliability Score} = 100 - \text{Suspicion Score}$$
- Added **Review Classification: GENUINE / FAKE** (Truthful $\rightarrow$ **GENUINE**, Deceptive $\rightarrow$ **FAKE** at standard $0.50$ decision boundary).
- Displayed both scores prominently with dedicated score cards, progress bars, and class probabilities.
- Preserved the primary binary model prediction (**Truthful** vs **Deceptive**) separately at the standard $0.50$ decision boundary.
- Implemented the three-level operational risk interpretation:
  - **Below 35%:** Likely Genuine (`st.success`)
  - **35% to 65%:** Suspicious (`st.warning`)
  - **Above 65%:** Potentially Fake (`st.error`)
- Added explicit clarification that the scores are model-based estimates reflecting lexical patterns, not legal proof of fraud.
- Kept Amazon Product (`OR`/`CG`) and Yelp Restaurant (`N`/`Y`) strictly separate without applying genuine/fake scoring.

### B. Review Analytics Dashboard:
- Added **Overall Review Reliability Score** and **Suspicious Reviews Percentage** for Hotel review analyses:
  $$\text{Suspicious Reviews Percentage} = \frac{N_{\text{suspicious}}}{N_{\text{total hotel}}} \times 100\% = \frac{\text{Count of hotel reviews with Suspicion Score} \ge 35\%}{\text{Total hotel reviews analyzed}} \times 100\%$$
  $$\text{Overall Review Reliability Score} = 100\% - \text{Suspicious Reviews Percentage}$$
- Displayed the exact number of hotel reviews analysed ($N$).
- Separated visualizations into:
  1. **Hotel Binary Model Predictions** (`Truthful` vs `Deceptive`) at $0.50$ threshold, with clean mapping to prevent legacy label contamination.
  2. **Hotel Derived Risk Bands** (`Likely Genuine`, `Suspicious`, `Potentially Fake`).
- Maintained strict separation for Amazon Product (Human-written OR vs Synthetic CG) and Yelp Restaurant (Recommended N vs Filtered Y) distributions with domain-specific non-conflation caveats.

### C. Validation & Verification:
- **`scratch/test_srs_metrics.py`:** 100% PASS (Verifies mathematical formulas, risk thresholds, and clean distributions).
- **`scratch/test_app_integration.py`:** 25/25 PASS (100%).
- **`test_system.py`:** 7/7 PASS (100%).
- **Streamlit App (`streamlit.testing.v1.AppTest`):** 0 exceptions across all navigated pages. Streamlit live server operational on port 8501.

---

## 6. Multi-Domain Unified Classification & Reliability Scoring Implementation

**Date:** November 2, 2026 (Final Submission Release)  
**Authors:** Himanshi (SID: 24106002), Samarth (SID: 24106046)  
**Scope:** Complete cross-domain standardization across Hotel, Restaurant, and Product review domains.

### A. Common 3-Tier Classification & Scoring Across All Domains
Every individual review across Hotel, Restaurant, and Product domains is mapped to a standardized, model-grounded 3-tier classification:
- **Genuine:** Suspicion Score $< 35\%$ (Low suspicion, high textual reliability)
- **Suspicious:** $35\% \le \text{Suspicion Score} \le 65\%$ (Moderate suspicion, ambiguous or borderline lexical signal)
- **Potentially Fake:** Suspicion Score $> 65\%$ (High suspicion, strong alignment with synthetic/deceptive/filtered training patterns)

### B. Mathematical Reciprocity Invariant
For every review across all three domains:
- $\text{Suspicion Score} = P(\text{Target Class}) \times 100\%$
  - Hotel: $P(\text{Deceptive}) \times 100\%$
  - Product: $P(\text{CG}) \times 100\%$
  - Restaurant: $P(\text{Filtered (Y)}) \times 100\%$
- $\text{Individual Review Reliability Score} = 100 - \text{Suspicion Score} = P(\text{Baseline Class}) \times 100\%$
- **Invariant:** $\text{Suspicion Score} + \text{Review Reliability Score} \equiv 100.0\%$ (exact reciprocity)
- **Original Model Predictions Preserved:**
  - Hotel: `Truthful` vs `Deceptive`
  - Product: `OR` (Original Human) vs `CG` (Computer Generated GPT-2)
  - Restaurant: `Recommended (N)` vs `Filtered / Not Recommended (Y)`

### C. Application Architecture & Navigation
The Streamlit application provides dedicated workspaces and centralized multi-domain tools across 9 navigation pages:
1. `🏠 Home & Project Overview` — System architecture, domain descriptions, author credits, and operational disclaimers.
2. `🏨 Hotel Reviews` — Dedicated workspace with tabs for Single Review Analysis and Bulk CSV Analysis.
3. `🍽️ Restaurant Reviews` — Dedicated workspace with tabs for Single Review Analysis and Bulk CSV Analysis.
4. `📦 Product Reviews` — Dedicated workspace with tabs for Single Review Analysis and Bulk CSV Analysis.
5. `🔍 Single Review Analysis` — Multi-domain single review analysis tool with active domain selector dropdown.
6. `📁 Bulk CSV Analysis` — Multi-domain batch CSV analysis tool with active domain selector dropdown and downloadable scored CSV.
7. `📊 Review Analytics Dashboard` — System-wide aggregate metrics ($N$, Genuine count, Suspicious count, Potentially Fake count, Suspicious %, Potentially Fake %, and Overall Review Reliability Score as average of individual reliability scores), system-wide bar charts, and domain-specific breakdown tabs.
8. `🗄️ Database & History` — SQLite history with formatted display of Classification, Suspicion Score, Reliability Score, Original Prediction, Confidence, Review Text, and Timestamp.
9. `📚 Multi-Domain Benchmark Report` — Empirical benchmark comparisons, confusion matrices, and construct boundary documentation.

### D. Verification Results
- **`test_system.py`:** **7 / 7 PASS (100.0%)**
- **`scratch/test_app_integration.py`:** **25 / 25 PASS (100.0%)**
- **`scratch/test_final_multi_domain.py`:** **100% PASS** (Inference, Bulk CSV scoring, CSV downloads, Dashboard metrics, Reciprocity invariant, AppTest navigation across all 9 pages).
- **Database (`reviews.db`):** 26 legacy records fully preserved with backward-compatible dynamic score derivation.
- **Streamlit Live Server:** Active and responsive on port 8501.

---

## 7. Institution Branding Standardization & Live Browser End-to-End Verification

**Date:** November 2, 2026  
**Institution:** Punjab Engineering College (PEC)  
**Authors:** Himanshi (SID: 24106002), Samarth (SID: 24106046)  
**Branch:** B.Tech CSE (Data Science)

### A. Institution Name Standardization
Standardized the academic institution across all documentation and presentation layers to **Punjab Engineering College (PEC)**:
- Streamlit application sidebar: `Punjab Engineering College • B.Tech CSE (Data Science)`
- Streamlit Home Page metadata: `Institution: Punjab Engineering College`
- Project Report (`docs/PROJECT_REPORT.md`): `Institution: Punjab Engineering College (PEC), Chandigarh`
- System Architecture (`docs/SYSTEM_ARCHITECTURE.md`): `Institution: Punjab Engineering College (PEC)`
- Viva Preparation Guide (`docs/VIVA_PREPARATION.md`): `Institution: Punjab Engineering College (PEC)`
- README (`README.md`): `Institution: Punjab Engineering College (PEC)`

### B. Live Browser Manual End-to-End Verification
Executed automated real-browser user journeys on the running Streamlit server (`http://localhost:8501`) via Microsoft Edge (Playwright):
1. **Home & Project Overview:** Verified student credentials (Himanshi 24106002, Samarth 24106046) and Punjab Engineering College branding.
2. **Hotel Reviews Expert:** Verified Single Review inference ($S_{\text{susp}}=35.0\%$, $S_{\text{rel}}=65.0\%$, Genuine/Truthful), mathematical reciprocity invariant ($S_{\text{susp}} + S_{\text{rel}} \equiv 100\%$), and XAI token attributions.
3. **Restaurant Reviews Expert:** Verified Single Review inference ($S_{\text{susp}}=34.4\%$, $S_{\text{rel}}=65.6\%$, Genuine/Recommended) and reciprocity.
4. **Product Reviews Expert:** Verified Single Review inference ($S_{\text{susp}}=7.5\%$, $S_{\text{rel}}=92.5\%$, Genuine/Human OR) and reciprocity.
5. **Bulk CSV Analysis:** Verified batch ingestion of multi-row CSVs, real-time progress, summary metric cards, interactive dataframe display, downloadable enriched CSV, and error handling for missing/empty columns.
6. **Review Analytics Dashboard:** Verified all 7 aggregate KPIs ($N=55$, Overall Reliability Score $67.4\%$, Suspicious Reviews $\% = 34.5\%$, Potentially Fake Reviews $\% = 5.5\%$, Genuine $= 33$, Suspicious $= 19$, Fake $= 3$), reciprocity equations, and domain tabs.
7. **Database & History:** Verified persistent retrieval from SQLite (`reviews.db`), live keyword search filter, domain filter, and formatted metric columns.
8. **Multi-Domain Benchmark Report:** Verified empirical comparison tables, confusion matrix graphics, and cross-domain construct boundaries.
9. **Navigation & Responsiveness:** Verified zero runtime exceptions across all 9 sidebar routes.
10. **Screenshots:** Captured and saved 9 live screenshots into `screenshots/` and artifact directories.


