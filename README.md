# ReviewTrust — Fake Review Detection and Review Reliability Analysis System

**Institution:** Punjab Engineering College (PEC)  
**B.Tech Minor Project &bull; Computer Science & Engineering (Data Science)**  
**Project Team:** Himanshi (SID: 24106002), Samarth (SID: 24106046) &nbsp;|&nbsp; **Academic Year:** 2026  
**Technology Stack:** Python, Scikit-learn, TF-IDF, Logistic Regression, Streamlit, Pandas, NumPy, SQLite, Matplotlib, Playwright  

---

## 1. Project Overview & Problem Statement

Online customer reviews across hospitality, e-commerce, and dining significantly influence modern purchasing decisions. However, review ecosystems are routinely compromised by opinion spam:
1. **Hotel Reviews:** Human crowdsourced deceptive fabrications (positive or negative lies written for financial incentive).
2. **Product Reviews:** Machine-generated synthetic reviews produced by modern generative language models (e.g. fine-tuned GPT-2).
3. **Restaurant Reviews:** Platform-curated reviews filtered by algorithmic spam detection heuristics.

**ReviewTrust** addresses the critical limitation of existing commercial tools: treating "fake reviews" as a single monolithic target. Our research demonstrates that these three domains represent fundamentally distinct statistical constructs. ReviewTrust deploys three dedicated, leakage-free machine learning task experts, combined with mathematical feature attribution (Explainable AI) and local relational database persistence.

---

## 2. Key Features

- **Multi-Domain Task Experts:**
  - 🏨 **Hotel Reviews Expert:** Evaluates crowdsourced deceptive reviews versus truthful TripAdvisor reviews using prediction labels **Truthful** and **Deceptive** based on the canonical Ott Deceptive Opinion Spam Corpus (**89.69% Test Accuracy**; Ott et al., 2011, 2013).
  - 📦 **Product Reviews Expert:** Classifies human-written reviews (`OR`) vs fine-tuned GPT-2 synthetic text (`CG`) across 10 Amazon product categories (**93.36% Test Accuracy**; Salminen et al., 2022) using a TF-IDF Vectorizer + Logistic Regression pipeline.
  - 🍽️ **Restaurant Reviews Expert:** Predicts platform recommendation filter decisions—Recommended (`N`) vs Filtered / Not Recommended (`Y`)—based on the YelpChi benchmark (**77.79% Test Accuracy**, **53.88% Balanced Accuracy**, **9.60% Filtered Recall**; Rayana & Akoglu, 2015). Discloses natural class imbalance (76.98% Recommended vs 23.02% Filtered). Clearly notes that neither label confirms genuine or fake reviews.
- **Three-Tier Probabilistic Uncertainty Classification:**
  - Maps continuous probabilities into high-certainty and borderline uncertainty bands (e.g. *Truthful*, *Uncertain (Borderline)*, *Deceptive*) to prevent overconfident classifications.
- **Explainable AI (XAI) Feature Attribution:**
  - Extracts exact n-gram token impact scores ($\text{Contribution} = \text{TF-IDF} \times \text{Logistic Regression Weight}$) and visually highlights influential words in real time.
- **Bulk CSV Analysis:**
  - Ingests multi-row CSV files, provides intelligent text-column auto-detection, executes batch inference without dropping user metadata, and exports enriched CSVs.
- **Persistent SQLite Audit Database:**
  - Stores all single and bulk prediction records in a local SQLite relational database (`reviews.db`) with parameterized queries to prevent SQL injection.
- **Interactive Review Analytics Dashboard:**
  - Visualizes domain volumes, category distributions, average confidence scores, and historical trends.

---

## 3. Academic Model Benchmark & Evaluation Summary

All models were evaluated strictly on separate, held-out test sets created using `StratifiedGroupKFold` ($k=5$, seed 42) to guarantee **0 text-hash overlap** and zero data leakage.

| Evaluation Metric | 🏨 Hotel Model (Ott Benchmark) | 📦 Product Model (Amazon 40K) | 🍽️ Restaurant Model (YelpChi) |
| :--- | :---: | :---: | :---: |
| **Prediction Task** | Truthful vs Deceptive | Human (`OR`) vs GPT-2 (`CG`) | Recommended (`N`) vs Filtered (`Y`) |
| **Corpus Reference** | Ott et al. (ACL 2011, NAACL 2013) | Salminen et al. (Expert Systems 2022) | Rayana & Akoglu (KDD 2015) |
| **Total Samples** | 1,600 reviews | 40,432 reviews | 26,955 reviews |
| **Training Samples (80%)** | 1,280 reviews | 32,345 reviews | 21,565 reviews |
| **Held-Out Test Set (20%)** | 320 reviews | 8,087 reviews | 5,390 reviews |
| **Vocabulary Size** | 23,047 n-grams | 139,582 n-grams | 196,654 n-grams |
| **Overall Accuracy** | **89.69%** | **93.36%** | **77.79%** |
| **Balanced Accuracy** | **89.69%** | **93.36%** | **53.88%** |
| **Macro Average F1** | **0.8969** | **0.9336** | **0.5189** |
| **Macro Precision** | **0.8970** | **0.9341** | **0.6972** |
| **Macro Recall** | **0.8969** | **0.9336** | **0.5388** |
| **Artifact Path** | `experiments/ott_hotel/` | `experiments/amazon_product/` | `experiments/yelp_restaurant/` |

---

## 4. Application Screenshots

Authentic screenshots captured directly from the live application running on Windows:

| View | Screenshot File | Description |
| :--- | :--- | :--- |
| **1. Home Page** | `docs/screenshots/01_home_page.png` | Project overview, student info, problem statement, and supported domains |
| **2. Hotel Analysis** | `docs/screenshots/02_hotel_analysis.png` | Single review analysis with Ott model, probability meters, and XAI feature tables |
| **3. Product Analysis** | `docs/screenshots/03_product_analysis.png` | Human (`OR`) vs GPT-2 (`CG`) text discrimination across product categories |
| **4. Restaurant Analysis** | `docs/screenshots/04_restaurant_analysis.png` | Yelp platform recommendation filter prediction and probability bars |
| **5. Bulk CSV Analysis** | `docs/screenshots/05_bulk_csv_analysis.png` | Batch processing interface with column picker and CSV export |
| **6. Analytics Dashboard** | `docs/screenshots/06_dashboard.png` | Domain breakdown charts, category distribution, and KPI summary cards |
| **7. SQL Database** | `docs/screenshots/07_database_history.png` | SQLite history logs with domain filter, search bar, and CSV export |

---

## 5. Project Directory Structure

```text
Fake_Review_Detection_System/
│
├── app.py                             # Main Streamlit web application
├── model_utils.py                     # Multi-domain inference & explainability engine
├── db_utils.py                        # SQLite persistence & query module
├── requirements.txt                   # Production dependencies
├── reviews.db                         # SQLite audit relational database
│
├── candidate_datasets/                # Original academic source datasets
│   ├── amazon_reviews_40k.csv         # 40,432 Amazon reviews (Salminen et al.)
│   ├── yelp_restaurant_reviews.csv    # 26,958 Yelp reviews (Rayana & Akoglu)
│   └── dataset_inventory.csv          # Audit inventory of all evaluated datasets
│
├── prepared_datasets/                 # Curated, leak-free train/test splits
│   ├── ott_hotel_train.csv            # 1,280 reviews (50% truthful, 50% deceptive)
│   ├── ott_hotel_test.csv             # 320 reviews held-out
│   ├── amazon_product_train.csv       # 32,345 reviews (50% OR, 50% CG)
│   ├── amazon_product_test.csv        # 8,087 reviews held-out
│   ├── yelp_restaurant_train.csv      # 21,565 reviews (77% N, 23% Y)
│   ├── yelp_restaurant_test.csv       # 5,390 reviews held-out
│   └── split_verification_report.json # Cryptographic zero-leakage verification
│
├── experiments/                       # Model artifacts and academic evaluation reports
│   ├── COMPARATIVE_EVALUATION.md      # Cross-domain academic comparative analysis
│   ├── ott_hotel/                     # Hotel model (.joblib, report, plot, config, preds)
│   ├── amazon_product/                # Product model (.joblib, report, plot, config, preds)
│   └── yelp_restaurant/               # Restaurant model (.joblib, report, plot, config, preds)
│
├── docs/                              # Academic project submission documentation
│   ├── PROJECT_REPORT.md              # Full 21-section academic minor project report
│   ├── VIVA_PREPARATION.md            # 30+ comprehensive viva Q&A guide
│   ├── SYSTEM_ARCHITECTURE.md         # System architecture diagram & design specs
│   └── screenshots/                   # Genuine high-res UI screenshots
│
└── scratch/                           # Verification & test automation scripts
    ├── test_app_integration.py        # 25-point automated integration test suite
    └── capture_genuine_screenshots.py # Playwright headless screenshot capture
```

---

## 6. How to Run on Windows

### Step 1: Open Terminal in Project Directory
Open PowerShell or Command Prompt in the project folder:
```powershell
cd C:\Users\himan\Downloads\Fake_Review_Detection_System\Fake_Review_Detection_System
```

### Step 2: Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### Step 3: Run the 25-Point Verification Test Suite (Optional but Recommended)
```powershell
python scratch/test_app_integration.py
```
*Expected Output: `TEST RESULTS: 25/25 TESTS PASSED (100.0%)`.*

### Step 4: Launch the Streamlit Web Application
```powershell
python -m streamlit run app.py
```

The application will start and display your local access URL:
```text
Local URL: http://localhost:8501
```
Open `http://localhost:8501` in Microsoft Edge, Google Chrome, or any web browser.

---

## 7. Critical Methodological Boundaries & Limitations

1. **Independent Learning Constructs:** Do not compare raw test accuracies as if they measure the same task:
   - **Amazon Model (93.36% Accuracy):** Implemented solely as a TF-IDF Vectorizer + Logistic Regression pipeline distinguishing human-written original reviews (`OR`) from fine-tuned GPT-2-generated synthetic text (`CG`). `OR` indicates human authorship (not verified honesty), and `CG` indicates synthetic text (not confirmed commercial fraud).
   - **Yelp Model (77.79% Accuracy):** Evaluates platform recommendation decisions with labels Recommended (`N`) and Filtered / Not Recommended (`Y`) under natural class imbalance (76.98% Recommended vs 23.02% Filtered). The text-only classifier achieves 98.17% recall on Recommended reviews and 9.60% recall on Filtered reviews (Balanced Accuracy: 53.88%, Macro F1: 51.89%). Crucially, neither label confirms genuine or fake reviews.
   - **Ott Hotel Model (89.69% Accuracy):** Uses Truthful and Deceptive prediction labels to evaluate crowdworker deceptive reviews versus truthful TripAdvisor reviews across 20 Chicago luxury hotels (Ott et al., 2011, 2013). Benchmark limitations apply: it evaluates crowdsourced deception in hotel reviews and does not evaluate e-commerce products or restaurants.
2. **Authorship and Platform Curation vs Commercial Fraud:**
   - In product reviews, an authentic human review (`OR`) can still be commercially biased, while an AI-assisted review (`CG`) can describe an honest consumer experience.
   - In restaurant reviews, the labels represent Yelp's automated recommendation filter (Recommended vs Filtered / Not Recommended); neither label confirms genuine or fake reviews.
3. **Generative Model Horizon:** The AI product detector was trained on fine-tuned GPT-2 text (Salminen et al., 2022). Detection rates on modern frontier LLMs (GPT-4o, Claude 3.5, Gemini) will differ due to higher fluency.
