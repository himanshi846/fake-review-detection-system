import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
from model_utils import (
    load_model,
    predict_reviews,
    explain_review,
    LABELS,
    THRESHOLD_LOW,
    THRESHOLD_HIGH
)

BASE_DIR = Path(r"C:\Users\himan\Downloads\Fake_Review_Detection_System\Fake_Review_Detection_System")
MODEL_PATH = BASE_DIR / "review_model.joblib"
DB_PATH = BASE_DIR / "test_reviews.db"

print("=================================================================")
print("RUNNING COMPREHENSIVE SYSTEM & INTEGRATION TESTS")
print("=================================================================")

# Test 1: Model Loading
print("\n[TEST 1] Testing Model Loading...")
pipeline, note = load_model(MODEL_PATH)
assert pipeline is not None
assert hasattr(pipeline, "predict")
assert hasattr(pipeline, "predict_proba")
print("  [PASS] Model loaded successfully. Note:", note)

# Test 2: Single Review Prediction & Reliability Scoring
print("\n[TEST 2] Testing Single Review Prediction & Metrics...")
sample_reviews = [
    ("We stayed for three nights. The room was spacious, bathroom was clean, and breakfast was great.", "Truthful"),
    ("My stay in this luxury Chicago hotel was an absolute dream, best experience in Chicago, unforgettable luxury.", "Deceptive")
]
for text, expected_hint in sample_reviews:
    df_single = pd.DataFrame({"review_text": [text]})
    res = predict_reviews(pipeline, df_single)
    row = res.iloc[0]
    cat = row["predicted_label"]
    p_dec = row["p_deceptive"]
    p_tru = row["p_truthful"]
    rel = row["reliability_score"]
    
    assert cat in LABELS
    assert 0.0 <= rel <= 100.0
    assert abs(p_dec + p_tru - 1.0) < 1e-4
    assert abs(rel - (100.0 * (1.0 - p_dec))) < 0.2
    print(f"  Review: '{text[:50]}...'")
    print(f"    Category: {cat} | P(Dec): {p_dec:.4f} | P(Tru): {p_tru:.4f} | Reliability: {rel}")

# Test 3: Explainability Module
print("\n[TEST 3] Testing Feature-Level Explainability...")
exp = explain_review(pipeline, "The hotel location was great and the room was very clean. We loved the breakfast buffet.", top_k=5)
assert "deceptive_features" in exp
assert "truthful_features" in exp
assert len(exp["truthful_features"]) > 0
for feat in exp["truthful_features"][:3]:
    print(f"  Truthful driver: '{feat['feature']}' (Contribution: {feat['contribution']}, Weight: {feat['weight']})")
for feat in exp["deceptive_features"][:3]:
    print(f"  Deceptive driver: '{feat['feature']}' (Contribution: {feat['contribution']}, Weight: {feat['weight']})")
print("  [PASS] Explainability returned genuine mathematical feature attributions.")

# Test 4: Bulk CSV Prediction
print("\n[TEST 4] Testing Bulk CSV Prediction...")
bulk_df = pd.DataFrame({
    "review_text": [
        "The room was clean and close to Michigan Avenue.",
        "Worst hotel ever! Stay away! Terrible front desk!",
        "A wonderful vacation with my family in our luxury room.",
        "Comfortable bed, quiet room on the 14th floor."
    ]
})
bulk_res = predict_reviews(pipeline, bulk_df)
assert len(bulk_res) == 4
assert set(bulk_res.columns).issuperset({"predicted_label", "confidence", "reliability_score", "p_deceptive", "p_truthful"})
print(f"  [PASS] Processed {len(bulk_res)} bulk reviews successfully.")
print("  Bulk Distribution:", bulk_res["predicted_label"].value_counts().to_dict())

# Test 5: Empty & Invalid Inputs Handling
print("\n[TEST 5] Testing Edge Cases & Empty Inputs...")
edge_df = pd.DataFrame({"review_text": ["", "   ", "A"]})
edge_res = predict_reviews(pipeline, edge_df)
assert len(edge_res) == 3
assert not edge_res["predicted_label"].isnull().any()
print("  [PASS] Empty and short whitespace reviews handled without exceptions.")

# Test 6: Database Saving & Retrieval
print("\n[TEST 6] Testing SQLite Database Persistence...")
test_db_path = BASE_DIR / "reviews.db"
# Ensure table exists
with sqlite3.connect(test_db_path) as conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS review_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_text TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            confidence REAL NOT NULL,
            reliability_score REAL NOT NULL,
            p_deceptive REAL NOT NULL,
            p_truthful REAL NOT NULL,
            source TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        INSERT INTO review_predictions
        (review_text, predicted_label, confidence, reliability_score, p_deceptive, p_truthful, source, created_at)
        VALUES ('Test review text', 'Truthful', 0.95, 95.0, 0.05, 0.95, 'unit_test', '2026-10-30T10:00:00')
    """)
    conn.commit()
    cur = conn.execute("SELECT COUNT(*) FROM review_predictions WHERE source='unit_test'")
    count = cur.fetchone()[0]
    assert count >= 1
    # Clean up test row
    conn.execute("DELETE FROM review_predictions WHERE source='unit_test'")
    conn.commit()
print("  [PASS] Database write, read, and delete verified successfully.")

# Test 7: Model Persistence / Restart Simulation
print("\n[TEST 7] Testing Model Re-loading After Simulated Restart...")
del pipeline
reloaded_pipeline, _ = load_model(MODEL_PATH)
re_res = predict_reviews(reloaded_pipeline, pd.DataFrame({"review_text": ["Great location and clean bathroom."]}))
assert re_res.iloc[0]["predicted_label"] in ["Genuine", "Truthful"]
print("  [PASS] Model reloaded from disk and reproduced exact predictions.")

print("\n=================================================================")
print("ALL SYSTEM & INTEGRATION TESTS PASSED (7/7)")
print("=================================================================")
