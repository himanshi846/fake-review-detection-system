import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = Path(r"C:\Users\himan\Downloads\Fake_Review_Detection_System\Fake_Review_Detection_System")
DATA_DIR = BASE_DIR / "data"
RAW_CSV = DATA_DIR / "deceptive-opinion.csv"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
MODEL_PATH = BASE_DIR / "review_model.joblib"
REPORT_JSON_PATH = DATA_DIR / "evaluation_report.json"
CONFUSION_JSON_PATH = DATA_DIR / "confusion_matrix.json"

# =====================================================================
# STEP 1: Leakage-Free Stratified Group Splitting
# =====================================================================
print("Loading raw dataset from:", RAW_CSV)
df = pd.read_csv(RAW_CSV)

# Group identical review texts so they NEVER cross the train/test split
text_to_group = {text: idx for idx, text in enumerate(df["text"].unique())}
df["group"] = df["text"].map(text_to_group)
df["strat_group"] = df["deceptive"].astype(str) + "_" + df["polarity"].astype(str)

# StratifiedGroupKFold guarantees 80/20 partition without any group overlap
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_indices, test_indices = next(sgkf.split(df, df["strat_group"], df["group"]))

train_df = df.iloc[train_indices].copy()
test_df = df.iloc[test_indices].copy()

# Integrity Checks
train_texts = set(train_df["text"])
test_texts = set(test_df["text"])
overlap = train_texts.intersection(test_texts)

assert len(overlap) == 0, f"Data leakage detected! Overlapping texts: {len(overlap)}"
assert len(train_df) == 1280, f"Expected 1280 train rows, got {len(train_df)}"
assert len(test_df) == 320, f"Expected 320 test rows, got {len(test_df)}"

# Ensure original schema is saved without temporary helper columns
cols_to_save = ["deceptive", "hotel", "polarity", "source", "text"]
train_df[cols_to_save].to_csv(TRAIN_CSV, index=False)
test_df[cols_to_save].to_csv(TEST_CSV, index=False)
print(f"Leakage check PASSED. Zero overlap between train and test.")
print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")

# =====================================================================
# STEP 2: Model Training (TF-IDF + Logistic Regression)
# ONLY train.csv is used. test.csv is completely untouched.
# =====================================================================
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42
    ))
])

print("\nFitting pipeline ONLY on train.csv...")
pipeline.fit(train_df["text"], train_df["deceptive"])

# Save the trained model
joblib.dump(pipeline, MODEL_PATH)
print(f"Trained model saved to: {MODEL_PATH}")

# =====================================================================
# STEP 3: Evaluation on Untouched test.csv
# =====================================================================
X_test = test_df["text"]
y_test = test_df["deceptive"]

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)
classes = list(pipeline.named_steps["classifier"].classes_)

# Metrics
acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
macro_prec = precision_score(y_test, y_pred, average="macro")
macro_rec = recall_score(y_test, y_pred, average="macro")
macro_f1 = f1_score(y_test, y_pred, average="macro")

clf_dict = classification_report(y_test, y_pred, output_dict=True)
clf_str = classification_report(y_test, y_pred, digits=4)
cm = confusion_matrix(y_test, y_pred, labels=classes)

cm_dict = {
    "labels": classes,
    "matrix": cm.tolist(),
    "breakdown": {
        f"actual_{actual}_pred_{pred}": int(cm[i][j])
        for i, actual in enumerate(classes)
        for j, pred in enumerate(classes)
    }
}

evaluation_summary = {
    "model_name": "TF-IDF + Logistic Regression",
    "dataset": "Ott Deceptive Opinion Spam Corpus (Chicago Hotel Dataset)",
    "vectorizer_config": {
        "ngram_range": [1, 2],
        "min_df": 2,
        "sublinear_tf": True,
        "strip_accents": "unicode",
        "vocabulary_size": len(pipeline.named_steps["tfidf"].vocabulary_)
    },
    "classifier_config": {
        "model": "LogisticRegression",
        "C": 1.0,
        "max_iter": 1000,
        "random_state": 42
    },
    "data_partitions": {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "overlap_count": len(overlap)
    },
    "overall_metrics": {
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "macro_f1": round(float(macro_f1), 4)
    },
    "class_wise_metrics": {
        cls: {
            "precision": round(float(clf_dict[cls]["precision"]), 4),
            "recall": round(float(clf_dict[cls]["recall"]), 4),
            "f1-score": round(float(clf_dict[cls]["f1-score"]), 4),
            "support": int(clf_dict[cls]["support"])
        }
        for cls in classes
    },
    "confusion_matrix": cm_dict
}

with open(REPORT_JSON_PATH, "w") as f:
    json.dump(evaluation_summary, f, indent=2)

with open(CONFUSION_JSON_PATH, "w") as f:
    json.dump(cm_dict, f, indent=2)

print("\n=== CLASSIFICATION REPORT ===")
print(clf_str)

print("\n=== CONFUSION MATRIX (Labels: {}) ===".format(classes))
print(pd.DataFrame(cm, index=[f"Actual {c}" for c in classes], columns=[f"Pred {c}" for c in classes]))

print("\n=== EVALUATION SUMMARY JSON ===")
print(json.dumps(evaluation_summary["overall_metrics"], indent=2))
