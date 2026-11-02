# model_utils.py
"""
Model utilities and inference engine for the Fake Review Detection System.
Supports multi-domain task experts:
  1. Hotel Reviews (Ott Deceptive Opinion Spam Corpus)
  2. Product Reviews (Amazon 40K Human vs GPT-2 Corpus)
  3. Restaurant Reviews (YelpChi Platform Filter Corpus)
"""

from pathlib import Path
import html
import re
import joblib
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

# =====================================================================
# Domain Configurations & Metadata
# =====================================================================
COMMON_CATEGORIES = ["Genuine", "Suspicious", "Potentially Fake"]
LABELS = COMMON_CATEGORIES  # Direct alias for full backwards compatibility

__all__ = [
    "COMMON_CATEGORIES",
    "LABELS",
    "DOMAIN_CONFIGS",
    "THRESHOLD_LOW",
    "THRESHOLD_HIGH",
    "load_domain_model",
    "load_model",
    "predict_domain_reviews",
    "predict_reviews",
    "explain_domain_review",
    "explain_review"
]

DOMAIN_CONFIGS = {
    "hotel": {
        "key": "hotel",
        "name": "Hotel Reviews",
        "icon": "🏨",
        "dataset_name": "Ott Deceptive Opinion Spam Corpus (Chicago Hotels)",
        "citation": "Ott et al. (ACL 2011, NAACL 2013)",
        "task_description": "Psychological Deception Classification (Truthful vs Crowdsourced Deceptive)",
        "model_path": BASE_DIR / "experiments" / "ott_hotel" / "ott_hotel_model.joblib",
        "root_model_path": BASE_DIR / "ott_hotel_model.joblib",
        "classes": ["deceptive", "truthful"],
        "class_0_label": "truthful",
        "class_0_name": "Truthful",
        "class_1_label": "deceptive",
        "class_1_name": "Deceptive",
        "threshold_low": 0.35,
        "threshold_high": 0.65,
        "labels": ["Genuine", "Suspicious", "Potentially Fake"],
        "risk_bands": ["Genuine", "Suspicious", "Potentially Fake"],
        "test_accuracy": "89.69%",
        "test_samples": 320,
        "sample_reviews": [
            {
                "label": "Truthful Example",
                "text": "We stayed at this hotel for a weekend getaway. Check-in was swift, the room was spotless with a comfortable king bed, and the location near Michigan Avenue made sightseeing very convenient. Breakfast had good variety."
            },
            {
                "label": "Deceptive Example",
                "text": "My stay at this Chicago luxury hotel was an absolute nightmare from beginning to end! The staff were exceedingly rude, the suite was dirty, and everything was grossly overpriced. Do not ever book here under any circumstances!"
            }
        ],
        "limitations": "Trained on 1,600 reviews across 20 luxury Chicago hotels (Ott et al., 2011, 2013). Evaluates crowdsourced deceptive reviews versus truthful TripAdvisor reviews within this specific benchmark setting."
    },
    "product": {
        "key": "product",
        "name": "Product Reviews",
        "icon": "📦",
        "dataset_name": "Amazon Reviews 40K (Salminen et al., 2022)",
        "citation": "Salminen et al. (ICWSM / Expert Systems 2022)",
        "task_description": "Human vs GPT-2 Text Classification (Human Original 'OR' vs GPT-2 'CG')",
        "model_path": BASE_DIR / "experiments" / "amazon_product" / "amazon_product_model.joblib",
        "classes": ["CG", "OR"],
        "class_0_label": "OR",
        "class_0_name": "Human-Written Original (OR)",
        "class_1_label": "CG",
        "class_1_name": "GPT-2 Generated Synthetic (CG)",
        "threshold_low": 0.35,
        "threshold_high": 0.65,
        "labels": ["Genuine", "Suspicious", "Potentially Fake"],
        "risk_bands": ["Genuine", "Suspicious", "Potentially Fake"],
        "test_accuracy": "93.36%",
        "test_samples": 8087,
        "sample_reviews": [
            {
                "label": "Human Original (OR) Example",
                "text": "I bought this cordless impact driver for a deck rebuilding project. Battery life lasted all afternoon through 3-inch deck screws, and the LED work light really helped in dim corners. Solid tool."
            },
            {
                "label": "GPT-2 Synthetic (CG) Example",
                "text": "This product is an excellent addition to my home. It is very sturdy and high quality. The price is reasonable and shipping was fast. I would recommend this product to everyone."
            }
        ],
        "limitations": "Implemented solely as a TF-IDF Vectorizer + Logistic Regression model distinguishing human-written reviews (OR) from fine-tuned GPT-2-generated synthetic text (CG). OR indicates human authorship and CG indicates synthetic text; neither label confirms genuine honesty or commercial fraud."
    },
    "restaurant": {
        "key": "restaurant",
        "name": "Restaurant Reviews",
        "icon": "🍽️",
        "dataset_name": "YelpChi Restaurant Dataset (Rayana & Akoglu, KDD 2015)",
        "citation": "Rayana & Akoglu (ACM SIGKDD 2015)",
        "task_description": "Platform Filter Classification (Recommended 'N' vs Filtered 'Y')",
        "model_path": BASE_DIR / "experiments" / "yelp_restaurant" / "yelp_restaurant_model.joblib",
        "classes": ["N", "Y"],
        "class_0_label": "N",
        "class_0_name": "Platform Recommended (N)",
        "class_1_label": "Y",
        "class_1_name": "Filtered / Not Recommended (Y)",
        "threshold_low": 0.35,
        "threshold_high": 0.65,
        "labels": ["Genuine", "Suspicious", "Potentially Fake"],
        "risk_bands": ["Genuine", "Suspicious", "Potentially Fake"],
        "test_accuracy": "77.79% (Balanced: 53.88%, Macro F1: 51.89%, Filtered Recall: 9.60%)",
        "test_samples": 5390,
        "sample_reviews": [
            {
                "label": "Recommended (N) Example",
                "text": "Came here on a busy Friday evening with friends. The wood-fired margherita pizza had a perfectly blistered crust and the house salad was crisp. Service was attentive despite the packed dining room."
            },
            {
                "label": "Filtered (Y) Example",
                "text": "BEST RESTAURANT IN THE ENTIRE CITY!!!!! You must order everything on the menu right now! Fantastic service and incredible atmosphere! 5 stars forever!"
            }
        ],
        "limitations": "Evaluates Yelp's platform recommendation filter: Recommended ('N') vs Filtered / Not Recommended ('Y'). The dataset exhibits natural class imbalance (76.98% Recommended vs 23.02% Filtered). The text-only model achieves 77.79% overall accuracy (98.17% Recommended recall, 9.60% Filtered recall, 53.88% Balanced accuracy). Neither label confirms genuine or fake reviews."
    }
}

# Legacy constants for backwards compatibility
THRESHOLD_LOW = DOMAIN_CONFIGS["hotel"]["threshold_low"]
THRESHOLD_HIGH = DOMAIN_CONFIGS["hotel"]["threshold_high"]
LABELS = DOMAIN_CONFIGS["hotel"]["labels"]

# =====================================================================
# Model Loading
# =====================================================================
def load_domain_model(domain_key):
    """
    Loads the trained scikit-learn pipeline for the specified domain.
    """
    if domain_key not in DOMAIN_CONFIGS:
        raise ValueError(f"Unknown domain: {domain_key}. Supported domains: {list(DOMAIN_CONFIGS.keys())}")
    
    cfg = DOMAIN_CONFIGS[domain_key]
    path = cfg["model_path"]
    
    # Fallback to root model copy for hotel if needed
    if not path.exists() and "root_model_path" in cfg and cfg["root_model_path"].exists():
        path = cfg["root_model_path"]
        
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model for {cfg['name']} not found at {path}. Please run training first."
        )
        
    pipeline = joblib.load(path)
    note = f"Loaded {cfg['name']} expert ({cfg['dataset_name']}, Test Accuracy: {cfg['test_accuracy']})."
    return pipeline, note

def load_model(model_path=None):
    """
    Backwards-compatible loader. Defaults to Ott Hotel model.
    """
    if model_path is None or str(model_path).endswith("review_model.joblib") or str(model_path).endswith("ott_hotel_model.joblib"):
        return load_domain_model("hotel")
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    pipeline = joblib.load(path)
    return pipeline, f"Loaded model from {path.name}"

# =====================================================================
# Prediction Logic
# =====================================================================
def predict_domain_reviews(pipeline, df, domain_key):
    """
    Accepts a DataFrame containing 'review_text'.
    Applies the domain-specific TF-IDF + Logistic Regression pipeline.
    Calculates class probabilities, binary label, categorical label with uncertainty,
    and confidence scores.
    """
    if domain_key not in DOMAIN_CONFIGS:
        raise ValueError(f"Unknown domain: {domain_key}")
        
    cfg = DOMAIN_CONFIGS[domain_key]
    out = df.copy()
    if "review_text" not in out.columns:
        raise ValueError("Missing 'review_text' column in input DataFrame.")
        
    texts = out["review_text"].fillna("").astype(str)
    clf = pipeline.named_steps["classifier"]
    classes = list(clf.classes_)
    
    c0_label = cfg["class_0_label"]
    c1_label = cfg["class_1_label"]
    
    idx_0 = classes.index(c0_label)
    idx_1 = classes.index(c1_label)
    
    probabilities = pipeline.predict_proba(texts)
    p_0 = probabilities[:, idx_0]
    p_1 = probabilities[:, idx_1]
    
    low_t = cfg["threshold_low"]
    high_t = cfg["threshold_high"]
    labels_3 = cfg["labels"]
    
    pred_categories = []
    confidence_scores = []
    reliability_metrics = []
    
    for prob0, prob1 in zip(p_0, p_1):
        # 3-tier categorization based on target class (prob1)
        if prob1 < low_t:
            cat = labels_3[0]
        elif prob1 > high_t:
            cat = labels_3[2]
        else:
            cat = labels_3[1]
        pred_categories.append(cat)
        
        # Confidence is the maximum posterior probability
        conf = max(prob0, prob1)
        confidence_scores.append(round(float(conf), 4))
        
        # Reliability / Alignment index: 100 * prob of baseline class (e.g. truthful / human / recommended)
        rel = max(0.0, min(100.0, 100.0 * prob0))
        reliability_metrics.append(round(float(rel), 1))
        
    out["predicted_label"] = pred_categories
    out["raw_prediction"] = [c1_label if p1 >= 0.5 else c0_label for p1 in p_1]
    out["prob_baseline"] = np.round(p_0, 4)
    out["prob_target"] = np.round(p_1, 4)
    out[f"prob_{c0_label}"] = np.round(p_0, 4)
    out[f"prob_{c1_label}"] = np.round(p_1, 4)
    out["confidence"] = confidence_scores
    out["model_domain"] = cfg["name"]
    out["reliability_score"] = reliability_metrics
    
    # Universal ReviewTrust SRS Metrics across all three domains:
    # 1. Suspicion Score (0-100%) = P(target class) * 100
    # 2. Individual Review Reliability Score (0-100%) = 100 - Suspicion Score
    # 3. Common 3-tier Classification: Genuine (<35%), Suspicious (35%-65%), Potentially Fake (>65%)
    out["suspicion_score"] = np.round(p_1 * 100.0, 1)
    out["individual_reliability_score"] = np.round(100.0 - (p_1 * 100.0), 1)
    out["review_classification"] = pred_categories
    out["predicted_label"] = pred_categories
    out["risk_category"] = pred_categories

    # Domain-specific original model binary predictions
    if domain_key == "hotel":
        out["binary_prediction"] = ["Deceptive" if p1 >= 0.5 else "Truthful" for p1 in p_1]
    elif domain_key == "product":
        out["binary_prediction"] = ["GPT-2 Generated (CG)" if p1 >= 0.5 else "Human-Written (OR)" for p1 in p_1]
    elif domain_key == "restaurant":
        out["binary_prediction"] = ["Filtered / Not Recommended (Y)" if p1 >= 0.5 else "Platform Recommended (N)" for p1 in p_1]
    else:
        out["binary_prediction"] = out["raw_prediction"]

    out["original_model_prediction"] = out["binary_prediction"]
        
    return out

def predict_reviews(pipeline, df):
    """
    Backwards-compatible prediction function (defaults to hotel semantics).
    """
    # Detect domain from pipeline classes
    classes = list(pipeline.named_steps["classifier"].classes_)
    if "CG" in classes or "OR" in classes:
        domain = "product"
    elif "N" in classes or "Y" in classes:
        domain = "restaurant"
    else:
        domain = "hotel"
        
    res = predict_domain_reviews(pipeline, df, domain)
    # Ensure legacy column names exist for backwards compatibility
    if "p_deceptive" not in res.columns:
        res["p_deceptive"] = res["prob_target"]
    if "p_truthful" not in res.columns:
        res["p_truthful"] = res["prob_baseline"]
    return res

# =====================================================================
# Explainability & Feature Contribution Engine
# =====================================================================
def explain_domain_review(pipeline, text, domain_key, top_k=6):
    """
    Performs mathematical feature attribution using Logistic Regression weights
    and TF-IDF token frequencies for the specified domain.
    """
    if not text or not str(text).strip():
        return {
            "baseline_features": [],
            "target_features": [],
            "top_baseline_terms": [],
            "top_target_terms": []
        }
        
    cfg = DOMAIN_CONFIGS.get(domain_key, DOMAIN_CONFIGS["hotel"])
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["classifier"]
    classes = list(clf.classes_)
    
    c0_label = cfg["class_0_label"]
    c1_label = cfg["class_1_label"]
    
    idx_1 = classes.index(c1_label)
    
    # In scikit-learn binary LogisticRegression, coef_[0] corresponds to classes[1]
    # Positive weights push towards classes[1] (target class), negative weights push towards classes[0]
    weights_target = clf.coef_[0] if idx_1 == 1 else -clf.coef_[0]
    weights_baseline = -weights_target
    
    feature_names = tfidf.get_feature_names_out()
    vec = tfidf.transform([str(text)])
    cx = vec.tocoo()
    
    baseline_drivers = []
    target_drivers = []
    
    for j, v in zip(cx.col, cx.data):
        feat = str(feature_names[j])
        wb = float(weights_baseline[j])
        wt = float(weights_target[j])
        cb = float(v * wb)
        ct = float(v * wt)
        
        if cb > 0:
            baseline_drivers.append({
                "feature": feat,
                "weight": round(wb, 4),
                "tfidf": round(float(v), 4),
                "contribution": round(cb, 4)
            })
        if ct > 0:
            target_drivers.append({
                "feature": feat,
                "weight": round(wt, 4),
                "tfidf": round(float(v), 4),
                "contribution": round(ct, 4)
            })
            
    baseline_drivers.sort(key=lambda x: x["contribution"], reverse=True)
    target_drivers.sort(key=lambda x: x["contribution"], reverse=True)
    
    top_base = baseline_drivers[:top_k]
    top_targ = target_drivers[:top_k]
    
    return {
        "baseline_label": cfg["class_0_name"],
        "target_label": cfg["class_1_name"],
        "baseline_features": top_base,
        "target_features": top_targ,
        "top_baseline_terms": [x["feature"] for x in top_base],
        "top_target_terms": [x["feature"] for x in top_targ],
        # Backwards compatibility keys
        "truthful_features": top_base,
        "deceptive_features": top_targ,
        "top_truthful_terms": [x["feature"] for x in top_base],
        "top_deceptive_terms": [x["feature"] for x in top_targ]
    }

def explain_review(pipeline, text, top_k=6):
    """
    Backwards compatible explainability function.
    """
    return explain_domain_review(pipeline, text, "hotel", top_k=top_k)
