# db_utils.py
"""
Database utilities for the Fake Review Detection System.
Implements SQLite storage with parameterized queries, backwards-compatible
schema migration, search, filtering, and summary analytics.
"""

from datetime import datetime
from pathlib import Path
import sqlite3
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "reviews.db"

def get_connection(db_path=None):
    path = db_path or DEFAULT_DB_PATH
    return sqlite3.connect(path)

def init_db(db_path=None):
    """
    Initializes the database table and migrates missing columns if needed.
    """
    conn = get_connection(db_path)
    try:
        with conn:
            # Create base table with comprehensive schema
            conn.execute("""
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
                )
            """)
            
            # Check existing columns and add any missing ones for schema evolution
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(review_predictions);")
            existing_cols = {col[1] for col in cursor.fetchall()}
            
            columns_to_add = {
                "domain": "TEXT DEFAULT 'Hotel Reviews'",
                "model_name": "TEXT DEFAULT 'Ott Hotel Model'",
                "raw_prediction": "TEXT DEFAULT ''",
                "prob_baseline": "REAL DEFAULT 0.0",
                "prob_target": "REAL DEFAULT 0.0",
                "p_deceptive": "REAL DEFAULT 0.0",
                "p_truthful": "REAL DEFAULT 0.0"
            }
            
            for col_name, col_def in columns_to_add.items():
                if col_name not in existing_cols:
                    cursor.execute(f"ALTER TABLE review_predictions ADD COLUMN {col_name} {col_def};")
    finally:
        conn.close()

def save_single_prediction(review_text, domain, model_name, predicted_label, raw_prediction,
                           confidence, prob_baseline, prob_target, reliability_score, source="single", db_path=None):
    """
    Saves a single analysis record using parameterized SQL.
    """
    init_db(db_path)
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute("""
                INSERT INTO review_predictions (
                    review_text, domain, model_name, predicted_label, raw_prediction,
                    confidence, prob_baseline, prob_target, p_deceptive, p_truthful, reliability_score, source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(review_text),
                str(domain),
                str(model_name),
                str(predicted_label),
                str(raw_prediction),
                float(confidence),
                float(prob_baseline),
                float(prob_target),
                float(prob_target),
                float(prob_baseline),
                float(reliability_score),
                str(source),
                now
            ))
        return True
    except sqlite3.Error as exc:
        print(f"Database save error: {exc}")
        return False
    finally:
        conn.close()

def save_bulk_predictions(df, domain, model_name, source="bulk CSV", db_path=None):
    """
    Saves a batch of predictions efficiently using executemany.
    """
    init_db(db_path)
    now = datetime.now().isoformat(timespec="seconds")
    rows = []
    for _, r in df.iterrows():
        p_base = float(r.get("prob_baseline", r.get("p_truthful", 0.0)))
        p_targ = float(r.get("prob_target", r.get("p_deceptive", 0.0)))
        plabel = str(r.get("review_classification", r.get("predicted_label", "")))
        raw_pred = str(r.get("original_model_prediction", r.get("raw_prediction", "")))
        rel_score = float(r.get("individual_reliability_score", r.get("reliability_score", 0.0)))
        rows.append((
            str(r.get("review_text", "")),
            str(domain),
            str(model_name),
            plabel,
            raw_pred,
            float(r.get("confidence", 0.0)),
            p_base,
            p_targ,
            p_targ,
            p_base,
            rel_score,
            str(source),
            now
        ))
    
    conn = get_connection(db_path)
    try:
        with conn:
            conn.executemany("""
                INSERT INTO review_predictions (
                    review_text, domain, model_name, predicted_label, raw_prediction,
                    confidence, prob_baseline, prob_target, p_deceptive, p_truthful, reliability_score, source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
        return len(rows)
    except sqlite3.Error as exc:
        print(f"Bulk database save error: {exc}")
        return 0
    finally:
        conn.close()

def get_history(domain=None, search_query=None, limit=500, db_path=None):
    """
    Retrieves historical predictions with optional domain filtering and text search.
    Computes standardized Suspicion Score and Review Classification across all records.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        query = "SELECT id, domain, predicted_label, confidence, reliability_score, prob_baseline, prob_target, raw_prediction, source, review_text, created_at FROM review_predictions"
        params = []
        conditions = []
        
        if domain and domain != "All Domains":
            conditions.append("domain = ?")
            params.append(domain)
            
        if search_query and search_query.strip():
            conditions.append("review_text LIKE ?")
            params.append(f"%{search_query.strip()}%")
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        query += f" ORDER BY id DESC LIMIT {int(limit)}"
        
        df = pd.read_sql_query(query, conn, params=params)
        if not df.empty:
            susp_scores = []
            rel_scores = []
            classifications = []
            for _, r in df.iterrows():
                p_t = float(r.get("prob_target") or 0.0)
                s = round(p_t * 100.0, 1)
                susp_scores.append(s)
                rel_scores.append(round(100.0 - s, 1))
                if s < 35.0:
                    classifications.append("Genuine")
                elif s <= 65.0:
                    classifications.append("Suspicious")
                else:
                    classifications.append("Potentially Fake")
            df["suspicion_score"] = susp_scores
            df["review_classification"] = classifications
            # Ensure predicted_label shows the clean 3-tier category
            df["classification"] = classifications
        return df
    except Exception as exc:
        print(f"Error reading history: {exc}")
        return pd.DataFrame()
    finally:
        conn.close()

def get_analytics_summary(db_path=None):
    """
    Returns structured summary statistics from the database, including
    system-wide common aggregates (Genuine, Suspicious, Potentially Fake,
    Suspicious %, Fake %, Overall Review Reliability Score) and domain-specific
    aggregations across Hotel, Product, and Restaurant analyses.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM review_predictions;")
        total_count = cursor.fetchone()[0]
        
        if total_count == 0:
            return {
                "total_count": 0,
                "domain_counts": {},
                "label_counts": {},
                "avg_confidence": 0.0,
                "avg_reliability": 0.0,
                "aggregate_metrics": {
                    "total_analyzed": 0,
                    "genuine_count": 0,
                    "suspicious_count": 0,
                    "fake_count": 0,
                    "suspicious_percentage": 0.0,
                    "fake_percentage": 0.0,
                    "overall_reliability_score": 100.0
                },
                "hotel_metrics": {
                    "total_hotel_reviews": 0,
                    "total": 0,
                    "genuine_count": 0,
                    "suspicious_count": 0,
                    "fake_count": 0,
                    "likely_genuine_count": 0,
                    "suspicious_percentage": 0.0,
                    "fake_percentage": 0.0,
                    "overall_reliability_score": 100.0,
                    "risk_bands": {
                        "Likely Genuine (<35%)": 0,
                        "Suspicious (35%-65%)": 0,
                        "Potentially Fake (>65%)": 0
                    },
                    "binary_predictions": {
                        "Truthful": 0,
                        "Deceptive": 0
                    },
                    "avg_suspicion_score": 0.0
                },
                "product_metrics": {
                    "total": 0,
                    "genuine_count": 0,
                    "suspicious_count": 0,
                    "fake_count": 0,
                    "suspicious_percentage": 0.0,
                    "fake_percentage": 0.0,
                    "overall_reliability_score": 100.0,
                    "categories": {},
                    "binary_predictions": {"Human-Written (OR)": 0, "GPT-2 Generated (CG)": 0}
                },
                "restaurant_metrics": {
                    "total": 0,
                    "genuine_count": 0,
                    "suspicious_count": 0,
                    "fake_count": 0,
                    "suspicious_percentage": 0.0,
                    "fake_percentage": 0.0,
                    "overall_reliability_score": 100.0,
                    "categories": {},
                    "binary_predictions": {"Recommended (N)": 0, "Filtered (Y)": 0}
                }
            }
            
        cursor.execute("SELECT domain, COUNT(*) FROM review_predictions GROUP BY domain;")
        domain_counts = dict(cursor.fetchall())
        
        cursor.execute("SELECT predicted_label, COUNT(*) FROM review_predictions GROUP BY predicted_label;")
        label_counts = dict(cursor.fetchall())
        
        cursor.execute("SELECT AVG(confidence), AVG(reliability_score) FROM review_predictions;")
        avg_conf, avg_rel = cursor.fetchone()

        # Query all records to calculate standardized 3-tier metrics and individual reliability averages
        cursor.execute("""
            SELECT id, domain, predicted_label, raw_prediction, prob_target, p_deceptive, prob_baseline, p_truthful, reliability_score, confidence 
            FROM review_predictions;
        """)
        all_rows = cursor.fetchall()
        tot = len(all_rows)

        genuine_count = 0
        suspicious_count = 0
        fake_count = 0
        all_rel_scores = []
        all_susp_scores = []

        domain_data = {
            "hotel": {
                "total": 0, "genuine": 0, "suspicious": 0, "fake": 0,
                "rel_scores": [], "susp_scores": [],
                "binary": {"Truthful": 0, "Deceptive": 0}
            },
            "product": {
                "total": 0, "genuine": 0, "suspicious": 0, "fake": 0,
                "rel_scores": [], "susp_scores": [],
                "binary": {"Human-Written (OR)": 0, "GPT-2 Generated (CG)": 0}
            },
            "restaurant": {
                "total": 0, "genuine": 0, "suspicious": 0, "fake": 0,
                "rel_scores": [], "susp_scores": [],
                "binary": {"Recommended (N)": 0, "Filtered (Y)": 0}
            }
        }

        for r in all_rows:
            rid, dom, plbl, raw_p, p_targ, p_dec, p_base, p_tru, rel_db, conf = r
            d_lower = str(dom or "").lower()
            if "product" in d_lower or "amazon" in d_lower:
                d_key = "product"
            elif "restaurant" in d_lower or "yelp" in d_lower:
                d_key = "restaurant"
            else:
                d_key = "hotel"

            # Compute calibrated target probability
            if d_key == "hotel":
                target_p = float(p_dec if (p_dec is not None and p_dec > 0) else (p_targ or 0.0))
            else:
                target_p = float(p_targ or 0.0)

            susp_score = round(target_p * 100.0, 1)
            indiv_rel = round(100.0 - susp_score, 1)

            all_rel_scores.append(indiv_rel)
            all_susp_scores.append(susp_score)

            if susp_score < 35.0:
                genuine_count += 1
                cat = "Genuine"
            elif susp_score <= 65.0:
                suspicious_count += 1
                cat = "Suspicious"
            else:
                fake_count += 1
                cat = "Potentially Fake"

            domain_data[d_key]["total"] += 1
            if cat == "Genuine":
                domain_data[d_key]["genuine"] += 1
            elif cat == "Suspicious":
                domain_data[d_key]["suspicious"] += 1
            else:
                domain_data[d_key]["fake"] += 1

            domain_data[d_key]["rel_scores"].append(indiv_rel)
            domain_data[d_key]["susp_scores"].append(susp_score)

            # Original binary mapping
            if d_key == "hotel":
                if target_p >= 0.50 or (raw_p and "deceptive" in str(raw_p).lower()):
                    domain_data[d_key]["binary"]["Deceptive"] += 1
                else:
                    domain_data[d_key]["binary"]["Truthful"] += 1
            elif d_key == "product":
                if target_p >= 0.50 or (raw_p and "cg" in str(raw_p).lower()):
                    domain_data[d_key]["binary"]["GPT-2 Generated (CG)"] += 1
                else:
                    domain_data[d_key]["binary"]["Human-Written (OR)"] += 1
            elif d_key == "restaurant":
                if target_p >= 0.50 or (raw_p and "y" in str(raw_p).lower()):
                    domain_data[d_key]["binary"]["Filtered (Y)"] += 1
                else:
                    domain_data[d_key]["binary"]["Recommended (N)"] += 1

        susp_pct = round((suspicious_count / tot) * 100.0, 1) if tot > 0 else 0.0
        fake_pct = round((fake_count / tot) * 100.0, 1) if tot > 0 else 0.0
        overall_rel = round(float(sum(all_rel_scores) / tot), 1) if tot > 0 else 100.0

        aggregate_metrics = {
            "total_analyzed": tot,
            "genuine_count": genuine_count,
            "suspicious_count": suspicious_count,
            "fake_count": fake_count,
            "suspicious_percentage": susp_pct,
            "fake_percentage": fake_pct,
            "overall_reliability_score": overall_rel
        }

        # Build hotel_metrics
        h_d = domain_data["hotel"]
        h_tot = h_d["total"]
        h_susp_pct = round((h_d["suspicious"] / h_tot) * 100.0, 1) if h_tot > 0 else 0.0
        h_fake_pct = round((h_d["fake"] / h_tot) * 100.0, 1) if h_tot > 0 else 0.0
        h_rel = round(float(sum(h_d["rel_scores"]) / h_tot), 1) if h_tot > 0 else 100.0
        hotel_metrics = {
            "total_hotel_reviews": h_tot,
            "total": h_tot,
            "genuine_count": h_d["genuine"],
            "suspicious_count": h_d["suspicious"],
            "fake_count": h_d["fake"],
            "likely_genuine_count": h_d["genuine"],
            "suspicious_percentage": h_susp_pct,
            "fake_percentage": h_fake_pct,
            "overall_reliability_score": h_rel,
            "risk_bands": {
                "Likely Genuine (<35%)": h_d["genuine"],
                "Suspicious (35%-65%)": h_d["suspicious"],
                "Potentially Fake (>65%)": h_d["fake"]
            },
            "binary_predictions": h_d["binary"],
            "avg_suspicion_score": round(float(sum(h_d["susp_scores"]) / h_tot), 1) if h_tot > 0 else 0.0
        }

        # Build product_metrics
        p_d = domain_data["product"]
        p_tot = p_d["total"]
        p_susp_pct = round((p_d["suspicious"] / p_tot) * 100.0, 1) if p_tot > 0 else 0.0
        p_fake_pct = round((p_d["fake"] / p_tot) * 100.0, 1) if p_tot > 0 else 0.0
        p_rel = round(float(sum(p_d["rel_scores"]) / p_tot), 1) if p_tot > 0 else 100.0
        product_metrics = {
            "total": p_tot,
            "genuine_count": p_d["genuine"],
            "suspicious_count": p_d["suspicious"],
            "fake_count": p_d["fake"],
            "suspicious_percentage": p_susp_pct,
            "fake_percentage": p_fake_pct,
            "overall_reliability_score": p_rel,
            "categories": {"Genuine": p_d["genuine"], "Suspicious": p_d["suspicious"], "Potentially Fake": p_d["fake"]},
            "binary_predictions": p_d["binary"]
        }

        # Build restaurant_metrics
        r_d = domain_data["restaurant"]
        r_tot = r_d["total"]
        r_susp_pct = round((r_d["suspicious"] / r_tot) * 100.0, 1) if r_tot > 0 else 0.0
        r_fake_pct = round((r_d["fake"] / r_tot) * 100.0, 1) if r_tot > 0 else 0.0
        r_rel = round(float(sum(r_d["rel_scores"]) / r_tot), 1) if r_tot > 0 else 100.0
        restaurant_metrics = {
            "total": r_tot,
            "genuine_count": r_d["genuine"],
            "suspicious_count": r_d["suspicious"],
            "fake_count": r_d["fake"],
            "suspicious_percentage": r_susp_pct,
            "fake_percentage": r_fake_pct,
            "overall_reliability_score": r_rel,
            "categories": {"Genuine": r_d["genuine"], "Suspicious": r_d["suspicious"], "Potentially Fake": r_d["fake"]},
            "binary_predictions": r_d["binary"]
        }

        return {
            "total_count": total_count,
            "domain_counts": domain_counts,
            "label_counts": label_counts,
            "avg_confidence": round(float(avg_conf or 0.0), 4),
            "avg_reliability": round(float(avg_rel or 0.0), 1),
            "aggregate_metrics": aggregate_metrics,
            "hotel_metrics": hotel_metrics,
            "product_metrics": product_metrics,
            "restaurant_metrics": restaurant_metrics
        }
    finally:
        conn.close()

def clear_history(domain=None, db_path=None):
    """
    Clears records from database with optional domain scoping.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        with conn:
            if domain and domain != "All Domains":
                conn.execute("DELETE FROM review_predictions WHERE domain = ?;", (domain,))
            else:
                conn.execute("DELETE FROM review_predictions;")
        return True
    except sqlite3.Error as exc:
        print(f"Error clearing database: {exc}")
        return False
    finally:
        conn.close()
