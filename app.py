# app.py
"""
Fake Review Detection and Review Reliability Analysis System (ReviewTrust)
Students: Himanshi (SID: 24106002), Samarth (SID: 24106046) | Branch: CSE (Data Science)
Multi-Domain Streamlit Application supporting Hotel, Product, and Restaurant models.
"""

from datetime import datetime
from pathlib import Path
import html
import io
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import db_utils
import model_utils
from model_utils import (
    DOMAIN_CONFIGS,
    load_domain_model,
    predict_domain_reviews,
    explain_domain_review
)

# Robust category import with direct fallback
try:
    from model_utils import COMMON_CATEGORIES
except ImportError:
    COMMON_CATEGORIES = getattr(model_utils, "COMMON_CATEGORIES", ["Genuine", "Suspicious", "Potentially Fake"])

BASE_DIR = Path(__file__).resolve().parent

# =====================================================================
# Streamlit Application Configuration
# =====================================================================
st.set_page_config(
    page_title="ReviewTrust | Multi-Domain Review Analysis System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
db_utils.init_db()

# =====================================================================
# Cached Model Loaders
# =====================================================================
@st.cache_resource(show_spinner=False)
def get_model(domain_key):
    """
    Loads and caches the domain-specific model pipeline.
    """
    return load_domain_model(domain_key)

# =====================================================================
# Reusable Domain UI Components
# =====================================================================
def render_single_review_ui(selected_domain, key_prefix="single"):
    cfg = DOMAIN_CONFIGS[selected_domain]
    st.markdown(f"### {cfg['icon']} {cfg['name']} — Single Review Analysis")
    st.write(f"Task: **{cfg['task_description']}** &bull; Dataset: *{cfg['dataset_name']}* &bull; Test Accuracy: **{cfg['test_accuracy']}**")
    
    try:
        pipeline, note = get_model(selected_domain)
    except Exception as exc:
        st.error(f"Cannot load {cfg['name']} model: {exc}")
        return
        
    st.markdown("#### Sample Reviews for Quick Testing")
    sample_cols = st.columns(len(cfg["sample_reviews"]))
    for i, s_dict in enumerate(cfg["sample_reviews"]):
        with sample_cols[i]:
            if st.button(f"Load: {s_dict['label']}", key=f"btn_sample_{key_prefix}_{selected_domain}_{i}", use_container_width=True):
                st.session_state[f"review_text_{key_prefix}_{selected_domain}"] = s_dict["text"]
                
    review_text_val = st.session_state.get(f"review_text_{key_prefix}_{selected_domain}", "")
    
    user_review = st.text_area(
        "Enter or paste the review text below:",
        value=review_text_val,
        height=140,
        placeholder=f"Paste a {cfg['name'].lower()[:-1]} review here to analyze...",
        key=f"txt_area_{key_prefix}_{selected_domain}"
    )
    
    col_action1, col_action2 = st.columns([1, 4])
    with col_action1:
        analyze_btn = st.button("🚀 Analyse Review", type="primary", use_container_width=True, key=f"btn_run_{key_prefix}_{selected_domain}")
    with col_action2:
        if st.button("🧹 Clear Input", use_container_width=False, key=f"btn_clear_{key_prefix}_{selected_domain}"):
            st.session_state[f"review_text_{key_prefix}_{selected_domain}"] = ""
            st.rerun()
            
    if analyze_btn:
        cleaned = user_review.strip()
        if len(cleaned) < 10:
            st.warning("⚠️ Review text is too short. Please enter at least 10 characters for a reliable prediction.")
        else:
            with st.spinner(f"Analyzing review using {cfg['name']} pipeline..."):
                input_df = pd.DataFrame({"review_text": [cleaned]})
                res_df = predict_domain_reviews(pipeline, input_df, selected_domain)
                
                # Self-healing column guarantee
                if "review_classification" not in res_df.columns:
                    res_df["review_classification"] = res_df.get("predicted_label", ["Genuine"] * len(res_df))
                if "suspicion_score" not in res_df.columns:
                    res_df["suspicion_score"] = np.round(res_df.get("prob_target", np.zeros(len(res_df))) * 100.0, 1)
                if "individual_reliability_score" not in res_df.columns:
                    res_df["individual_reliability_score"] = np.round(100.0 - res_df["suspicion_score"], 1)
                if "original_model_prediction" not in res_df.columns:
                    res_df["original_model_prediction"] = res_df.get("binary_prediction", res_df.get("raw_prediction", res_df["review_classification"]))
                
                row = res_df.iloc[0]
                explanation = explain_domain_review(pipeline, cleaned, selected_domain, top_k=6)
                
                # Save to database
                db_utils.save_single_prediction(
                    review_text=cleaned,
                    domain=cfg["name"],
                    model_name=cfg["task_description"],
                    predicted_label=row["review_classification"],
                    raw_prediction=row["original_model_prediction"],
                    confidence=row["confidence"],
                    prob_baseline=row["prob_baseline"],
                    prob_target=row["prob_target"],
                    reliability_score=row["individual_reliability_score"],
                    source="single"
                )
                
            st.divider()
            st.markdown("### 📊 Prediction & Review Reliability Breakdown")
            
            rev_class = row["review_classification"]
            susp_score = row["suspicion_score"]
            indiv_rel = row["individual_reliability_score"]
            orig_pred = row["original_model_prediction"]
            conf = row["confidence"]
            p_base = row["prob_baseline"]
            p_targ = row["prob_target"]
            
            # 1. Primary Classification Banner
            if rev_class == "Genuine":
                st.success(f"### Review Classification: **GENUINE** &nbsp;|&nbsp; Reliability Score: **{indiv_rel:.1f}%** &nbsp;|&nbsp; Suspicion Score: **{susp_score:.1f}%**")
            elif rev_class == "Suspicious":
                st.warning(f"### Review Classification: **SUSPICIOUS** &nbsp;|&nbsp; Suspicion Score: **{susp_score:.1f}%** &nbsp;|&nbsp; Reliability Score: **{indiv_rel:.1f}%**")
            else:
                st.error(f"### Review Classification: **POTENTIALLY FAKE** &nbsp;|&nbsp; Suspicion Score: **{susp_score:.1f}%** &nbsp;|&nbsp; Reliability Score: **{indiv_rel:.1f}%**")
                
            # 2. Prominent Score Cards
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            sc1.metric("Review Classification", rev_class)
            sc2.metric("Suspicion Score", f"{susp_score:.1f}%")
            sc3.metric("Review Reliability Score", f"{indiv_rel:.1f}%")
            sc4.metric("Original Model Prediction", str(orig_pred)[:22])
            sc5.metric("Model Confidence", f"{conf*100:.1f}%")
            
            # 3. Visual Progress Bars
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                st.markdown(f"**🛡️ Individual Review Reliability Score: `{indiv_rel:.1f}%`**")
                st.progress(max(0.0, min(1.0, indiv_rel / 100.0)))
                st.caption(f"Formula: $100 - \\text{{Suspicion Score}} = 100 - {susp_score:.1f}\\% = \\mathbf{{{indiv_rel:.1f}\\%}}$")
            with b_col2:
                st.markdown(f"**⚠️ Suspicion Score: `{susp_score:.1f}%`**")
                st.progress(max(0.0, min(1.0, susp_score / 100.0)))
                st.caption(f"Formula: $P(\\text{{{cfg['class_1_name']}}}) \\times 100 = \\mathbf{{{susp_score:.1f}\\%}}$")
                
            # 4. Class Probabilities & Alignment Bar
            st.markdown(f"**Posterior Class Probabilities ({cfg['name']}):**")
            p_cols = st.columns([int(p_base*100) or 1, int(p_targ*100) or 1])
            p_cols[0].info(f"{cfg['class_0_name']}: {p_base*100:.1f}%")
            p_cols[1].warning(f"{cfg['class_1_name']}: {p_targ*100:.1f}%")
            
            # 5. Theoretical Formulation & Limitations Box
            st.info(
                f"ℹ️ **Evaluation Formulation & Scientific Mapping:**  \n"
                f"• **Review Classification ({rev_class}):** Derived from model risk thresholds justified on benchmark validation data "
                f"(<35%: Genuine, 35%–65%: Suspicious, >65%: Potentially Fake).  \n"
                f"• **Suspicion Score:** Probability of the target risk construct ({cfg['class_1_name']}) $\\times 100 = \\mathbf{{{susp_score:.1f}\\%}}$.  \n"
                f"• **Individual Review Reliability Score:** $100 - \\text{{Suspicion Score}} = \\mathbf{{{indiv_rel:.1f}\\%}}$.  \n"
                f"• **Original Model Prediction:** `{orig_pred}` (Direct output at standard $0.50$ decision boundary).  \n"
                f"• **Domain Construct:** {cfg['task_description']} ({cfg['citation']})."
            )
            
            st.caption(
                "⚠️ **Model-Based Estimation Notice:** These individual scores are model-based statistical estimates reflecting lexical n-gram patterns learned from benchmark data. "
                "They indicate linguistic resemblance and do **not** constitute guaranteed forensic or legal proof that a review is genuine or fake."
            )
            
            # 6. Feature Attribution & Highlighter
            st.divider()
            st.markdown("### 🔬 Mathematical Feature Explanations")
            st.write(
                "N-gram tokens driving the prediction, calculated via $\\text{Contribution} = \\text{TF-IDF} \\times \\text{Logistic Regression Weight}$:"
            )
            
            exp_c1, exp_c2 = st.columns(2)
            with exp_c1:
                st.markdown(f"##### 🟢 Tokens Driving Towards **{cfg['class_0_name']}**")
                b_feats = explanation["baseline_features"]
                if b_feats:
                    b_table = pd.DataFrame(b_feats)[["feature", "contribution", "tfidf", "weight"]]
                    b_table.columns = ["Token / Phrase", "Impact Score", "TF-IDF Value", "Model Weight"]
                    st.dataframe(b_table, use_container_width=True, hide_index=True)
                else:
                    st.info(f"No strong {cfg['class_0_name']} tokens detected.")
                    
            with exp_c2:
                st.markdown(f"##### 🔴 Tokens Driving Towards **{cfg['class_1_name']}**")
                t_feats = explanation["target_features"]
                if t_feats:
                    t_table = pd.DataFrame(t_feats)[["feature", "contribution", "tfidf", "weight"]]
                    t_table.columns = ["Token / Phrase", "Impact Score", "TF-IDF Value", "Model Weight"]
                    st.dataframe(t_table, use_container_width=True, hide_index=True)
                else:
                    st.info(f"No strong {cfg['class_1_name']} tokens detected.")
                    
            # Visual Text Highlighter
            st.markdown("##### 📝 Text with Influential Terms Highlighted")
            top_b_words = explanation["top_baseline_terms"]
            top_t_words = explanation["top_target_terms"]
            
            annotated = html.escape(cleaned)
            for term in top_b_words:
                pat = re.compile(rf"\b({re.escape(term)})\b", re.IGNORECASE)
                annotated = pat.sub(r'<span style="background-color: #d4edda; color: #155724; padding: 2px 4px; border-radius: 4px; font-weight: bold;">\1</span>', annotated)
            for term in top_t_words:
                pat = re.compile(rf"\b({re.escape(term)})\b", re.IGNORECASE)
                annotated = pat.sub(r'<span style="background-color: #f8d7da; color: #721c24; padding: 2px 4px; border-radius: 4px; font-weight: bold;">\1</span>', annotated)
                
            st.markdown(
                f'<div style="background-color: #fafafa; padding: 14px; border: 1px solid #ddd; border-radius: 6px; line-height: 1.8;">{annotated}</div>',
                unsafe_allow_html=True
            )
            st.caption(f"🟢 Green: Associates with {cfg['class_0_name']} | 🔴 Red: Associates with {cfg['class_1_name']}")

def render_bulk_csv_ui(bulk_domain, key_prefix="bulk"):
    b_cfg = DOMAIN_CONFIGS[bulk_domain]
    st.markdown(f"### {b_cfg['icon']} {b_cfg['name']} — Bulk CSV Analysis")
    st.write(f"Upload a CSV file containing reviews to score with the **{b_cfg['name']}** expert ({b_cfg['task_description']}).")
    
    try:
        bulk_pipeline, _ = get_model(bulk_domain)
    except Exception as exc:
        st.error(f"Cannot load bulk model for {b_cfg['name']}: {exc}")
        return
        
    sample_texts = [s["text"] for s in b_cfg["sample_reviews"]]
    sample_df = pd.DataFrame({
        "review_id": [f"REV_{i+1:03d}" for i in range(len(sample_texts))],
        "review_text": sample_texts,
        "rating": [5.0, 1.0] if len(sample_texts) == 2 else [5.0] * len(sample_texts)
    })
    st.download_button(
        label=f"📥 Download Sample Input CSV ({b_cfg['name']})",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name=f"sample_{bulk_domain}_reviews.csv",
        mime="text/csv",
        key=f"btn_dl_sample_{key_prefix}_{bulk_domain}"
    )
    
    uploaded_file = st.file_uploader(f"Upload a CSV file for {b_cfg['name']}:", type=["csv"], key=f"csv_upload_{key_prefix}_{bulk_domain}")
    
    if uploaded_file is not None:
        try:
            incoming = pd.read_csv(uploaded_file)
            st.markdown(f"**Loaded CSV Preview:** ({len(incoming):,} rows, {len(incoming.columns)} columns)")
            st.dataframe(incoming.head(3), use_container_width=True)
            
            potential_text_cols = ["review_text", "review", "text", "Review", "Text", "content", "Comments", "reviewContent"]
            default_col = None
            for col in potential_text_cols:
                if col in incoming.columns:
                    default_col = col
                    break
            if default_col is None and len(incoming.columns) > 0:
                default_col = incoming.columns[0]
                
            selected_text_col = st.selectbox(
                "Select review text column:",
                options=list(incoming.columns),
                index=list(incoming.columns).index(default_col) if default_col in incoming.columns else 0,
                key=f"col_sel_{key_prefix}_{bulk_domain}"
            )
            
            if st.button("⚡ Run Bulk Analysis", type="primary", key=f"btn_bulk_{key_prefix}_{bulk_domain}"):
                prep_df = incoming.copy()
                prep_df["review_text"] = prep_df[selected_text_col].fillna("").astype(str)
                valid_mask = prep_df["review_text"].str.strip().ne("")
                
                if not valid_mask.any():
                    st.warning("All rows in the selected column are empty or whitespace.")
                else:
                    with st.spinner(f"Scoring {len(prep_df):,} reviews with {b_cfg['name']} expert..."):
                        scored_df = predict_domain_reviews(bulk_pipeline, prep_df, bulk_domain)
                        
                        # Self-healing column guarantee
                        if "review_classification" not in scored_df.columns:
                            scored_df["review_classification"] = scored_df.get("predicted_label", ["Genuine"] * len(scored_df))
                        if "suspicion_score" not in scored_df.columns:
                            scored_df["suspicion_score"] = np.round(scored_df.get("prob_target", np.zeros(len(scored_df))) * 100.0, 1)
                        if "individual_reliability_score" not in scored_df.columns:
                            scored_df["individual_reliability_score"] = np.round(100.0 - scored_df["suspicion_score"], 1)
                        if "original_model_prediction" not in scored_df.columns:
                            scored_df["original_model_prediction"] = scored_df.get("binary_prediction", scored_df.get("raw_prediction", scored_df["review_classification"]))
                        
                        db_utils.save_bulk_predictions(
                            df=scored_df[valid_mask],
                            domain=b_cfg["name"],
                            model_name=b_cfg["task_description"],
                            source="bulk CSV"
                        )
                        st.session_state[f"bulk_scored_{key_prefix}_{bulk_domain}"] = scored_df
                        
            if f"bulk_scored_{key_prefix}_{bulk_domain}" in st.session_state:
                res = st.session_state[f"bulk_scored_{key_prefix}_{bulk_domain}"]
                st.divider()
                st.markdown("### 📊 Bulk Processing Summary")
                
                n_total = len(res)
                n_gen = (res["review_classification"] == "Genuine").sum()
                n_susp = (res["review_classification"] == "Suspicious").sum()
                n_fake = (res["review_classification"] == "Potentially Fake").sum()
                avg_susp = res["suspicion_score"].mean()
                avg_rel = res["individual_reliability_score"].mean()
                avg_conf = res["confidence"].mean()
                
                b1, b2, b3, b4, b5 = st.columns(5)
                b1.metric("Total Rows", f"{n_total:,}")
                b2.metric("Genuine", f"{n_gen:,}", f"{(n_gen/n_total)*100:.1f}%")
                b3.metric("Suspicious", f"{n_susp:,}", f"{(n_susp/n_total)*100:.1f}%")
                b4.metric("Potentially Fake", f"{n_fake:,}", f"{(n_fake/n_total)*100:.1f}%")
                b5.metric("Avg Reliability Score", f"{avg_rel:.1f}%", f"Susp: {avg_susp:.1f}%")
                
                st.markdown("#### Scored Dataset Results")
                display_cols = ["review_classification", "suspicion_score", "individual_reliability_score", "original_model_prediction", "confidence", "review_text"]
                extra_cols = [c for c in res.columns if c not in display_cols and c not in ["prob_baseline", "prob_target", "model_domain", "reliability_score", "predicted_label", "risk_category", "binary_prediction", "raw_prediction"]]
                final_cols = display_cols + extra_cols
                st.dataframe(res[[c for c in final_cols if c in res.columns]], use_container_width=True)
                
                csv_bytes = res.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"💾 Download Scored Results (CSV)",
                    data=csv_bytes,
                    file_name=f"scored_{bulk_domain}_reviews.csv",
                    mime="text/csv",
                    key=f"dl_csv_{key_prefix}_{bulk_domain}"
                )
        except Exception as exc:
            st.error(f"Error parsing uploaded CSV: {exc}")

# =====================================================================
# Sidebar Navigation & Domain Information
# =====================================================================
with st.sidebar:
    st.markdown("## 🛡️ **ReviewTrust**")
    st.caption("**Punjab Engineering College** &bull; B.Tech CSE (Data Science)")
    st.markdown("**Team Members:**  \n- **Himanshi** (SID: 24106002)  \n- **Samarth** (SID: 24106046)")
    
    st.divider()
    st.markdown("### 📌 Navigation")
    
    page_options = [
        "🏠 Home & Project Overview",
        "🏨 Hotel Reviews",
        "🍽️ Restaurant Reviews",
        "📦 Product Reviews",
        "🔍 Single Review Analysis",
        "📁 Bulk CSV Analysis",
        "📊 Review Analytics Dashboard",
        "🗄️ Database & History",
        "📚 Multi-Domain Benchmark Report"
    ]
    query_page = st.query_params.get("page", None)
    default_page_idx = 0
    if query_page:
        q_clean = str(query_page).lower()
        for idx, opt in enumerate(page_options):
            if any(term in opt.lower() for term in [q_clean, q_clean.replace("_", " ")]):
                default_page_idx = idx
                break

    nav_selection = st.radio(
        "Go to page:",
        page_options,
        index=default_page_idx,
        label_visibility="collapsed"
    )
    
    st.divider()
    st.markdown("### ⚙️ Active Domain Expert")
    query_domain = st.query_params.get("domain", None)
    domain_keys = list(DOMAIN_CONFIGS.keys())
    default_domain_idx = 0
    if query_domain and str(query_domain).lower() in domain_keys:
        default_domain_idx = domain_keys.index(str(query_domain).lower())

    active_domain_key = st.selectbox(
        "Select Domain Expert:",
        options=domain_keys,
        format_func=lambda k: f"{DOMAIN_CONFIGS[k]['icon']} {DOMAIN_CONFIGS[k]['name']}",
        index=default_domain_idx
    )
    
    active_cfg = DOMAIN_CONFIGS[active_domain_key]
    
    # Attempt to load model for active domain
    model_loaded = False
    pipeline = None
    try:
        pipeline, load_note = get_model(active_domain_key)
        model_loaded = True
        st.success(f"✅ {active_cfg['name']} Model Active")
    except Exception as exc:
        st.error(f"❌ Model Unavailable: {exc}")
        
    with st.expander("ℹ️ Domain Specifications", expanded=False):
        st.markdown(f"**Dataset:** {active_cfg['dataset_name']}")
        st.markdown(f"**Citation:** *{active_cfg['citation']}*")
        st.markdown(f"**Task:** {active_cfg['task_description']}")
        st.markdown(f"**Test Accuracy:** **{active_cfg['test_accuracy']}** ({active_cfg['test_samples']} test samples)")
        st.markdown(f"**Class 0:** {active_cfg['class_0_name']}")
        st.markdown(f"**Class 1:** {active_cfg['class_1_name']}")
        st.markdown(f"**3-Tier Categories:** Genuine (<35%), Suspicious (35%–65%), Potentially Fake (>65%)")
        st.markdown(f"**Scope Boundary:**")
        st.caption(active_cfg["limitations"])
        
    st.divider()
    st.caption("ReviewTrust Academic System v2.0 &bull; Python & Scikit-learn")

# =====================================================================
# PAGE 1: HOME & PROJECT OVERVIEW
# =====================================================================
if nav_selection == "🏠 Home & Project Overview":
    st.title("🛡️ Fake Review Detection & Review Reliability Analysis System")
    st.subheader("An Academic Machine Learning System for Multi-Domain Review Veracity Analysis")
    
    st.markdown("""
    **Project Team:**  
    1. **Himanshi** — SID: 24106002  
    2. **Samarth** — SID: 24106046  
    **Branch:** B.Tech CSE (Data Science) &nbsp;|&nbsp; **Academic Year:** 2026  
    **Institution:** Punjab Engineering College  
    """)
    
    st.divider()
    
    col_intro1, col_intro2 = st.columns([3, 2])
    with col_intro1:
        st.markdown("### 🎯 Problem Statement")
        st.write(
            "Online reviews across hospitality, dining, and e-commerce heavily influence consumer decisions. "
            "However, platforms face diverse forms of deceptive opinion spam—ranging from imaginative human crowdsourced fabrications "
            "and platform filtering decisions to machine-generated synthetic text (e.g. GPT-2). "
            "Commercial anti-spam systems often treat fake reviews as a monolithic entity, ignoring fundamental differences "
            "in underlying ground-truth constructs. This project establishes an authentic, multi-domain machine learning "
            "system that evaluates reviews using dedicated task experts, transparent uncertainty bands, and mathematically explainable attribution."
        )
        
        st.markdown("### 🏆 Core Objectives")
        st.markdown("""
        1. **Multi-Domain Specialization:** Deploy dedicated, leak-free models for Hotel (Ott Corpus), Product (Amazon 40K), and Restaurant (YelpChi) domains.
        2. **Unified 3-Tier Categorization:** Classify every review as **Genuine**, **Suspicious**, or **Potentially Fake** using validation-justified thresholds.
        3. **Dual Review Metrics:** Compute an individual **Suspicion Score** (0–100%) and **Individual Review Reliability Score** (100 − Suspicion Score).
        4. **Explainable AI (XAI):** Compute direct feature contributions using TF-IDF weights and Logistic Regression coefficients.
        5. **Dual Operational Modes:** Deliver interactive single-review inference alongside bulk CSV ingestion and export.
        6. **Auditable Persistence:** Log analysis records using a local, parameterized SQLite relational database (`reviews.db`).
        """)
        
    with col_intro2:
        st.markdown("### 💻 Technology Stack")
        st.markdown("""
        | Layer | Technology |
        | :--- | :--- |
        | **Language** | Python 3.10+ |
        | **Machine Learning** | Scikit-learn (TF-IDF + Logistic Regression) |
        | **Frontend UI** | Streamlit Web Framework |
        | **Data Processing** | Pandas, NumPy |
        | **Database** | SQLite3 (Parameterized Relational Storage) |
        | **Visualizations** | Matplotlib, Streamlit Native Charts |
        | **Model Serialization** | Joblib |
        """)
        
        st.markdown("### 🔬 Academic Integrity Rule")
        st.info(
            "**Construct Integrity:** Each model evaluates a distinct task. "
            "Hotel models classify Truthful vs Deceptive reviews (Ott et al.); "
            "Amazon models distinguish human-written OR from GPT-2-generated CG text (Salminen et al.); "
            "Yelp models classify platform Recommended (N) vs Filtered (Y) decisions (Rayana & Akoglu). "
            "Original model predictions are always preserved separately."
        )
        
    st.divider()
    st.markdown("### 🌐 Supported Domains & Independent Task Experts")
    
    c_d1, c_d2, c_d3 = st.columns(3)
    with c_d1:
        st.markdown(f"#### 🏨 {DOMAIN_CONFIGS['hotel']['name']}")
        st.write(f"**Task:** {DOMAIN_CONFIGS['hotel']['task_description']}")
        st.write(f"**Corpus:** {DOMAIN_CONFIGS['hotel']['dataset_name']}")
        st.metric("Test Accuracy", DOMAIN_CONFIGS['hotel']['test_accuracy'], "N=320 test reviews")
        st.caption(f"**Original Labels:** {DOMAIN_CONFIGS['hotel']['class_0_name']} vs {DOMAIN_CONFIGS['hotel']['class_1_name']}")
        st.caption(DOMAIN_CONFIGS['hotel']['limitations'])
        
    with c_d2:
        st.markdown(f"#### 📦 {DOMAIN_CONFIGS['product']['name']}")
        st.write(f"**Task:** {DOMAIN_CONFIGS['product']['task_description']}")
        st.write(f"**Corpus:** {DOMAIN_CONFIGS['product']['dataset_name']}")
        st.metric("Test Accuracy", DOMAIN_CONFIGS['product']['test_accuracy'], "N=8,087 test reviews")
        st.caption(f"**Original Labels:** {DOMAIN_CONFIGS['product']['class_0_name']} vs {DOMAIN_CONFIGS['product']['class_1_name']}")
        st.caption(DOMAIN_CONFIGS['product']['limitations'])
        
    with c_d3:
        st.markdown(f"#### 🍽️ {DOMAIN_CONFIGS['restaurant']['name']}")
        st.write(f"**Task:** {DOMAIN_CONFIGS['restaurant']['task_description']}")
        st.write(f"**Corpus:** {DOMAIN_CONFIGS['restaurant']['dataset_name']}")
        st.metric("Test Accuracy", "77.79%", "Balanced Acc: 53.88% (Filtered Recall: 9.60%)")
        st.caption(f"**Original Labels:** {DOMAIN_CONFIGS['restaurant']['class_0_name']} vs {DOMAIN_CONFIGS['restaurant']['class_1_name']}")
        st.caption(DOMAIN_CONFIGS['restaurant']['limitations'])

# =====================================================================
# DOMAIN PAGES: HOTEL, RESTAURANT, PRODUCT
# =====================================================================
elif nav_selection == "🏨 Hotel Reviews":
    st.title("🏨 Hotel Reviews Analysis & Verification")
    st.write("Specialized Deceptive Opinion Spam analysis for hospitality reviews based on the Ott et al. benchmark.")
    t1, t2 = st.tabs(["🔍 Single Review Analysis", "📁 Bulk CSV Analysis"])
    with t1:
        render_single_review_ui("hotel", "hotel_page")
    with t2:
        render_bulk_csv_ui("hotel", "hotel_page")

elif nav_selection == "🍽️ Restaurant Reviews":
    st.title("🍽️ Restaurant Reviews Analysis & Verification")
    st.write("Specialized platform recommendation filter analysis for dining reviews based on the YelpChi benchmark.")
    t1, t2 = st.tabs(["🔍 Single Review Analysis", "📁 Bulk CSV Analysis"])
    with t1:
        render_single_review_ui("restaurant", "rest_page")
    with t2:
        render_bulk_csv_ui("restaurant", "rest_page")

elif nav_selection == "📦 Product Reviews":
    st.title("📦 Product Reviews Analysis & Verification")
    st.write("Specialized human vs machine-generated review analysis for e-commerce products based on the Amazon 40K benchmark.")
    t1, t2 = st.tabs(["🔍 Single Review Analysis", "📁 Bulk CSV Analysis"])
    with t1:
        render_single_review_ui("product", "prod_page")
    with t2:
        render_bulk_csv_ui("product", "prod_page")

# =====================================================================
# UNIVERSAL TASK PAGES: SINGLE REVIEW & BULK CSV
# =====================================================================
elif nav_selection == "🔍 Single Review Analysis":
    st.title("🔍 Multi-Domain Single Review Analysis")
    st.write("Inspect individual reviews across Hotel, Restaurant, or Product domain experts.")
    d_col1, d_col2 = st.columns([2, 3])
    with d_col1:
        selected_domain = st.selectbox(
            "Select Review Domain Expert:",
            options=["hotel", "product", "restaurant"],
            format_func=lambda k: f"{DOMAIN_CONFIGS[k]['icon']} {DOMAIN_CONFIGS[k]['name']}",
            index=["hotel", "product", "restaurant"].index(active_domain_key),
            key="multi_single_domain_select"
        )
    cfg = DOMAIN_CONFIGS[selected_domain]
    with d_col2:
        st.info(f"**Expert:** {cfg['icon']} {cfg['name']} &bull; **Dataset:** {cfg['dataset_name']} &bull; **Accuracy:** {cfg['test_accuracy']}")
    render_single_review_ui(selected_domain, "multi_single")

elif nav_selection == "📁 Bulk CSV Analysis":
    st.title("📁 Multi-Domain Bulk CSV Analysis")
    st.write("Upload a CSV file containing multiple reviews to run batch inference across any of the three domain experts.")
    col_csv_top1, col_csv_top2 = st.columns([2, 3])
    with col_csv_top1:
        bulk_domain = st.selectbox(
            "Select Processing Domain Expert:",
            options=["hotel", "product", "restaurant"],
            format_func=lambda k: f"{DOMAIN_CONFIGS[k]['icon']} {DOMAIN_CONFIGS[k]['name']}",
            index=["hotel", "product", "restaurant"].index(active_domain_key),
            key="multi_bulk_domain_select"
        )
    b_cfg = DOMAIN_CONFIGS[bulk_domain]
    with col_csv_top2:
        st.info(f"Target Model: **{b_cfg['name']}** &bull; Task: *{b_cfg['task_description']}*")
    render_bulk_csv_ui(bulk_domain, "multi_bulk")

# =====================================================================
# REVIEW ANALYTICS DASHBOARD
# =====================================================================
elif nav_selection == "📊 Review Analytics Dashboard":
    st.title("📊 Review Analytics Dashboard")
    st.write("Aggregated analytics, empirical prediction distributions, and review reliability metrics across domain experts.")
    
    summary = db_utils.get_analytics_summary()
    agg = summary["aggregate_metrics"]
    tot = agg["total_analyzed"]
    
    if tot == 0:
        st.info("ℹ️ No historical reviews have been analyzed yet. Run single or bulk analyses to populate the dashboard.")
    else:
        # Primary Aggregate Metrics Row 1
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Reviews Analysed (N)", f"{tot:,}")
        kpi2.metric("Overall Review Reliability Score", f"{agg['overall_reliability_score']:.1f}%")
        kpi3.metric("Suspicious Reviews Percentage", f"{agg['suspicious_percentage']:.1f}%")
        kpi4.metric("Potentially Fake Reviews Percentage", f"{agg['fake_percentage']:.1f}%")

        # Primary Aggregate Metrics Row 2
        kpi2_1, kpi2_2, kpi2_3, kpi2_4 = st.columns(4)
        kpi2_1.metric("Genuine Reviews Count", f"{agg['genuine_count']:,}")
        kpi2_2.metric("Suspicious Reviews Count", f"{agg['suspicious_count']:,}")
        kpi2_3.metric("Potentially Fake Reviews Count", f"{agg['fake_count']:,}")
        kpi2_4.metric("Domains Evaluated", f"{len(summary['domain_counts'])}")

        st.divider()

        # Explicit calculation formula & denominator breakdown callout
        st.info(
            f"📐 **Dashboard Metric Calculation Formulas & Denominator Breakdown:**  \n"
            f"• **Sample Denominator (N):** ${tot}$ total compatible reviews analysed across all active domains.  \n"
            f"• **Overall Review Reliability Score:** $\\frac{{1}}{{N}} \\sum_{{i=1}}^N \\text{{Individual Reliability Score}}_i = "
            f"\\mathbf{{{agg['overall_reliability_score']:.1f}\\%}}$  \n"
            f"• **Suspicious Reviews Percentage:** $\\frac{{N_{{\\text{{suspicious}}}}}}{{N}} \\times 100 = "
            f"\\frac{{{agg['suspicious_count']}}}{{{tot}}} \\times 100 = \\mathbf{{{agg['suspicious_percentage']:.1f}\\%}}$  \n"
            f"• **Potentially Fake Reviews Percentage:** $\\frac{{N_{{\\text{{fake}}}}}}{{N}} \\times 100 = "
            f"\\frac{{{agg['fake_count']}}}{{{tot}}} \\times 100 = \\mathbf{{{agg['fake_percentage']:.1f}\\%}}$"
        )

        # System-Wide Visualizations
        st.markdown("#### 📈 Cross-Domain Empirical Distributions")
        ch_col1, ch_col2 = st.columns(2)
        with ch_col1:
            st.markdown("##### 🚦 3-Tier Classification Distribution")
            cat_df = pd.Series({
                "Genuine (<35%)": agg["genuine_count"],
                "Suspicious (35%-65%)": agg["suspicious_count"],
                "Potentially Fake (>65%)": agg["fake_count"]
            })
            st.bar_chart(cat_df)

        with ch_col2:
            st.markdown("##### 🌐 Analyses by Domain")
            dom_s = pd.Series(summary["domain_counts"])
            st.bar_chart(dom_s)

        st.divider()

        # Domain-Specific Breakdown Sections
        st.markdown("### 🌐 Domain-Specific Analysis & Original Predictions")
        
        tab_h, tab_p, tab_r = st.tabs(["🏨 Hotel Reviews", "📦 Product Reviews", "🍽️ Restaurant Reviews"])
        
        with tab_h:
            h_m = summary["hotel_metrics"]
            if h_m["total"] > 0:
                hc1, hc2, hc3, hc4 = st.columns(4)
                hc1.metric("Hotel Reviews (N)", f"{h_m['total']}")
                hc2.metric("Reliability Score", f"{h_m['overall_reliability_score']:.1f}%")
                hc3.metric("Suspicious %", f"{h_m['suspicious_percentage']:.1f}%")
                hc4.metric("Fake %", f"{h_m['fake_percentage']:.1f}%")
                
                h_ch1, h_ch2 = st.columns(2)
                with h_ch1:
                    st.markdown("###### 3-Tier Risk Bands")
                    st.bar_chart(pd.Series(h_m["risk_bands"]))
                with h_ch2:
                    st.markdown("###### Original Model Predictions (Truthful vs Deceptive)")
                    st.bar_chart(pd.Series(h_m["binary_predictions"]))
            else:
                st.caption("No hotel reviews evaluated yet.")

        with tab_p:
            p_m = summary["product_metrics"]
            if p_m["total"] > 0:
                pc1, pc2, pc3, pc4 = st.columns(4)
                pc1.metric("Product Reviews (N)", f"{p_m['total']}")
                pc2.metric("Reliability Score", f"{p_m['overall_reliability_score']:.1f}%")
                pc3.metric("Suspicious %", f"{p_m['suspicious_percentage']:.1f}%")
                pc4.metric("Fake %", f"{p_m['fake_percentage']:.1f}%")
                
                p_ch1, p_ch2 = st.columns(2)
                with p_ch1:
                    st.markdown("###### 3-Tier Risk Bands")
                    st.bar_chart(pd.Series(p_m["categories"]))
                with p_ch2:
                    st.markdown("###### Original Model Predictions (Human OR vs GPT-2 CG)")
                    st.bar_chart(pd.Series(p_m["binary_predictions"]))
            else:
                st.caption("No product reviews evaluated yet.")

        with tab_r:
            r_m = summary["restaurant_metrics"]
            if r_m["total"] > 0:
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Restaurant Reviews (N)", f"{r_m['total']}")
                rc2.metric("Reliability Score", f"{r_m['overall_reliability_score']:.1f}%")
                rc3.metric("Suspicious %", f"{r_m['suspicious_percentage']:.1f}%")
                rc4.metric("Fake %", f"{r_m['fake_percentage']:.1f}%")
                
                r_ch1, r_ch2 = st.columns(2)
                with r_ch1:
                    st.markdown("###### 3-Tier Risk Bands")
                    st.bar_chart(pd.Series(r_m["categories"]))
                with r_ch2:
                    st.markdown("###### Original Model Predictions (Recommended N vs Filtered Y)")
                    st.bar_chart(pd.Series(r_m["binary_predictions"]))
            else:
                st.caption("No restaurant reviews evaluated yet.")

        st.divider()
        st.markdown("##### 🛡️ Academic Interpretation & Non-Fraud Disclaimer")
        st.caption(
            "⚠️ **Methodological Boundary:** Prediction counts reflect classifier outputs across user inputs, "
            "not a verified census of platform authenticity. Specifically, synthetic text rates (`CG`) or platform filter rates (`Y`) "
            "must not be conflated with commercial fraud rates. No model output constitutes legal or forensic proof of real-world deception."
        )

# =====================================================================
# DATABASE & HISTORY
# =====================================================================
elif nav_selection == "🗄️ Database & History":
    st.title("🗄️ Relational Database & Prediction Logs")
    st.write("Inspect, search, filter, and export historical analysis records stored in SQLite (`reviews.db`).")
    
    f_col1, f_col2, f_col3 = st.columns([2, 3, 1])
    with f_col1:
        dom_filter = st.selectbox(
            "Filter by Domain:",
            options=["All Domains", "Hotel Reviews", "Product Reviews", "Restaurant Reviews"]
        )
    with f_col2:
        search_kw = st.text_input("Search Review Text:", placeholder="Type a keyword to filter...")
    with f_col3:
        record_limit = st.selectbox("Max Rows:", [50, 100, 250, 500, 1000], index=1)
        
    hist_df = db_utils.get_history(domain=dom_filter, search_query=search_kw, limit=record_limit)
    
    if hist_df.empty:
        st.info("No matching records found in database.")
    else:
        st.markdown(f"**Found {len(hist_df):,} records** (Ordered by most recent):")
        
        # Display formatted table
        cols_to_show = ["id", "domain", "classification", "suspicion_score", "reliability_score", "raw_prediction", "confidence", "review_text", "created_at"]
        disp_df = hist_df[[c for c in cols_to_show if c in hist_df.columns]].copy()
        rename_map = {
            "id": "ID",
            "domain": "Domain",
            "classification": "Classification",
            "suspicion_score": "Suspicion Score (%)",
            "reliability_score": "Review Reliability Score (%)",
            "raw_prediction": "Original Model Prediction",
            "confidence": "Confidence",
            "review_text": "Review Text",
            "created_at": "Timestamp"
        }
        disp_df.rename(columns=rename_map, inplace=True)
        st.dataframe(disp_df, use_container_width=True)
        
        act_col1, act_col2 = st.columns([2, 2])
        with act_col1:
            csv_exp = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Filtered History (CSV)",
                data=csv_exp,
                file_name="review_prediction_history.csv",
                mime="text/csv"
            )
        with act_col2:
            confirm_clear = st.checkbox("Confirm database clearance", key="chk_confirm_clear")
            if st.button("🗑️ Clear History", disabled=not confirm_clear, type="secondary"):
                if db_utils.clear_history(domain=dom_filter):
                    st.success("Database records successfully cleared.")
                    st.rerun()

# =====================================================================
# MULTI-DOMAIN BENCHMARK REPORT
# =====================================================================
elif nav_selection == "📚 Multi-Domain Benchmark Report":
    st.title("📚 Multi-Domain Empirical Benchmark Report")
    st.subheader("Academic Comparison of Three Independent Domain Experts")
    
    comp_file = BASE_DIR / "experiments" / "COMPARATIVE_EVALUATION.md"
    if comp_file.exists():
        with open(comp_file, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("Comparative evaluation report is located at `experiments/COMPARATIVE_EVALUATION.md`.")
