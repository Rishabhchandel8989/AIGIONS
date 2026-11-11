"""
Streamlit Web Dashboard for Student Performance & At-Risk Prediction.
Interactive decision-support interface for academic advisors and educational leadership.
Real-time artifact inference, SHAP explanations, batch cohort auditing, and ethical guardrails.
"""

import os
os.environ["MPLCONFIGDIR"] = os.path.abspath(".mpl_cache")
os.makedirs(".mpl_cache", exist_ok=True)
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Internal application modules
from src.explainability import ModelExplainer
from src.transformers import EducationalFeatureEngineer

# Streamlit Page Configuration
st.set_page_config(
    page_title="Algions AI - Student At-Risk Advisory System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .risk-card-low {
        background-color: #ECFDF5;
        border-left: 6px solid #10B981;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .risk-card-mod {
        background-color: #FFFBEB;
        border-left: 6px solid #F59E0B;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .risk-card-high {
        background-color: #FEF2F2;
        border-left: 6px solid #EF4444;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .ethical-banner {
        background-color: #F3F4F6;
        border: 1px solid #D1D5DB;
        padding: 0.8rem 1.2rem;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #374151;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_explainer_engine():
    return ModelExplainer(artifacts_dir="artifacts")


try:
    explainer = load_explainer_engine()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please ensure pipeline training has executed.")
    st.stop()

# Header Section
st.markdown('<div class="main-title">🎓 Student Performance & At-Risk Advisory System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Algions AI Labs | Advanced Individual ML Decision-Support Solution</div>', unsafe_allow_html=True)

st.markdown("""
<div class="ethical-banner">
    ⚠️ <strong>ETHICAL GOVERNANCE & DECISION-SUPPORT NOTICE:</strong><br>
    This machine learning system is strictly an advisory aid designed for early proactive academic support.
    It operates purely on observable academic and engagement behaviors up to Week 8.
    <strong>It must NEVER be used for automated punishment, grading deductions, academic probation, or admission denial.</strong>
    Final intervention decisions require qualitative appraisal by professional faculty advisors.
</div>
""", unsafe_allow_html=True)

# Navigation Tabs
tab_single, tab_batch, tab_audit = st.tabs([
    "🔍 Individual Student Advisory",
    "📊 Cohort Batch Analysis",
    "📈 Model Governance & Validation"
])

# =========================================================================
# TAB 1: INDIVIDUAL STUDENT INFERENCE
# =========================================================================
with tab_single:
    st.subheader("Student Academic & Behavioral Profile (Week 8 Mid-Term Cut-Off)")

    # Preset archetypes selector
    archetype = st.selectbox(
        "Select a Preset Student Archetype or customize below:",
        [
            "Custom Profile (Manual Input)",
            "Archetype A: Thriving High Achiever",
            "Archetype B: Slipping / Disengaging Student (Attendance Drop)",
            "Archetype C: Consistently Struggling Student",
            "Archetype D: Resilient Comeback Student"
        ]
    )

    # Default values based on archetype
    if archetype == "Archetype A: Thriving High Achiever":
        defaults = dict(att1=95.0, att2=94.0, sub_ratio=1.0, delay=0.0, q1=88.0, q2=91.0, mid=89.0, lms_log=12.0, lms_view=65.0, forum=8, gpa=3.80, creds=15, study=1)
    elif archetype == "Archetype B: Slipping / Disengaging Student (Attendance Drop)":
        defaults = dict(att1=88.0, att2=45.0, sub_ratio=0.65, delay=4.5, q1=74.0, q2=58.0, mid=48.0, lms_log=3.5, lms_view=18.0, forum=1, gpa=2.90, creds=18, study=0)
    elif archetype == "Archetype C: Consistently Struggling Student":
        defaults = dict(att1=52.0, att2=48.0, sub_ratio=0.50, delay=7.0, q1=42.0, q2=38.0, mid=36.0, lms_log=2.0, lms_view=12.0, forum=0, gpa=1.95, creds=15, study=0)
    elif archetype == "Archetype D: Resilient Comeback Student":
        defaults = dict(att1=75.0, att2=82.0, sub_ratio=0.90, delay=1.0, q1=52.0, q2=70.0, mid=68.0, lms_log=8.0, lms_view=45.0, forum=5, gpa=2.65, creds=15, study=1)
    else:
        defaults = dict(att1=75.0, att2=70.0, sub_ratio=0.80, delay=2.0, q1=65.0, q2=62.0, mid=58.0, lms_log=6.0, lms_view=30.0, forum=2, gpa=2.75, creds=15, study=1)

    with st.form("student_signals_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 📚 Academic Assessments")
            student_id = st.text_input("Student Identifier", value="STU-2025-0842")
            quiz_1 = st.slider("Diagnostic Quiz 1 Score (Week 3)", 0.0, 100.0, float(defaults["q1"]), 1.0)
            quiz_2 = st.slider("Progress Quiz 2 Score (Week 6)", 0.0, 100.0, float(defaults["q2"]), 1.0)
            midterm_score = st.slider("Midterm Examination Score (Week 8)", 0.0, 100.0, float(defaults["mid"]), 1.0)
            prior_gpa = st.slider("Prior Cumulative GPA (0.00 - 4.00)", 0.0, 4.0, float(defaults["gpa"]), 0.05)

        with col2:
            st.markdown("##### 🏫 Attendance & Submissions")
            att_w1_w4 = st.slider("Attendance Rate Weeks 1-4 (%)", 0.0, 100.0, float(defaults["att1"]), 1.0)
            att_w5_w8 = st.slider("Attendance Rate Weeks 5-8 (%)", 0.0, 100.0, float(defaults["att2"]), 1.0)
            sub_ratio = st.slider("Assignment Submission Ratio", 0.0, 1.0, float(defaults["sub_ratio"]), 0.05)
            delay_days = st.slider("Average Submission Delay (Days)", 0.0, 14.0, float(defaults["delay"]), 0.5)

        with col3:
            st.markdown("##### 💻 LMS & Peer Engagement")
            lms_logins = st.slider("Weekly LMS Logins", 0.0, 30.0, float(defaults["lms_log"]), 0.5)
            lms_views = st.slider("Resource / Reading Views", 0.0, 150.0, float(defaults["lms_view"]), 1.0)
            forum_posts = st.slider("Discussion Forum Posts", 0, 30, int(defaults["forum"]), 1)
            enrolled_creds = st.selectbox("Enrolled Credits", [12, 14, 15, 16, 18, 20, 21], index=2)
            study_group = st.radio("Peer Study Group Member", ["Yes (1)", "No (0)"], index=0 if defaults["study"] == 1 else 1)

        submit_btn = st.form_submit_button("⚡ Evaluate Student Academic Risk & Generate Advisory", use_container_width=True)

    if submit_btn:
        student_payload = {
            "student_id": student_id,
            "attendance_rate_w1_w4": att_w1_w4,
            "attendance_rate_w5_w8": att_w5_w8,
            "assignments_submitted_ratio": sub_ratio,
            "avg_assignment_delay_days": delay_days,
            "quiz_1_score": quiz_1,
            "quiz_2_score": quiz_2,
            "midterm_exam_score": midterm_score,
            "lms_login_frequency_per_week": lms_logins,
            "lms_resource_views": lms_views,
            "discussion_forum_posts": forum_posts,
            "cumulative_gpa_prior": prior_gpa,
            "course_credits_enrolled": enrolled_creds,
            "study_group_participation": 1 if "Yes" in study_group else 0
        }

        with st.spinner("Executing real model inference and SHAP local attribution..."):
            prob, band, is_at_risk, signals, interventions = explainer.explain_instance(student_payload)

        st.markdown("---")
        st.subheader("🎯 Advisory Prediction Assessment")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if prob < 0.30:
                card_class = "risk-card-low"
                badge = "🟢 LOW RISK (ON TRACK)"
            elif prob < 0.65:
                card_class = "risk-card-mod"
                badge = "🟡 MODERATE RISK (ADVISORY CHECK-IN)"
            else:
                card_class = "risk-card-high"
                badge = "🔴 HIGH RISK (PRIORITY ACADEMIC SUPPORT)"

            st.markdown(f"""
            <div class="{card_class}">
                <h3>{badge}</h3>
                <p><strong>Predicted Risk Probability:</strong> {prob*100:.1f}%</p>
                <p><strong>Evaluation Point:</strong> End of Week 8</p>
                <p><strong>Target Horizon:</strong> Final Exam (< 50% Threshold)</p>
                <p><strong>Classification:</strong> {'At-Risk' if is_at_risk else 'Satisfactory Progress'}</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(prob)

        with res_col2:
            st.markdown("##### 💡 Recommended Pedagogical Interventions")
            for item in interventions:
                st.markdown(f"- **Action:** {item}")

        st.markdown("---")
        st.subheader("🔎 Local Feature Attribution (SHAP Signals)")
        st.write("Displays the top 5 influential signals that shifted this specific student's risk probability:")

        sig_data = []
        for s in signals:
            sig_data.append({
                "Feature Signal": s.feature_name,
                "Observed Value": s.feature_value,
                "SHAP Impact Weight": s.importance_weight,
                "Direction": "Increases Risk ⚠️" if s.direction == "increases_risk" else "Decreases Risk ✅",
                "Advisor Context": s.clinical_note
            })
        st.table(pd.DataFrame(sig_data))

# =========================================================================
# TAB 2: COHORT BATCH ANALYSIS
# =========================================================================
with tab_batch:
    st.subheader("Cohort-wide Screening & Prioritized Advisory Queue")
    st.write("Evaluate an entire academic cohort to allocate academic support resources efficiently.")

    data_source = st.radio(
        "Cohort Data Source:",
        ["Use Held-out Blind Test Cohort (Cohort_2025_Spring - 890 students)", "Upload Custom Student CSV"],
        horizontal=True
    )

    if data_source == "Upload Custom Student CSV":
        uploaded_file = st.file_uploader("Upload CSV formatted with student features", type=["csv"])
        if uploaded_file:
            cohort_df = pd.read_csv(uploaded_file)
        else:
            cohort_df = None
    else:
        test_csv_path = "data/processed/test.csv"
        if os.path.exists(test_csv_path):
            cohort_df = pd.read_csv(test_csv_path)
        else:
            cohort_df = None

    if cohort_df is not None:
        st.success(f"Loaded cohort with {len(cohort_df)} student records.")

        if st.button("🚀 Run Cohort Risk Screening"):
            with st.spinner("Processing cohort predictions through LightGBM pipeline..."):
                raw_cols = explainer.raw_features
                X_cohort = explainer.pipeline.transform(cohort_df[raw_cols])
                cohort_probs = explainer.model.predict_proba(X_cohort)[:, 1]

                results_df = cohort_df[["student_id", "attendance_rate_w5_w8", "midterm_exam_score", "cumulative_gpa_prior"]].copy()
                results_df["Risk Probability (%)"] = np.round(cohort_probs * 100, 1)

                bands = []
                for p in cohort_probs:
                    if p < 0.30:
                        bands.append("Low Risk")
                    elif p < 0.65:
                        bands.append("Moderate Risk")
                    else:
                        bands.append("High Risk")
                results_df["Advisory Band"] = bands

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("##### Risk Band Distribution")
                    st.bar_chart(results_df["Advisory Band"].value_counts())

                with col_b:
                    st.markdown("##### Priority Advisory Intervention Queue (Top 10 High-Risk)")
                    priority_queue = results_df.sort_values(by="Risk Probability (%)", ascending=False).head(10)
                    st.dataframe(priority_queue, use_container_width=True)

                csv_data = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Full Cohort Advisory Report (CSV)",
                    data=csv_data,
                    file_name="cohort_at_risk_advisory_report.csv",
                    mime="text/csv"
                )

# =========================================================================
# TAB 3: MODEL GOVERNANCE & VALIDATION
# =========================================================================
with tab_audit:
    st.subheader("Model Governance, Comparative Benchmarks & Limitations")

    metrics_file = "artifacts/metrics_summary.json"
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            metrics_summary = json.load(f)

        st.markdown("##### Model Comparison on Held-out Out-of-Time Test Set (Cohort_2025_Spring)")
        comp_records = []
        for model_name, m in metrics_summary["model_comparison"].items():
            cm = m["confusion_matrix"]
            comp_records.append({
                "Model Candidate": model_name,
                "Accuracy": f"{m['test_accuracy']*100:.1f}%",
                "Precision": f"{m['test_precision']*100:.1f}%",
                "Recall (Sensitivity)": f"{m['test_recall']*100:.1f}%",
                "F1-Score": f"{m['test_f1']*100:.1f}%",
                "ROC-AUC": f"{m['test_roc_auc']:.3f}",
                "PR-AUC": f"{m['test_pr_auc']:.3f}",
                "True Positives (TP)": cm["TP"],
                "False Negatives (FN)": cm["FN"],
                "False Positives (FP)": cm["FP"]
            })
        st.dataframe(pd.DataFrame(comp_records), use_container_width=True)

    st.markdown("---")
    st.markdown("##### Analytical Diagnostic Visualizations")
    col_img1, col_img2 = st.columns(2)

    with col_img1:
        if os.path.exists("artifacts/figures/roc_pr_curves.png"):
            st.image("artifacts/figures/roc_pr_curves.png", caption="ROC & Precision-Recall Curves")
        if os.path.exists("artifacts/figures/confusion_matrices.png"):
            st.image("artifacts/figures/confusion_matrices.png", caption="Comparative Confusion Matrices")

    with col_img2:
        if os.path.exists("artifacts/figures/shap_summary.png"):
            st.image("artifacts/figures/shap_summary.png", caption="Global SHAP Feature Importance & Impact")
        if os.path.exists("artifacts/figures/threshold_tradeoff.png"):
            st.image("artifacts/figures/threshold_tradeoff.png", caption="Threshold vs Precision-Recall Trade-off")

    st.markdown("---")
    st.markdown("##### 🛡️ Ethical Principles, Data Privacy & Known Limitations")
    st.markdown("""
    1. **Strictly Non-Demographic (FERPA/GDPR Compliant):**
       - The model excludes all sensitive and demographic attributes (race, gender, religion, socioeconomic status, zip codes, disability).
       - Features capture exclusively learning habits and academic milestones up to Week 8.
    2. **False Negatives vs False Positives Cost Asymmetry:**
       - **False Negative Cost (Severe):** A student in distress is missed and fails without support.
       - **False Positive Cost (Low):** A student receives a supportive advisory check-in or optional peer mentoring invitation.
       - Operating threshold is calibrated to maintain high sensitivity (Recall > 80%).
    3. **Documented Failure Modes & Uncertainties:**
       - **Unmeasured Life Events:** The model cannot anticipate sudden personal, medical, or family emergencies occurring after Week 8.
       - **Unrecorded Offline Effort:** Students who study extensively in offline groups without LMS interaction may receive an inflated risk score.
       - **Human-in-the-Loop Mandate:** Every automated advisory flag must be validated by a human counselor before academic action is taken.
    """)
