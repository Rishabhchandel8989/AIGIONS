# 🎥 3-to-5 Minute Demonstration Video Script
### Student Performance & At-Risk Prediction | Algions AI Labs Assessment
**Target Length:** 3:30 to 4:30 Minutes | **Presenter Role:** Candidate (AI & Machine Learning Intern)

---

## 🎬 Video Recording Checklist Before You Record:
1. Terminal 1 open: `.venv` activated, ready to launch Streamlit (`streamlit run app/streamlit_app.py`).
2. Terminal 2 open: ready to show test suite execution (`pytest tests -v`).
3. Browser open with:
   - Streamlit App at `http://localhost:8501`
   - FastAPI Interactive Swagger Docs at `http://localhost:8000/docs`
   - Folder `artifacts/figures/` open in your file viewer or IDE.

---

## ⏱️ Timed Presentation Script

### ⏱️ [0:00 - 0:45] Introduction & Problem Framing
* **What to Show on Screen:** Title slide or the top of `README.md` / Streamlit UI.
* **What to Speak:**
  > *"Hello everyone and reviewers at Algions AI Labs. My name is [Your Full Name], and today I am excited to present my solution for the Student Performance & At-Risk Prediction assessment in the School Management domain.*
  >
  > *In educational institutions, reactive interventions come too late—usually after students have already failed their final exams. Our mission was to build an early decision-support system at the mid-term (Week 8) to identify students who may benefit from academic tutoring before Week 16 final examinations.*
  >
  > *From an ethical and compliance standpoint, our solution strictly complies with FERPA and GDPR: it excludes all sensitive demographic or protected attributes like race, gender, or socioeconomic status, operating exclusively on observable learning habits, attendance trends, and assessment scores. Most importantly, it is strictly a decision-support tool—it never triggers automated punishment, grade reduction, or admission decisions."*

---

### ⏱️ [0:45 - 1:45] Data Preparation, Leakage Safeguards & Model Comparison
* **What to Show on Screen:** Open the **"Model Governance & Validation"** tab in Streamlit, or show `artifacts/figures/roc_pr_curves.png` and `confusion_matrices.png`.
* **What to Speak:**
  > *"Let's examine our data preparation and validation strategy.*
  >
  > *To ensure zero temporal data leakage, our dataset of 3,500 students is split chronologically across academic cohorts: historical cohorts from 2023 Fall and 2024 Spring form our training set, 2024 Fall serves as validation, and 2025 Spring—comprising 890 students—is our completely blind, held-out test set.*
  >
  > *We engineered domain-specific signals including attendance momentum—measuring the change in attendance between weeks 1-4 and 5-8—academic quiz velocity, assignment delay days, and LMS engagement density.*
  >
  > *We trained and benchmarked five models: a heuristic educator rule baseline, Logistic Regression with balanced weighting, Random Forest, XGBoost, and our champion model, LightGBM.*
  >
  > *On the blind held-out test cohort, our champion LightGBM model achieved a ROC-AUC of 0.928 and a PR-AUC of 0.807, maintaining an 85% recall. Crucially, compared to the heuristic baseline used in many schools, our model slashed false positive alarms by over 55%, preventing academic advisor fatigue while ensuring struggling students are not missed."*

---

### ⏱️ [1:45 - 3:15] Live Inference & Interactive Demonstration
* **What to Show on Screen:** Switch to the **"Individual Student Advisory"** tab in the Streamlit application (`http://localhost:8501`).
* **What to Speak:**
  > *"Now, let's look at the working inference interface.*
  >
  > *Notice that our application directly loads the serialized trained model and feature pipeline artifacts—there are zero hardcoded predictions.*
  >
  > *Let's test Archetype B: A Slipping / Disengaging Student. In weeks 1 through 4, this student had strong attendance at 88%, but in weeks 5 to 8, attendance collapsed to 45%, assignment delays averaged 4.5 days, and their midterm score dropped to 48%.*
  >
  > *(Click '⚡ Evaluate Student Academic Risk & Generate Advisory')*
  >
  > *Instantly, the model predicts an 88.3% risk probability, placing the student in the 'High Risk: Priority Academic Support' band.*
  >
  > *Notice the local explainability section powered by SHAP: the advisor immediately sees that the primary risk drivers are the midterm mastery deficit and the sharp negative attendance momentum. The system automatically prescribes concrete pedagogical interventions: scheduling a 1-on-1 advisor check-in to identify personal barriers, academic time-management coaching, and assigning a subject peer tutor.*
  >
  > *Next, let's switch to the 'Cohort Batch Analysis' tab.*
  > *(Click on Cohort Batch Analysis tab, select blind test cohort, and click 'Run Cohort Risk Screening')*
  > *Here, the academic dean or department chair can screen an entire cohort of hundreds of students in seconds, view the prioritized high-risk intervention queue, and download an advisory CSV report."*

---

### ⏱️ [3:15 - 4:00] API & Engineering Verification
* **What to Show on Screen:** Switch to the browser tab showing FastAPI Swagger Docs (`http://localhost:8000/docs`) or terminal running `pytest tests -v`.
* **What to Speak:**
  > *"To ensure seamless institutional integration, we also expose this entire pipeline through a production-grade FastAPI REST service with endpoints for single predictions, batch scoring, model health, and audit metrics, backed by strict Pydantic data schemas.*
  >
  > *In the terminal, our automated test suite runs 12 unit tests covering data generation, schema validation, ethical non-discrimination assertions, temporal leakage checks, and pipeline persistence.*
  > *(Show terminal with 12 passed tests)*
  > *The entire system can be reproduced end-to-end with a single command: `python run_pipeline.py`."*

---

### ⏱️ [4:00 - 4:30] Conclusion & Responsible AI Summary
* **What to Show on Screen:** Return to Streamlit UI with the Ethical Governance notice visible.
* **What to Speak:**
  > *"In conclusion, this project delivers a defensible, reproducible, and ethical machine learning decision-support solution for school management. It respects student privacy, provides transparent SHAP explanations, optimizes for the asymmetric cost of false negatives, and empowers educators to intervene early.*
  >
  > *Thank you for your time, and I look forward to discussing this in detail during the technical walkthrough!"*

---

## 💡 Top Tips for Recording:
* **Audio Quality:** Use a clean headset mic in a quiet room.
* **Resolution:** Record at 1080p (1920x1080) at 30 or 60 fps using OBS Studio or Loom.
* **Pacing:** Speak at a steady, calm, professional pace. Don't rush!
* **Upload:** Upload as unlisted on YouTube, Google Drive (make sure link sharing is set to **"Anyone with the link can view"**), or Loom, and paste the link into the submission form.

