"""
Model Explainability and Responsible AI Engine.
Generates SHAP (SHapley Additive exPlanations) global importance plots and
per-prediction local attributions with domain-aligned actionable recommendations.
"""

import os
os.environ["MPLCONFIGDIR"] = os.path.abspath(".mpl_cache")
os.makedirs(".mpl_cache", exist_ok=True)
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple

from src.transformers import EducationalFeatureEngineer
from src.schemas import InfluentialSignal


class ModelExplainer:
    """
    SHAP-based explainer for global model auditing and local student advisory explanations.
    """
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        best_bundle = joblib.load(os.path.join(artifacts_dir, "best_model.joblib"))
        pipeline_bundle = joblib.load(os.path.join(artifacts_dir, "feature_pipeline.joblib"))

        self.model = best_bundle["model"]
        self.pipeline = pipeline_bundle["pipeline"]
        self.raw_features = pipeline_bundle["raw_features"]
        self.engineered_features = pipeline_bundle["engineered_features"]
        self.optimal_threshold = best_bundle.get("optimal_threshold", 0.50)

        # Initialize TreeExplainer on fitted booster
        self.explainer = shap.TreeExplainer(self.model)

    def generate_global_summary_plots(
        self,
        test_data_path: str = "data/processed/test.csv",
        output_fig_dir: str = "artifacts/figures"
    ):
        """
        Computes SHAP values on held-out test data and exports beeswarm and bar summary plots.
        """
        os.makedirs(output_fig_dir, exist_ok=True)
        test_df = pd.read_csv(test_data_path)
        X_test_raw = test_df[self.raw_features]
        X_test_trans = self.pipeline.transform(X_test_raw)

        shap_values = self.explainer.shap_values(X_test_trans)
        # For binary classification in LightGBM, shap_values can be a list [class0, class1] or 2D array
        if isinstance(shap_values, list):
            sv = shap_values[1]
        elif len(shap_values.shape) == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            sv,
            X_test_trans,
            feature_names=self.engineered_features,
            show=False,
            max_display=15
        )
        plt.title("SHAP Global Feature Importance & Impact Directionality", fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        out_path = os.path.join(output_fig_dir, "shap_summary.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"SHAP summary plot saved to: {out_path}")

    def explain_instance(
        self,
        student_data: Dict[str, Any]
    ) -> Tuple[float, str, bool, List[InfluentialSignal], List[str]]:
        """
        Explains an individual student prediction using local SHAP attribution values.
        Returns risk probability, advisory tier, at-risk flag, influential signals, and tailored interventions.
        """
        df_single = pd.DataFrame([student_data])
        X_trans = self.pipeline.transform(df_single[self.raw_features])

        # Risk probability
        prob = float(self.model.predict_proba(X_trans)[0, 1])

        # Advisory Risk Band
        if prob < 0.30:
            band = "Low Risk (On Track)"
            is_at_risk = False
        elif prob < 0.65:
            band = "Moderate Risk (Advisory Check-in)"
            is_at_risk = True
        else:
            band = "High Risk (Priority Academic Support)"
            is_at_risk = True

        # Local SHAP attributions
        shap_vals = self.explainer.shap_values(X_trans)
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]
        elif len(shap_vals.shape) == 3:
            sv = shap_vals[:, :, 1][0]
        elif len(shap_vals.shape) == 2:
            sv = shap_vals[0]
        else:
            sv = shap_vals

        # Map top influential features
        top_indices = np.argsort(np.abs(sv))[::-1][:5]
        influential_signals = []

        fe_df = self.pipeline.named_steps['fe'].transform(df_single[self.raw_features])

        for idx in top_indices:
            feat_name = self.engineered_features[idx]
            feat_val = float(fe_df[feat_name].iloc[0])
            attribution = float(sv[idx])
            direction = "increases_risk" if attribution > 0 else "decreases_risk"

            note = self._generate_pedagogical_note(feat_name, feat_val, direction)

            influential_signals.append(InfluentialSignal(
                feature_name=feat_name,
                feature_value=round(feat_val, 2),
                importance_weight=round(attribution, 4),
                direction=direction,
                clinical_note=note
            ))

        # Contextual intervention recommendations
        interventions = self._generate_interventions(student_data, prob)

        return prob, band, is_at_risk, influential_signals, interventions

    def _generate_pedagogical_note(self, feat_name: str, val: float, direction: str) -> str:
        templates = {
            "midterm_exam_score": f"Midterm score of {val:.1f}% {'signals mastery deficit' if direction == 'increases_risk' else 'reflects solid core grasp'}.",
            "attendance_rate_w5_w8": f"Weeks 5-8 attendance at {val:.1f}% {'indicates concerning absenteeism' if direction == 'increases_risk' else 'shows consistent class engagement'}.",
            "attendance_momentum": f"Attendance trend of {val:+.1f}% {'shows steep mid-semester drop-off' if direction == 'increases_risk' else 'shows stable or improving presence'}.",
            "quiz_2_score": f"Quiz 2 score of {val:.1f}% {'signals difficulty with advanced concepts' if direction == 'increases_risk' else 'confirms academic velocity'}.",
            "avg_assignment_delay_days": f"Average submission delay of {val:.1f} days {'points to time-management obstacles' if direction == 'increases_risk' else 'reflects timely milestone completion'}.",
            "assignments_submitted_ratio": f"Assignment submission rate of {val*100:.1f}% {'indicates incomplete coursework' if direction == 'increases_risk' else 'demonstrates consistent effort'}.",
            "cumulative_gpa_prior": f"Prior cumulative GPA of {val:.2f} {'shows prior academic vulnerability' if direction == 'increases_risk' else 'provides a strong foundational buffer'}.",
            "lms_engagement_density": f"LMS engagement density score of {val:.1f} {'indicates low platform interaction' if direction == 'increases_risk' else 'reflects proactive resource review'}."
        }
        return templates.get(feat_name, f"Feature {feat_name} ({val:.2f}) {direction.replace('_', ' ')}.")

    def _generate_interventions(self, data: Dict[str, Any], prob: float) -> List[str]:
        recommendations = []
        if prob < 0.30:
            recommendations.append("Continue standard pedagogical support; encourage peer tutoring participation.")
            return recommendations

        if data.get("attendance_rate_w5_w8", 100) < 70 or (data.get("attendance_rate_w5_w8", 100) - data.get("attendance_rate_w1_w4", 100)) < -10:
            recommendations.append("Schedule 1-on-1 advisor outreach to identify personal or schedule barriers causing attendance drop.")

        if data.get("avg_assignment_delay_days", 0) > 2.0 or data.get("assignments_submitted_ratio", 1.0) < 0.75:
            recommendations.append("Enroll student in Academic Time-Management Coaching; offer structured incremental assignment milestones.")

        if data.get("midterm_exam_score", 100) < 55 or data.get("quiz_2_score", 100) < 55:
            recommendations.append("Assign dedicated Subject Peer Tutor for targeted concept review before final exam.")

        if data.get("study_group_participation", 1) == 0:
            recommendations.append("Facilitate placement into a collaborative course study circle.")

        if data.get("lms_login_frequency_per_week", 10) < 4.0:
            recommendations.append("Send digital learning resource guide and schedule brief LMS navigation tutorial.")

        if not recommendations:
            recommendations.append("General academic advisory check-in to review mid-term study strategies.")

        return recommendations


if __name__ == "__main__":
    explainer = ModelExplainer()
    explainer.generate_global_summary_plots()
