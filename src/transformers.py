"""
Scikit-learn compatible educational feature engineering transformers.
Isolated in a dedicated module to ensure clean serialization across all execution contexts.
"""

from typing import List
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class EducationalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that engineers domain-specific learning signals
    strictly from Week 8 available features.
    """
    def __init__(self):
        self.engineered_feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Attendance Momentum: Trajectory between early and mid semester
        df["attendance_momentum"] = df["attendance_rate_w5_w8"] - df["attendance_rate_w1_w4"]

        # 2. Critical Attendance Drop Indicator: Drop of > 15 percentage points
        df["attendance_drop_flag"] = (df["attendance_momentum"] < -15.0).astype(float)

        # 3. Academic Quiz Velocity: Improvement or decline from Quiz 1 to Quiz 2
        df["quiz_velocity"] = df["quiz_2_score"] - df["quiz_1_score"]

        # 4. Midterm to Quiz Ratio: High-stakes vs low-stakes examination performance
        df["midterm_to_quiz_ratio"] = df["midterm_exam_score"] / (df["quiz_2_score"] + 1e-5)

        # 5. Assignment Latency Risk Indicator: Excessive delay (> 3 days) or low submission (< 70%)
        df["assignment_risk_flag"] = (
            (df["avg_assignment_delay_days"] > 3.0) | (df["assignments_submitted_ratio"] < 0.70)
        ).astype(float)

        # 6. Academic Momentum Index: Combines prior GPA and midterm mastery
        df["academic_standing_index"] = (df["cumulative_gpa_prior"] * df["midterm_exam_score"]) / 100.0

        # 7. LMS Total Engagement Index: Activity density
        df["lms_engagement_density"] = (
            df["lms_login_frequency_per_week"] * 2.0 +
            df["lms_resource_views"] * 0.1 +
            df["discussion_forum_posts"] * 1.5
        )

        return df

