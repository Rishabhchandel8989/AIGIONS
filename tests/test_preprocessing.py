"""
Unit tests for feature engineering transformations and scaling pipeline.
"""

import os
import pytest
import pandas as pd
import numpy as np
import joblib

from src.transformers import EducationalFeatureEngineer
from src.preprocessing import RAW_FEATURE_COLS


def test_educational_feature_engineer():
    sample_data = pd.DataFrame({
        "attendance_rate_w1_w4": [80.0, 90.0],
        "attendance_rate_w5_w8": [60.0, 95.0],
        "assignments_submitted_ratio": [0.60, 1.0],
        "avg_assignment_delay_days": [4.0, 0.0],
        "quiz_1_score": [65.0, 85.0],
        "quiz_2_score": [50.0, 90.0],
        "midterm_exam_score": [45.0, 92.0],
        "lms_login_frequency_per_week": [3.0, 15.0],
        "lms_resource_views": [20.0, 80.0],
        "discussion_forum_posts": [1, 10],
        "cumulative_gpa_prior": [2.4, 3.8],
        "course_credits_enrolled": [15, 15],
        "study_group_participation": [0, 1]
    })

    fe = EducationalFeatureEngineer()
    transformed = fe.transform(sample_data)

    assert "attendance_momentum" in transformed.columns
    assert "attendance_drop_flag" in transformed.columns
    assert "quiz_velocity" in transformed.columns
    assert "midterm_to_quiz_ratio" in transformed.columns
    assert "assignment_risk_flag" in transformed.columns
    assert "academic_standing_index" in transformed.columns
    assert "lms_engagement_density" in transformed.columns

    # Check momentum values: 60 - 80 = -20
    assert transformed["attendance_momentum"].iloc[0] == -20.0
    assert transformed["attendance_drop_flag"].iloc[0] == 1.0  # dropped by > 15
    assert transformed["attendance_drop_flag"].iloc[1] == 0.0


def test_saved_feature_pipeline_persistence():
    pipeline_path = os.path.join("artifacts", "feature_pipeline.joblib")
    assert os.path.exists(pipeline_path), "Serialized feature pipeline artifact must exist"

    bundle = joblib.load(pipeline_path)
    assert "pipeline" in bundle
    assert "raw_features" in bundle
    assert "engineered_features" in bundle

    pipeline = bundle["pipeline"]
    assert len(bundle["raw_features"]) == len(RAW_FEATURE_COLS)
