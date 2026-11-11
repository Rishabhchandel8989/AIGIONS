"""
Unit tests for data generation and schema integrity.
"""

import pytest
import pandas as pd
from src.data_generator import generate_synthetic_student_data


def test_synthetic_data_generation_volume_and_columns():
    df = generate_synthetic_student_data(n_samples=1200, random_seed=42)

    assert len(df) == 1200, "Dataset must match requested record count"
    assert "student_id" in df.columns
    assert "cohort_term" in df.columns
    assert "final_score" in df.columns
    assert "is_at_risk" in df.columns

    # Verify no null values
    assert df.isnull().sum().sum() == 0, "Generated data must contain zero null values"

    # Verify unique student IDs
    assert df["student_id"].nunique() == 1200, "Every student ID must be unique"


def test_synthetic_data_distributions_and_ranges():
    df = generate_synthetic_student_data(n_samples=500, random_seed=42)

    assert df["attendance_rate_w1_w4"].between(0.0, 100.0).all()
    assert df["attendance_rate_w5_w8"].between(0.0, 100.0).all()
    assert df["assignments_submitted_ratio"].between(0.0, 1.0).all()
    assert df["avg_assignment_delay_days"].ge(0.0).all()
    assert df["midterm_exam_score"].between(0.0, 100.0).all()
    assert df["cumulative_gpa_prior"].between(0.0, 4.0).all()

    # Target sanity check
    assert set(df["is_at_risk"].unique()).issubset({0, 1})
    at_risk_ratio = df["is_at_risk"].mean()
    assert 0.15 <= at_risk_ratio <= 0.35, f"Realistic academic at-risk ratio expected, got {at_risk_ratio:.2f}"

