"""
Preprocessing and Feature Engineering Pipeline.
Transforms raw educational signals into high-signal predictive features.
Enforces chronological split to eliminate temporal leakage.
Fits scalers strictly on training data.
"""

import os
from typing import Tuple, List, Dict
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.transformers import EducationalFeatureEngineer


RAW_FEATURE_COLS = [
    "attendance_rate_w1_w4",
    "attendance_rate_w5_w8",
    "assignments_submitted_ratio",
    "avg_assignment_delay_days",
    "quiz_1_score",
    "quiz_2_score",
    "midterm_exam_score",
    "lms_login_frequency_per_week",
    "lms_resource_views",
    "discussion_forum_posts",
    "cumulative_gpa_prior",
    "course_credits_enrolled",
    "study_group_participation"
]


def split_data_chronologically(
    df: pd.DataFrame,
    train_cohorts: List[str] = ["Cohort_2023_Fall", "Cohort_2024_Spring"],
    val_cohorts: List[str] = ["Cohort_2024_Fall"],
    test_cohorts: List[str] = ["Cohort_2025_Spring"]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs out-of-time chronological cohort splitting to rigorously prevent data leakage.
    Train: Historical cohorts
    Val: Intermediary cohort for tuning & threshold optimization
    Test: Future held-out cohort for blind acceptance testing
    """
    train_df = df[df["cohort_term"].isin(train_cohorts)].copy().reset_index(drop=True)
    val_df = df[df["cohort_term"].isin(val_cohorts)].copy().reset_index(drop=True)
    test_df = df[df["cohort_term"].isin(test_cohorts)].copy().reset_index(drop=True)

    print(f"Chronological split completed:")
    print(f"  - Train cohorts {train_cohorts}: {len(train_df)} records ({train_df['is_at_risk'].mean()*100:.1f}% at-risk)")
    print(f"  - Val cohorts   {val_cohorts}: {len(val_df)} records ({val_df['is_at_risk'].mean()*100:.1f}% at-risk)")
    print(f"  - Test cohorts  {test_cohorts}: {len(test_df)} records ({test_df['is_at_risk'].mean()*100:.1f}% at-risk)")

    return train_df, val_df, test_df


def build_and_fit_pipeline(
    train_df: pd.DataFrame,
    artifacts_dir: str = "artifacts"
) -> Tuple[Pipeline, List[str]]:
    """
    Fits the feature engineering and standard scaling pipeline strictly on training data.
    Saves serialized pipeline artifact.
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    fe = EducationalFeatureEngineer()
    train_fe = fe.transform(train_df[RAW_FEATURE_COLS])
    feature_names = list(train_fe.columns)

    full_pipeline = Pipeline([
        ('fe', EducationalFeatureEngineer()),
        ('scaler', StandardScaler())
    ])
    full_pipeline.fit(train_df[RAW_FEATURE_COLS])

    pipeline_path = os.path.join(artifacts_dir, "feature_pipeline.joblib")
    joblib.dump({
        "pipeline": full_pipeline,
        "raw_features": RAW_FEATURE_COLS,
        "engineered_features": feature_names
    }, pipeline_path)
    print(f"Fitted feature pipeline saved to: {pipeline_path}")

    return full_pipeline, feature_names


def prepare_processed_datasets(
    raw_path: str = "data/raw/student_academic_records.csv",
    output_dir: str = "data/processed",
    artifacts_dir: str = "artifacts"
):
    """
    Orchestrates end-to-end splitting, feature engineering, and data export.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(raw_path)

    train_df, val_df, test_df = split_data_chronologically(df)

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Exported processed train/val/test splits to {output_dir}")

    build_and_fit_pipeline(train_df, artifacts_dir=artifacts_dir)


if __name__ == "__main__":
    prepare_processed_datasets()
