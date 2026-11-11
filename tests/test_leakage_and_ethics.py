"""
Unit tests for ethical compliance (no protected attributes) and temporal data leakage prevention.
"""

import os
import re
import pytest
import pandas as pd
import joblib


def test_ethical_non_discrimination():
    """
    Ensure zero sensitive demographic or protected attributes exist in training data or feature pipeline.
    Uses regex word boundaries to avoid false positives (e.g. 'age' inside 'engagement').
    """
    pipeline_bundle = joblib.load(os.path.join("artifacts", "feature_pipeline.joblib"))
    raw_features = pipeline_bundle["raw_features"]
    engineered_features = pipeline_bundle["engineered_features"]

    prohibited_keywords = [
        r"\bgender\b", r"\bsex\b", r"\brace\b", r"\bethnicity\b", r"\breligion\b",
        r"\bcaste\b", r"\bincome\b", r"\bzipcode\b", r"\bdisability\b",
        r"\bnationality\b", r"\bmarital\b", r"\bage\b"
    ]

    for feat in raw_features + engineered_features:
        for pattern in prohibited_keywords:
            assert not re.search(pattern, feat.lower()), f"Ethical violation: prohibited term '{pattern}' found in feature '{feat}'"


def test_temporal_leakage_safeguards():
    """
    Verify that features used in prediction strictly represent signals available on or before Week 8.
    """
    pipeline_bundle = joblib.load(os.path.join("artifacts", "feature_pipeline.joblib"))
    raw_features = pipeline_bundle["raw_features"]

    post_week8_keywords = ["final", "week_9", "week_10", "week_11", "week_12", "week_13", "week_14", "week_15", "week_16"]

    for feat in raw_features:
        for forbidden in post_week8_keywords:
            assert forbidden not in feat.lower(), f"Leakage detected: post-Week 8 attribute '{forbidden}' found in '{feat}'"


def test_chronological_splits():
    """
    Verify that train, validation, and test cohorts are mutually exclusive chronologically.
    """
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    train_cohorts = set(train_df["cohort_term"].unique())
    val_cohorts = set(val_df["cohort_term"].unique())
    test_cohorts = set(test_df["cohort_term"].unique())

    # Mutual exclusivity
    assert len(train_cohorts.intersection(val_cohorts)) == 0, "Train and Val cohorts must not overlap"
    assert len(train_cohorts.intersection(test_cohorts)) == 0, "Train and Test cohorts must not overlap"
    assert len(val_cohorts.intersection(test_cohorts)) == 0, "Val and Test cohorts must not overlap"

    # Held out future test cohort verification
    assert "Cohort_2025_Spring" in test_cohorts
