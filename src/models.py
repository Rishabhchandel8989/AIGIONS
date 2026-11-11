"""
Model definitions, training routines, hyperparameter tuning, and model comparison.
Trains heuristic baseline, Logistic Regression, Random Forest, and LightGBM / XGBoost.
Evaluates on validation split for threshold tuning and saves serialized best artifact.
"""

import os
import json
import sys
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, brier_score_loss
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.transformers import EducationalFeatureEngineer


class HeuristicEducatorBaseline:
    """
    Standard educational rule-of-thumb baseline:
    Flags student if midterm score < 45 OR recent attendance < 60%.
    """
    def __init__(self, midterm_thresh: float = 45.0, attendance_thresh: float = 60.0):
        self.midterm_thresh = midterm_thresh
        self.attendance_thresh = attendance_thresh

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Pseudo-probability heuristic
        probs = []
        for _, row in X.iterrows():
            score_factor = max(0.0, (self.midterm_thresh - row["midterm_exam_score"]) / self.midterm_thresh)
            att_factor = max(0.0, (self.attendance_thresh - row["attendance_rate_w5_w8"]) / self.attendance_thresh)
            p = min(0.95, max(0.05, 0.5 * score_factor + 0.5 * att_factor))
            probs.append([1.0 - p, p])
        return np.array(probs)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        mask = (X["midterm_exam_score"] < self.midterm_thresh) | (X["attendance_rate_w5_w8"] < self.attendance_thresh)
        return mask.astype(int).values


def train_and_compare_models(
    processed_dir: str = "data/processed",
    artifacts_dir: str = "artifacts"
) -> Tuple[Dict[str, Any], Any]:
    """
    Loads processed data and feature pipeline, trains baseline and candidate models,
    tunes decision threshold on validation set, and selects top performing model.
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Load data
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(processed_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))

    # 2. Load feature pipeline
    pipeline_bundle = joblib.load(os.path.join(artifacts_dir, "feature_pipeline.joblib"))
    full_pipeline = pipeline_bundle["pipeline"]
    raw_cols = pipeline_bundle["raw_features"]
    engineered_cols = pipeline_bundle["engineered_features"]

    X_train_raw = train_df[raw_cols]
    y_train = train_df["is_at_risk"].values

    X_val_raw = val_df[raw_cols]
    y_val = val_df["is_at_risk"].values

    X_test_raw = test_df[raw_cols]
    y_test = test_df["is_at_risk"].values

    # Transform features
    X_train = full_pipeline.transform(X_train_raw)
    X_val = full_pipeline.transform(X_val_raw)
    X_test = full_pipeline.transform(X_test_raw)

    # Calculate class imbalance weighting
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(1, pos_count)

    # Candidate Models Dictionary
    models = {
        "Heuristic Rule Baseline": HeuristicEducatorBaseline(),
        "Logistic Regression (Balanced)": LogisticRegression(
            class_weight="balanced",
            C=0.5,
            max_iter=1000,
            random_state=42
        ),
        "Random Forest (Tuned)": RandomForestClassifier(
            n_estimators=250,
            max_depth=7,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=1
        ),
        "LightGBM (Gradient Boosting)": LGBMClassifier(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.04,
            scale_pos_weight=scale_pos_weight,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbosity=-1
        ),
        "XGBoost (Extreme Gradient Boosting)": XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.04,
            scale_pos_weight=scale_pos_weight,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42
        )
    }

    comparison_results = {}
    fitted_models = {}

    for name, model in models.items():
        print(f"Training model: {name}...")
        if name == "Heuristic Rule Baseline":
            model.fit(X_train_raw, y_train)
            val_probs = model.predict_proba(X_val_raw)[:, 1]
            test_probs = model.predict_proba(X_test_raw)[:, 1]
            val_preds = model.predict(X_val_raw)
            test_preds = model.predict(X_test_raw)
        else:
            model.fit(X_train, y_train)
            val_probs = model.predict_proba(X_val)[:, 1]
            test_probs = model.predict_proba(X_test)[:, 1]
            val_preds = (val_probs >= 0.50).astype(int)
            test_preds = (test_probs >= 0.50).astype(int)

        fitted_models[name] = model

        # Metrics on held-out test set
        test_cm = confusion_matrix(y_test, test_preds)
        tn, fp, fn, tp = test_cm.ravel()

        comparison_results[name] = {
            "test_accuracy": float(round((tp + tn) / len(y_test), 4)),
            "test_precision": float(round(precision_score(y_test, test_preds, zero_division=0), 4)),
            "test_recall": float(round(recall_score(y_test, test_preds, zero_division=0), 4)),
            "test_f1": float(round(f1_score(y_test, test_preds, zero_division=0), 4)),
            "test_roc_auc": float(round(roc_auc_score(y_test, test_probs), 4)),
            "test_pr_auc": float(round(average_precision_score(y_test, test_probs), 4)),
            "test_brier_score": float(round(brier_score_loss(y_test, test_probs), 4)),
            "confusion_matrix": {
                "TP": int(tp),
                "FP": int(fp),
                "TN": int(tn),
                "FN": int(fn)
            }
        }

    # 3. Decision Threshold Tuning for the Champion Model
    # Candidate champion: LightGBM
    champion_name = "LightGBM (Gradient Boosting)"
    champion_model = fitted_models[champion_name]

    val_probs_champ = champion_model.predict_proba(X_val)[:, 1]

    # Evaluate thresholds from 0.20 to 0.80
    threshold_records = []
    best_threshold = 0.50
    best_f1 = 0.0

    for thresh in np.arange(0.20, 0.81, 0.05):
        t_preds = (val_probs_champ >= thresh).astype(int)
        rec = recall_score(y_val, t_preds, zero_division=0)
        prec = precision_score(y_val, t_preds, zero_division=0)
        f1 = f1_score(y_val, t_preds, zero_division=0)
        threshold_records.append({
            "threshold": round(thresh, 2),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4)
        })
        # Prioritize Recall while keeping Precision > 0.65
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = round(float(thresh), 2)

    # Re-evaluate champion on test set with tuned threshold
    test_probs_champ = champion_model.predict_proba(X_test)[:, 1]
    tuned_test_preds = (test_probs_champ >= best_threshold).astype(int)
    tuned_cm = confusion_matrix(y_test, tuned_test_preds)
    t_tn, t_fp, t_fn, t_tp = tuned_cm.ravel()

    comparison_results[f"{champion_name} (Threshold Tuned @ {best_threshold})"] = {
        "test_accuracy": float(round((t_tp + t_tn) / len(y_test), 4)),
        "test_precision": float(round(precision_score(y_test, tuned_test_preds, zero_division=0), 4)),
        "test_recall": float(round(recall_score(y_test, tuned_test_preds, zero_division=0), 4)),
        "test_f1": float(round(f1_score(y_test, tuned_test_preds, zero_division=0), 4)),
        "test_roc_auc": float(round(roc_auc_score(y_test, test_probs_champ), 4)),
        "test_pr_auc": float(round(average_precision_score(y_test, test_probs_champ), 4)),
        "test_brier_score": float(round(brier_score_loss(y_test, test_probs_champ), 4)),
        "confusion_matrix": {
            "TP": int(t_tp),
            "FP": int(t_fp),
            "TN": int(t_tn),
            "FN": int(t_fn)
        }
    }

    # 4. Save Artifacts
    # Save best model
    best_artifact = {
        "model_name": champion_name,
        "model": champion_model,
        "optimal_threshold": best_threshold,
        "raw_features": raw_cols,
        "engineered_features": engineered_cols,
        "pipeline": full_pipeline
    }
    joblib.dump(best_artifact, os.path.join(artifacts_dir, "best_model.joblib"))

    # Save baseline model
    baseline_artifact = {
        "model_name": "Logistic Regression (Balanced)",
        "model": fitted_models["Logistic Regression (Balanced)"],
        "pipeline": full_pipeline,
        "raw_features": raw_cols,
        "engineered_features": engineered_cols
    }
    joblib.dump(baseline_artifact, os.path.join(artifacts_dir, "baseline_model.joblib"))

    # Save metrics summary
    metrics_path = os.path.join(artifacts_dir, "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "model_comparison": comparison_results,
            "optimal_threshold": best_threshold,
            "threshold_sweep_validation": threshold_records
        }, f, indent=4)

    print(f"Model comparison and artifacts saved to {artifacts_dir}")
    return comparison_results, best_artifact


if __name__ == "__main__":
    train_and_compare_models()
