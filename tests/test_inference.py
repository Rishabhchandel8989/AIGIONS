"""
Unit tests for model inference, SHAP explanations, schemas, and API handlers.
"""

import os
os.environ["MPLCONFIGDIR"] = os.path.abspath(".mpl_cache")
import pytest
from pydantic import ValidationError

from src.schemas import StudentSignals, BatchPredictionRequest, PredictionResponse
from src.explainability import ModelExplainer
import src.api as api_module


@pytest.fixture(scope="module")
def sample_student_dict():
    return {
        "student_id": "STU-TEST-9999",
        "attendance_rate_w1_w4": 82.0,
        "attendance_rate_w5_w8": 55.0,
        "assignments_submitted_ratio": 0.65,
        "avg_assignment_delay_days": 4.5,
        "quiz_1_score": 62.0,
        "quiz_2_score": 48.0,
        "midterm_exam_score": 42.0,
        "lms_login_frequency_per_week": 3.0,
        "lms_resource_views": 18.0,
        "discussion_forum_posts": 1,
        "cumulative_gpa_prior": 2.30,
        "course_credits_enrolled": 16,
        "study_group_participation": 0
    }


@pytest.fixture(scope="module")
def explainer_instance():
    exp = ModelExplainer(artifacts_dir="artifacts")
    api_module.explainer = exp
    return exp


def test_model_explainer_real_inference(explainer_instance, sample_student_dict):
    prob, band, is_at_risk, signals, interventions = explainer_instance.explain_instance(sample_student_dict)

    assert 0.0 <= prob <= 1.0, "Probability must be between 0 and 1"
    assert band in [
        "Low Risk (On Track)",
        "Moderate Risk (Advisory Check-in)",
        "High Risk (Priority Academic Support)"
    ]
    assert is_at_risk is True  # Given low scores and attendance drop
    assert len(signals) == 5, "Must return top 5 influential features"
    assert len(interventions) >= 1, "Must return actionable interventions"


def test_api_predict_handler(explainer_instance, sample_student_dict):
    signals_obj = StudentSignals(**sample_student_dict)
    response = api_module.predict_student_risk(signals_obj)

    assert isinstance(response, PredictionResponse)
    assert response.student_id == "STU-TEST-9999"
    assert 0.0 <= response.risk_probability <= 1.0
    assert "DECISION SUPPORT ONLY" in response.human_oversight_notice
    assert len(response.influential_signals) == 5


def test_api_batch_predict_handler(explainer_instance, sample_student_dict):
    signals_obj = StudentSignals(**sample_student_dict)
    batch_req = BatchPredictionRequest(students=[signals_obj, signals_obj])
    response = api_module.predict_batch_students(batch_req)

    assert response.total_processed == 2
    assert len(response.predictions) == 2
    assert response.predictions[0].student_id == "STU-TEST-9999"


def test_schema_input_validation(sample_student_dict):
    # Valid input succeeds
    obj = StudentSignals(**sample_student_dict)
    assert obj.attendance_rate_w5_w8 == 55.0

    # Illegal attendance > 100 must raise ValidationError
    bad_dict = sample_student_dict.copy()
    bad_dict["attendance_rate_w5_w8"] = 145.0
    with pytest.raises(ValidationError):
        StudentSignals(**bad_dict)

    # Illegal study group (must be 0 or 1)
    bad_dict2 = sample_student_dict.copy()
    bad_dict2["study_group_participation"] = 5
    with pytest.raises(ValidationError):
        StudentSignals(**bad_dict2)


def test_api_health_handler(explainer_instance):
    health = api_module.health_check()
    assert health["status"] == "healthy"
    assert health["model_loaded"] is True
