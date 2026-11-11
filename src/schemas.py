"""
Data schemas and validation contracts using Pydantic.
Ensures strict validation of incoming student signals and structured inference responses.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field, field_validator


class StudentSignals(BaseModel):
    """
    Historical and mid-term learning signals collected strictly up to Week 8.
    Does NOT contain any sensitive or protected demographic attributes.
    """
    student_id: Optional[str] = Field(default="STU-0000", description="Unique anonymized student identifier")
    attendance_rate_w1_w4: float = Field(..., ge=0.0, le=100.0, description="Attendance rate in weeks 1 to 4 (%)")
    attendance_rate_w5_w8: float = Field(..., ge=0.0, le=100.0, description="Attendance rate in weeks 5 to 8 (%)")
    assignments_submitted_ratio: float = Field(..., ge=0.0, le=1.0, description="Proportion of assignments submitted up to week 8 (0.0 to 1.0)")
    avg_assignment_delay_days: float = Field(..., ge=0.0, le=30.0, description="Average submission delay in days")
    quiz_1_score: float = Field(..., ge=0.0, le=100.0, description="Diagnostic Quiz 1 score (0-100)")
    quiz_2_score: float = Field(..., ge=0.0, le=100.0, description="Progress Quiz 2 score (0-100)")
    midterm_exam_score: float = Field(..., ge=0.0, le=100.0, description="Midterm examination score (0-100)")
    lms_login_frequency_per_week: float = Field(..., ge=0.0, le=50.0, description="Average LMS portal logins per week")
    lms_resource_views: float = Field(..., ge=0.0, le=500.0, description="Total learning resources/documents viewed up to week 8")
    discussion_forum_posts: float = Field(..., ge=0.0, le=200.0, description="Number of forum contributions/questions posted")
    cumulative_gpa_prior: float = Field(..., ge=0.0, le=4.0, description="Cumulative GPA prior to this semester (0.0 to 4.0 scale)")
    course_credits_enrolled: float = Field(..., ge=6.0, le=30.0, description="Total academic credits enrolled in current semester")
    study_group_participation: int = Field(..., ge=0, le=1, description="Participation in organized peer study groups (1=Yes, 0=No)")

    @field_validator('study_group_participation')
    @classmethod
    def validate_study_group(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError("study_group_participation must be 0 or 1")
        return v


class InfluentialSignal(BaseModel):
    feature_name: str
    feature_value: float
    importance_weight: float
    direction: str  # 'increases_risk' or 'decreases_risk'
    clinical_note: str


class PredictionResponse(BaseModel):
    """
    Defensible decision-support output contract.
    Presents advisory risk tier, probability, influential factors, and human oversight mandates.
    """
    student_id: str
    risk_probability: float = Field(..., ge=0.0, le=1.0, description="Calibrated risk probability of scoring < 50% in final outcome")
    advisory_band: str = Field(..., description="'Low Risk (On Track)', 'Moderate Risk (Advisory)', or 'High Risk (Priority Support)'")
    is_at_risk_flag: bool = Field(..., description="Binary thresholded decision support flag")
    influential_signals: List[InfluentialSignal]
    recommended_interventions: List[str]
    human_oversight_notice: str = (
        "DECISION SUPPORT ONLY: This model output is an early advisory signal to guide educator support. "
        "It MUST NOT be used for automated punishment, grading, academic probation, or admission denial. "
        "All academic interventions require qualitative review by a designated faculty advisor or counselor."
    )
    model_version: str
    prediction_timestamp: str


class BatchPredictionRequest(BaseModel):
    students: List[StudentSignals]


class BatchPredictionResponse(BaseModel):
    total_processed: int
    predictions: List[PredictionResponse]

