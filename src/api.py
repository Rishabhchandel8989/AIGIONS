"""
FastAPI REST API Service for Student Performance & At-Risk Prediction.
Exposes real serialized model inference, SHAP explanations, batch scoring, and health checks.
Strictly adheres to decision-support ethical guardrails and Pydantic validation contracts.
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.schemas import (
    StudentSignals,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    InfluentialSignal
)
from src.explainability import ModelExplainer
import json
import os

# Global model explainer instance (loads saved artifacts)
explainer: ModelExplainer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global explainer
    explainer = ModelExplainer(artifacts_dir="artifacts")
    print("Model and Explainer successfully initialized into memory.")
    yield


app = FastAPI(
    title="Student Performance & At-Risk Advisory Service",
    description=(
        "Decision-support Machine Learning API for early identification of students who may benefit "
        "from proactive academic support. Strictly non-punitive and FERPA/GDPR aligned."
    ),
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["System"])
def root():
    return {
        "service": "Algions Student Performance & At-Risk Advisory API",
        "version": "1.0.0",
        "status": "online",
        "ethical_mandate": (
            "Advisory decision-support only. Strictly prohibited from triggering automated punishment, "
            "grading alterations, academic probation, or admission denial."
        ),
        "docs_url": "/docs"
    }


@app.get("/health", tags=["System"])
def health_check():
    if explainer is None or explainer.model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model artifact not loaded")
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": True,
        "champion_model": "LightGBM Classifier",
        "prediction_horizon": "Week 16 final outcome from Week 8 signals"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_student_risk(signals: StudentSignals):
    """
    Evaluates individual student mid-term learning signals and generates real, calibrated risk advisory.
    """
    if explainer is None:
        raise HTTPException(status_code=503, detail="Model explainer uninitialized")

    student_dict = signals.model_dump()

    try:
        prob, band, is_at_risk, influential_signals, interventions = explainer.explain_instance(student_dict)

        return PredictionResponse(
            student_id=signals.student_id or "STU-UNKNOWN",
            risk_probability=round(prob, 4),
            advisory_band=band,
            is_at_risk_flag=is_at_risk,
            influential_signals=influential_signals,
            recommended_interventions=interventions,
            model_version="1.0.0-LightGBM",
            prediction_timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference computation error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch_students(request: BatchPredictionRequest):
    """
    Batch scoring endpoint for advising entire cohorts or departments.
    """
    if explainer is None:
        raise HTTPException(status_code=503, detail="Model explainer uninitialized")

    predictions = []
    for signals in request.students:
        student_dict = signals.model_dump()
        prob, band, is_at_risk, influential_signals, interventions = explainer.explain_instance(student_dict)

        predictions.append(PredictionResponse(
            student_id=signals.student_id or "STU-UNKNOWN",
            risk_probability=round(prob, 4),
            advisory_band=band,
            is_at_risk_flag=is_at_risk,
            influential_signals=influential_signals,
            recommended_interventions=interventions,
            model_version="1.0.0-LightGBM",
            prediction_timestamp=datetime.utcnow().isoformat()
        ))

    return BatchPredictionResponse(
        total_processed=len(predictions),
        predictions=predictions
    )


@app.get("/metrics", tags=["Audit & Governance"])
def get_model_metrics():
    """
    Returns held-out test evaluation benchmarks across baseline and candidate models.
    """
    metrics_path = os.path.join("artifacts", "metrics_summary.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Metrics summary not found")
    with open(metrics_path, "r") as f:
        data = json.load(f)
    return data
