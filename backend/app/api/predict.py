"""
Nexsure — Prediction & Observability API Routes

Endpoints:
  GET  /api/health           — liveness probe
  GET  /api/training-status  — current training stage (for polling)
  GET  /api/system-info      — full model metadata + health state
  GET  /api/metrics          — per-model evaluation metrics
  GET  /api/model-insights   — feature importance + model comparison
  GET  /api/version-history  — retraining audit trail
  GET  /api/logs             — recent prediction logs
  POST /api/predict          — single-record risk assessment
  POST /api/train            — manual retrain trigger (for emergencies)
"""

import logging
import time
import traceback
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.data_loader import load_dataset
from app.core.evaluate import evaluate_models
from app.core.explain import explain_local_prediction, get_global_feature_importance
from app.core.preprocess import fit_preprocessing_pipeline, split_features_target
from app.core.train import (
    artifacts_exist,
    get_artifacts_folder,
    get_model_root_folder,
    load_feature_columns,
    load_metadata,
    load_model,
    load_pipeline,
    load_version_history,
    save_feature_columns,
    save_metadata,
    save_pipeline,
    append_version_history,
    train_and_select_best_model,
)

logger = logging.getLogger(__name__)
router = APIRouter()

MODEL_ROOT = get_model_root_folder()
ARTIFACTS_ROOT = get_artifacts_folder()

# ─── Global State ──────────────────────────────────────────────────────────────
# All state is encapsulated in a single dict to make it easy to inject
# during the startup lifecycle.

_STATE: Dict[str, Any] = {
    "pipeline": None,
    "trained_models": {},
    "feature_columns": None,
    "best_model_name": None,
    "metadata": {},
    "version_history": [],
    "X_test_transformed": None,
    "y_test": None,
    "X_test_df": None,
}

# Training status for frontend polling
_TRAINING_STATUS: Dict[str, str] = {
    "status": "idle",   # idle | training | ready | degraded
    "stage": "idle",    # initializing | preprocessing | training | evaluating | selecting | saving | idle
}

# Prediction logs (capped at 200 in memory)
_PREDICTION_LOGS: List[Dict[str, Any]] = []


# ─── State Management Helpers ─────────────────────────────────────────────────

def _set_global_state(
    pipeline, trained_models, feature_columns, best_model_name,
    metadata, version_history, X_test_transformed, y_test, X_test_df
) -> None:
    """Called by main.py startup to inject trained state."""
    _STATE["pipeline"] = pipeline
    _STATE["trained_models"] = trained_models or {}
    _STATE["feature_columns"] = feature_columns
    _STATE["best_model_name"] = best_model_name
    _STATE["metadata"] = metadata or {}
    _STATE["version_history"] = version_history or []
    _STATE["X_test_transformed"] = X_test_transformed
    _STATE["y_test"] = y_test
    _STATE["X_test_df"] = X_test_df


def _set_training_status(status: str, stage: str) -> None:
    _TRAINING_STATUS["status"] = status
    _TRAINING_STATUS["stage"] = stage


def _get_training_status() -> Dict[str, str]:
    return dict(_TRAINING_STATUS)


def _is_ready() -> bool:
    return (
        _STATE["pipeline"] is not None
        and bool(_STATE["trained_models"])
        and _STATE["feature_columns"] is not None
    )


def _derive_model_health() -> str:
    """Determine model health from current state."""
    status = _TRAINING_STATUS["status"]
    if status == "training":
        return "training"
    if status == "degraded":
        return "degraded"
    if _is_ready():
        return "healthy"
    return "unavailable"


# ─── Schemas ──────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    dataset_filename: str = Field("insurance3r2.csv", description="CSV filename in backend/data/")
    target_column: str = Field("insuranceclaim", description="Original target column in the dataset")
    test_size: float = Field(0.20, gt=0.0, lt=0.5)
    random_state: int = Field(42)


class TrainResponse(BaseModel):
    models_trained: Dict[str, str]
    best_model: str
    dataset_rows: int
    dataset_columns: int
    training_duration_s: Optional[float] = None
    inference_latency_ms: Optional[float] = None


class PredictRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    features: Dict[str, Any] = Field(..., description="6 patient features for risk assessment")
    model_name: str = Field("best_model", description="Model key to use for prediction")


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str
    prediction: str
    confidence: float
    confidence_tier: str   # Low | Medium | High
    label: str
    top_features: List[Dict[str, Any]]
    explanation: str
    feature_importance: Optional[Dict[str, float]] = None
    inference_latency_ms: Optional[float] = None
    shap: Optional[Dict[str, Any]] = None


class SystemInfoResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    model_health: str
    training_timestamp: Optional[str] = None
    dataset_size: Optional[int] = None
    feature_count: Optional[int] = None
    training_duration_s: Optional[float] = None
    inference_latency_ms: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    feature_columns: Optional[List[str]] = None
    trained_models: Optional[List[str]] = None


class TrainingStatusResponse(BaseModel):
    status: str
    stage: str


class ModelInsightsResponse(BaseModel):
    top_features: List[Dict[str, Any]]
    feature_importance: Optional[Dict[str, float]] = None
    all_model_metrics: Optional[Dict[str, Any]] = None


class EvaluateResponse(BaseModel):
    evaluations: Dict[str, Dict[str, Any]]


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.get("/health")
def health_check() -> dict:
    """Liveness probe — always returns OK if the process is running."""
    return {
        "status": "ok",
        "model_health": _derive_model_health(),
        "engine": "Nexsure v2.0",
    }


@router.get("/training-status", response_model=TrainingStatusResponse)
def get_training_status() -> TrainingStatusResponse:
    """
    Lightweight polling endpoint for the frontend training visualization.
    Returns current training status and stage.
    """
    return TrainingStatusResponse(**_TRAINING_STATUS)


@router.get("/system-info", response_model=SystemInfoResponse)
def get_system_info() -> SystemInfoResponse:
    """
    Full system observability — all metadata, health state, and ML metrics
    in a single call. Powers the AI Observatory panel on the frontend.
    """
    meta = _STATE.get("metadata", {})
    health = _derive_model_health()
    feature_columns = _STATE.get("feature_columns")
    trained_models_keys = list(_STATE.get("trained_models", {}).keys())

    return SystemInfoResponse(
        model_name=meta.get("best_model_name") or _STATE.get("best_model_name"),
        model_version=meta.get("model_version"),
        model_health=health,
        training_timestamp=meta.get("training_timestamp"),
        dataset_size=meta.get("dataset_size"),
        feature_count=meta.get("feature_count") or (len(feature_columns) if feature_columns else None),
        training_duration_s=meta.get("training_duration_s"),
        inference_latency_ms=meta.get("inference_latency_ms"),
        accuracy=meta.get("accuracy"),
        precision=meta.get("precision"),
        recall=meta.get("recall"),
        f1_score=meta.get("f1_score"),
        roc_auc=meta.get("roc_auc"),
        confusion_matrix=meta.get("confusion_matrix"),
        feature_columns=feature_columns,
        trained_models=trained_models_keys,
    )


@router.get("/metrics", response_model=EvaluateResponse)
def get_model_metrics() -> EvaluateResponse:
    """Return evaluation metrics for all trained models on the held-out test set."""
    if not _is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not ready. System is initializing or training.",
        )

    meta = _STATE.get("metadata", {})
    all_metrics = meta.get("all_model_metrics")

    # If we have pre-computed metrics in metadata, use those
    if all_metrics:
        return EvaluateResponse(evaluations=all_metrics)

    # Otherwise compute live (requires test set in memory)
    X_test_t = _STATE.get("X_test_transformed")
    y_test = _STATE.get("y_test")
    if X_test_t is None or y_test is None:
        raise HTTPException(
            status_code=503,
            detail="Test set not available in memory. Restart the backend to re-initialize.",
        )
    try:
        evaluations = evaluate_models(_STATE["trained_models"], X_test_t, y_test)
        return EvaluateResponse(evaluations=evaluations)
    except Exception as exc:
        logger.error("Error evaluating models: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to evaluate models.")


@router.get("/model-insights", response_model=ModelInsightsResponse)
def get_model_insights() -> ModelInsightsResponse:
    """
    Feature importance from the best model and comparison metrics for all models.
    Automatically chooses the correct method: feature_importances_ for RF,
    coefficients for LogisticRegression.
    """
    if not _is_ready():
        raise HTTPException(status_code=503, detail="Model not ready.")

    X_test_df = _STATE.get("X_test_df")
    best_model_name = _STATE.get("best_model_name")
    trained_models = _STATE.get("trained_models", {})
    meta = _STATE.get("metadata", {})

    if X_test_df is None or best_model_name is None:
        raise HTTPException(
            status_code=503,
            detail="Insight data not available in memory. Restart to re-initialize.",
        )

    try:
        best_model = trained_models.get(best_model_name) or trained_models.get("best_model")
        pipeline = _STATE.get("pipeline")
        top_features = get_global_feature_importance(best_model, pipeline, X_test_df)
        feature_importance = {
            item["feature"]: float(item.get("importance", 0.0))
            for item in top_features
            if isinstance(item, dict) and "feature" in item
        }
        return ModelInsightsResponse(
            top_features=top_features,
            feature_importance=feature_importance,
            all_model_metrics=meta.get("all_model_metrics"),
        )
    except Exception as exc:
        logger.error("Error generating model insights: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to generate model insights.")


@router.get("/version-history")
def get_version_history() -> List[Dict[str, Any]]:
    """Return the full model version history (retraining audit trail)."""
    return _STATE.get("version_history", [])


@router.get("/logs")
def get_prediction_logs() -> List[Dict[str, Any]]:
    """Return the last 50 prediction log entries."""
    return _PREDICTION_LOGS[-50:]


@router.post("/predict", response_model=PredictResponse)
def predict_claim(request: PredictRequest) -> PredictResponse:
    """
    Generate a risk assessment for a single patient record with SHAP explanations.
    
    Accepts 6 features: age, sex, bmi, children, smoker, region.
    Returns: verdict, confidence, confidence tier, and explainability breakdown.
    """
    if not _is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not ready. The system is initializing. Please retry in a moment.",
        )

    pipeline = _STATE["pipeline"]
    trained_models = _STATE["trained_models"]
    feature_columns = _STATE["feature_columns"]
    best_model_name = _STATE.get("best_model_name", "best_model")

    # Resolve model — allow "best_model" as alias
    model_key = request.model_name
    if model_key not in trained_models:
        if model_key == "best_model" and best_model_name in trained_models:
            model_key = best_model_name
        else:
            available = list(trained_models.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model '{request.model_name}'. Available: {available}",
            )

    features = request.features
    if not features:
        raise HTTPException(status_code=400, detail="No features provided.")

    missing = [col for col in feature_columns if col not in features]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {missing}. Required: {feature_columns}",
        )

    try:
        input_df = pd.DataFrame([features])

        # ── Encode categorical features to match training schema ───────────────
        if "sex" in input_df.columns:
            input_df["sex"] = input_df["sex"].map({"male": 1, "female": 0}).fillna(0)

        if "smoker" in input_df.columns:
            input_df["smoker"] = input_df["smoker"].map({"yes": 1, "no": 0}).fillna(0)

        if "region" in input_df.columns:
            input_df["region"] = input_df["region"].map(
                {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
            ).fillna(0)

        # ── Fill any missing columns with zero ────────────────────────────────
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[feature_columns]

        # Coerce numeric types
        for col in ["age", "bmi", "children"]:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0)

        # ── Transform & Predict with latency tracking ─────────────────────────
        lat_start = time.perf_counter()
        transformed = pipeline.transform(input_df)
        model = trained_models[model_key]
        proba = model.predict_proba(transformed)[0]

        # ── Determine approval probability ────────────────────────────────────
        try:
            approved_index = list(model.classes_).index(1)
            approval_prob = float(proba[approved_index])
        except (ValueError, IndexError):
            approval_prob = 0.0

        # Decision threshold: 0.60 for Approved
        prediction = "APPROVED" if approval_prob >= 0.60 else "REJECTED"
        confidence = round(approval_prob * 100, 2)

        # ── Confidence tier ───────────────────────────────────────────────────
        if confidence >= 95:
            confidence_tier = "Very High Confidence"
        elif confidence >= 80:
            confidence_tier = "High Confidence"
        elif confidence >= 60:
            confidence_tier = "Moderate Confidence"
        else:
            confidence_tier = "Low Confidence"

        # ── SHAP / Feature explanations ───────────────────────────────────────
        feature_contributions, _ = explain_local_prediction(model, pipeline, input_df)
        
        top_features = feature_contributions[:5]
        
        pos_drivers = [fc for fc in feature_contributions if fc["shap_value"] > 0]
        neg_drivers = [fc for fc in feature_contributions if fc["shap_value"] <= 0]
        
        verdict_word = "approved" if prediction == "APPROVED" else "rejected"
        if len(pos_drivers) > 0:
            primary = pos_drivers[0]["feature"]
            executive_summary = f"Underwriting assessment {verdict_word} primarily driven by {primary}."
        else:
            executive_summary = f"Underwriting assessment {verdict_word} based on aggregated risk factors."
        
        shap_payload = {
            "top_positive_drivers": pos_drivers,
            "top_negative_drivers": neg_drivers,
            "global_importance": [],
            "feature_summary": top_features,
            "executive_summary": executive_summary
        }

        lat_end = time.perf_counter()
        inference_latency_ms = round((lat_end - lat_start) * 1000, 2)

        # ── Update Real-Time Telemetry ────────────────────────────────────────
        if _STATE.get("metadata") is not None:
            current_avg = _STATE["metadata"].get("inference_latency_ms")
            if current_avg is None:
                new_avg = inference_latency_ms
            else:
                # Exponential moving average (alpha=0.2) for smooth real-time tracking
                new_avg = round((0.8 * current_avg) + (0.2 * inference_latency_ms), 2)
            _STATE["metadata"]["inference_latency_ms"] = new_avg
            try:
                save_metadata(_STATE["metadata"])
            except Exception as e:
                logger.error("Failed to save live telemetry: %s", e)

        # Build feature importance dict
        feature_importance = {
            item["feature"]: float(abs(item.get("shap_value", 0.0)))
            for item in top_features
            if isinstance(item, dict) and "feature" in item
        }

        # ── Dynamic explanation text ──────────────────────────────────────────
        explanation = executive_summary

        # ── Log prediction ────────────────────────────────────────────────────
        _PREDICTION_LOGS.append({
            "id": len(_PREDICTION_LOGS) + 1,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "age": int(features.get("age", 0)),
            "smoker": features.get("smoker", "unknown"),
            "result": prediction,
            "confidence": confidence,
            "confidence_tier": confidence_tier,
            "model_used": model_key,
            "inference_latency_ms": inference_latency_ms,
        })
        # Cap in-memory logs
        if len(_PREDICTION_LOGS) > 200:
            _PREDICTION_LOGS.pop(0)

        return PredictResponse(
            model_name=model_key,
            prediction=prediction,
            confidence=confidence,
            confidence_tier=confidence_tier,
            label=prediction,
            top_features=top_features,
            explanation=explanation,
            feature_importance=feature_importance,
            inference_latency_ms=inference_latency_ms,
            shap=shap_payload,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/train", response_model=TrainResponse)
def trigger_manual_train(request: TrainRequest) -> TrainResponse:
    """
    Emergency manual retrain endpoint.
    Prefer auto-training on startup — this is for operational recovery only.
    """
    from datetime import datetime, timezone
    from app.core.train import (
        save_feature_columns,
        save_metadata,
        save_pipeline,
        append_version_history,
        load_version_history,
    )

    _set_training_status("training", "initializing")

    try:
        df = load_dataset(request.dataset_filename, request.target_column)
        threshold = df["charges"].median()
        df["target"] = (df["charges"] < threshold).astype(int)

        cols_to_drop = ["charges", "steps"]
        if request.target_column in df.columns and request.target_column != "target":
            cols_to_drop.append(request.target_column)
        if "insuranceclaim" in df.columns and "insuranceclaim" not in cols_to_drop:
            cols_to_drop.append("insuranceclaim")
        df = df.drop(columns=cols_to_drop, errors="ignore")

        _set_training_status("training", "preprocessing")
        X_train, X_test, y_train, y_test = split_features_target(
            df, target_column="target",
            test_size=request.test_size,
            random_state=request.random_state,
        )
        feature_columns = X_train.columns.tolist()
        pipeline, X_train_t, X_test_t = fit_preprocessing_pipeline(X_train, X_test)

        _set_training_status("training", "training")
        train_result = train_and_select_best_model(
            X_train_t, X_test_t, y_train, y_test, random_state=request.random_state
        )
        trained_models = train_result["models"]
        all_metrics = train_result["metrics"]
        best_model_name = train_result["best_model_name"]
        inference_latency_ms = train_result["inference_latency_ms"]
        training_duration_s = train_result["training_duration_s"]

        _set_training_status("training", "saving")
        save_pipeline(pipeline)
        save_feature_columns(feature_columns)

        now_iso = datetime.now(timezone.utc).isoformat()
        history = load_version_history()
        version_tag = f"v{len(history) + 1}.0"
        best_metrics = all_metrics[best_model_name]

        full_metadata = {
            "best_model_name": best_model_name,
            "feature_columns": feature_columns,
            "trained_models": list(trained_models.keys()),
            "model_version": version_tag,
            "training_timestamp": now_iso,
            "dataset_size": df.shape[0],
            "feature_count": len(feature_columns),
            "training_duration_s": training_duration_s,
            "inference_latency_ms": inference_latency_ms,
            "model_health": "healthy",
            "accuracy": best_metrics.get("accuracy"),
            "precision": best_metrics.get("precision"),
            "recall": best_metrics.get("recall"),
            "f1_score": best_metrics.get("f1_score"),
            "roc_auc": best_metrics.get("roc_auc"),
            "confusion_matrix": best_metrics.get("confusion_matrix"),
            "all_model_metrics": all_metrics,
        }
        save_metadata(full_metadata)
        append_version_history({
            "version": version_tag,
            "timestamp": now_iso,
            "model": best_model_name,
            "accuracy": best_metrics.get("accuracy"),
            "f1_score": best_metrics.get("f1_score"),
            "dataset_size": df.shape[0],
            "training_duration_s": training_duration_s,
        })

        trained_models["best_model"] = trained_models.get(best_model_name)
        version_history = load_version_history()

        _set_global_state(
            pipeline=pipeline,
            trained_models=trained_models,
            feature_columns=feature_columns,
            best_model_name=best_model_name,
            metadata=full_metadata,
            version_history=version_history,
            X_test_transformed=X_test_t,
            y_test=y_test,
            X_test_df=X_test,
        )
        _set_training_status("ready", "idle")

        return TrainResponse(
            models_trained={name: "trained" for name in trained_models},
            best_model=best_model_name,
            dataset_rows=df.shape[0],
            dataset_columns=df.shape[1],
            training_duration_s=training_duration_s,
            inference_latency_ms=inference_latency_ms,
        )

    except FileNotFoundError as exc:
        _set_training_status("degraded", "idle")
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        _set_training_status("degraded", "idle")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _set_training_status("degraded", "idle")
        logger.error("Manual training failed: %s", exc)
        raise HTTPException(status_code=500, detail="Training failed. Check server logs.")


@router.get("/train-status")
def get_train_status_legacy() -> dict:
    """Legacy endpoint kept for backward compatibility."""
    meta = _STATE.get("metadata", {})
    best = _STATE.get("best_model_name") or meta.get("best_model_name", "Unavailable")
    accuracy = meta.get("accuracy")
    return {"best_model": best, "accuracy": accuracy}


@router.get("/history")
def get_training_history() -> List[Dict[str, Any]]:
    """
    Return the real model version history (not fake epoch data).
    Each entry represents a retrain event with real metrics.
    """
    return _STATE.get("version_history", [])


@router.get("/explain")
def get_shap_explanation() -> dict:
    """Return global SHAP feature importance for the best trained model."""
    if not _is_ready():
        raise HTTPException(status_code=503, detail="Model not ready.")

    X_test_df = _STATE.get("X_test_df")
    best_model_name = _STATE.get("best_model_name")
    trained_models = _STATE.get("trained_models", {})
    pipeline = _STATE.get("pipeline")

    if X_test_df is None or best_model_name is None:
        raise HTTPException(status_code=503, detail="Insight data not available.")

    try:
        model = trained_models.get(best_model_name) or trained_models.get("best_model")
        top_features = get_global_feature_importance(model, pipeline, X_test_df)
        feature_importance = {
            item["feature"]: float(item.get("importance", 0.0))
            for item in top_features
            if isinstance(item, dict) and "feature" in item
        }
        return {"top_features": top_features, "feature_importance": feature_importance}
    except Exception as exc:
        logger.error("SHAP explanation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to generate explanations.")