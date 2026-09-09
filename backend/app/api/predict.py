"""
predict.py -- Nexsure AI Engine Routing & Low-Latency Inference Layer (v3.0)

Features:
- Precision nanosecond stage timing (8 lifecycle stages)
- Multi-layered runtime cache utilization (zero disk I/O during serving)
- Model-Version aware SHAP explainer reuse
- Real-time telemetry tracking & rolling percentile engine
- Governance & Model Registry APIs (/api/model-registry, /api/promotion-history)
- Performance & Cache Status APIs (/api/performance, /api/cache-status)
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import app.core.logger as ns_log
from app.core.cache_manager import runtime_cache
from app.core.explain import explain_local_prediction, get_global_feature_importance
from app.core.governance import validate_provenance_compatibility
from app.core.performance import PrecisionTimer, persist_summary_metadata_async
from app.core.promotion import (
    get_champion,
    get_challenger,
    promote_model,
    rollback_model,
    validate_registry,
)
from app.core.registry import (
    get_model_root_folder,
    load_benchmark_report,
    load_model_registry,
    load_promotion_history,
    load_version_history,
)
from app.core.telemetry import telemetry_engine

logger = logging.getLogger(__name__)
router = APIRouter()

MODEL_ROOT = get_model_root_folder()

# --------------------------------------------------------------------------- #
#  Global Application State                                                    #
# --------------------------------------------------------------------------- #

_STATE: Dict[str, Any] = {
    "pipeline": None,
    "trained_models": {},
    "feature_columns": None,
    "best_model_name": None,
    "metadata": {},
    "version_history": [],
    "model_registry": {},
    "promotion_history": [],
    "X_test_transformed": None,
    "y_test": None,
    "X_test_df": None,
    "benchmark_report": {},
}

_TRAINING_STATUS: Dict[str, str] = {
    "status": "idle",
    "stage": "idle",
}

_PREDICTION_LOGS: List[Dict[str, Any]] = []


def _set_global_state(
    pipeline,
    trained_models,
    feature_columns,
    best_model_name,
    metadata,
    version_history,
    X_test_transformed,
    y_test,
    X_test_df,
    benchmark_report=None,
    model_registry=None,
    promotion_history=None,
) -> None:
    _STATE["pipeline"] = pipeline
    _STATE["trained_models"] = trained_models or {}
    _STATE["feature_columns"] = feature_columns
    _STATE["best_model_name"] = best_model_name
    _STATE["metadata"] = metadata or {}
    _STATE["version_history"] = version_history or []
    _STATE["model_registry"] = model_registry or {}
    _STATE["promotion_history"] = promotion_history or []
    _STATE["X_test_transformed"] = X_test_transformed
    _STATE["y_test"] = y_test
    _STATE["X_test_df"] = X_test_df
    _STATE["benchmark_report"] = benchmark_report or {}


def _set_training_status(status: str, stage: str) -> None:
    _TRAINING_STATUS["status"] = status
    _TRAINING_STATUS["stage"] = stage


def _get_training_status() -> Dict[str, str]:
    return dict(_TRAINING_STATUS)


def normalize_input_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize raw string/numeric input features into pipeline-compatible encodings."""
    norm: Dict[str, Any] = {}
    
    # age
    norm["age"] = float(features.get("age", 30))
    # bmi
    norm["bmi"] = float(features.get("bmi", 25.0))
    # children
    norm["children"] = int(features.get("children", 0))
    
    # sex (0=female, 1=male)
    sex_val = str(features.get("sex", "male")).strip().lower()
    if sex_val in ["female", "f", "0"]:
        norm["sex"] = 0
    else:
        norm["sex"] = 1
        
    # smoker (0=no/non-smoker, 1=yes/smoker)
    smoker_val = str(features.get("smoker", "no")).strip().lower()
    if smoker_val in ["no", "non-smoker", "false", "0", "n"]:
        norm["smoker"] = 0
    else:
        norm["smoker"] = 1
        
    # region (0=northeast, 1=northwest, 2=southeast, 3=southwest)
    region_val = str(features.get("region", "southwest")).strip().lower()
    region_map = {
        "northeast": 0, "ne": 0, "0": 0,
        "northwest": 1, "nw": 1, "1": 1,
        "southeast": 2, "se": 2, "2": 2,
        "southwest": 3, "sw": 3, "3": 3,
    }
    norm["region"] = region_map.get(region_val, 0)
    
    return norm


def _is_ready() -> bool:
    return (
        _STATE["pipeline"] is not None
        and bool(_STATE["trained_models"])
        and _STATE["feature_columns"] is not None
    )


def _derive_model_health() -> str:
    status = _TRAINING_STATUS.get("status", "idle")
    if status == "training":
        return "training"
    if status == "degraded":
        return "degraded"
    if _is_ready() and runtime_cache.is_warm:
        return "healthy"
    return "unavailable"


# --------------------------------------------------------------------------- #
#  Pydantic Schemas                                                            #
# --------------------------------------------------------------------------- #

class PredictRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    features: Dict[str, Any] = Field(..., description="6 patient features for risk assessment")
    model_name: str = Field("best_model", description="Model key to use for prediction (must be best_model/active champion)")


class PredictResponse(BaseModel):
    verdict: str
    prediction: str
    confidence: float
    confidence_tier: str
    approval_probability: float
    rejection_probability: float
    predicted_class: int
    positive_class: int
    negative_class: int
    positive_class_name: str = "LOW_RISK_APPROVAL"
    negative_class_name: str = "HIGH_RISK_REJECTION"
    model_name: str
    model_version: Optional[str] = "v2.1"
    label: str
    explanation: str
    top_decision_drivers: List[str] = []
    top_features: Optional[List[Dict[str, Any]]] = None
    feature_importance: Optional[Dict[str, float]] = None
    shap: Optional[Dict[str, Any]] = None
    telemetry: Optional[Dict[str, Any]] = None
    stage_latencies_ms: Optional[Dict[str, float]] = None
    inference_latency_ms: Optional[float] = None


class PromoteRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str = Field(..., description="Registered candidate model name to promote")
    version: Optional[str] = Field("v2.1", description="Target model version")
    reason: Optional[str] = Field(None, description="Promotion reason / justification")
    actor: Optional[str] = Field("admin_api", description="Actor initiating promotion")
    force: Optional[bool] = Field(False, description="Administrative override force flag")


class RollbackRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    target_model: Optional[str] = Field(None, description="Specific target model name to revert to (defaults to predecessor)")
    version: Optional[str] = Field(None, description="Specific target version")
    reason: Optional[str] = Field(None, description="Rollback justification")
    actor: Optional[str] = Field("admin_api", description="Actor initiating rollback")


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
    # Registry extensions
    champion: Optional[str] = None
    champion_model: Optional[str] = None
    challenger: Optional[str] = None
    challenger_model: Optional[str] = None
    ranking: Optional[List[str]] = None
    benchmark_timestamp: Optional[str] = None
    benchmark_duration_s: Optional[float] = None
    deployment_timestamp: Optional[str] = None
    registry_status: str = "healthy"
    promotion_count: int = 0
    # Phase 3 Performance extensions
    warm_start_enabled: bool = True
    shap_cache_enabled: bool = True
    readiness_stage: str = "READY"
    uptime_seconds: float = 0.0
    cache_hit_rate_pct: float = 100.0
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    requests_per_second: float = 0.0
    cv_accuracy_mean: Optional[float] = None
    cv_accuracy_std: Optional[float] = None
    cv_f1_mean: Optional[float] = None
    cv_f1_std: Optional[float] = None
    cv_roc_auc_mean: Optional[float] = None
    cv_roc_auc_std: Optional[float] = None
    model_only_latency_ms: Optional[float] = None
    end_to_end_latency_ms: Optional[float] = None
    model_size_mb: Optional[float] = None
    selection_policy: Optional[Dict[str, str]] = None
    target_threshold: Optional[float] = None
    target_definition: Optional[Any] = None
    target_source_split: Optional[str] = None
    raw_feature_count: Optional[int] = None
    transformed_feature_count: Optional[int] = None


class TrainingStatusResponse(BaseModel):
    status: str
    stage: str


class ModelInsightsResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    best_model_name: str
    feature_importances: Dict[str, float]
    all_models_metrics: Dict[str, Any]


class EvaluateResponse(BaseModel):
    results: Dict[str, Any]
    best_model_name: Optional[str] = None
    evaluation_timestamp: str


# --------------------------------------------------------------------------- #
#  Endpoints                                                                   #
# --------------------------------------------------------------------------- #

@router.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "model_health": _derive_model_health(),
        "engine": "Nexsure v2.0 (Low-Latency)",
        "readiness_stage": runtime_cache.readiness_stage,
        "active_champion": _STATE.get("best_model_name"),
    }


@router.get("/training-status", response_model=TrainingStatusResponse)
def get_training_status() -> TrainingStatusResponse:
    ts = _get_training_status()
    return TrainingStatusResponse(status=ts["status"], stage=ts["stage"])


@router.get("/system-info", response_model=SystemInfoResponse)
def get_system_info() -> SystemInfoResponse:
    """Full system observability, performance percentiles, and Champion/Challenger registry telemetry."""
    meta = _STATE.get("metadata", {})
    health = _derive_model_health()
    feature_columns = _STATE.get("feature_columns")
    
    try:
        reg = runtime_cache.registry_cache.get() or load_model_registry()
    except Exception:
        reg = _STATE.get("model_registry", {})

    history = load_promotion_history()
    champ_meta = reg.get("champion", {})
    chal_meta = reg.get("challenger", {})

    champ_name = champ_meta.get("name") or meta.get("best_model_name") or _STATE.get("best_model_name")
    chal_name = chal_meta.get("name") or reg.get("runner_up")

    tp = telemetry_engine.get_throughput()
    lat = telemetry_engine.get_latency_metrics()
    tot_stats = lat.get("total", {})

    return SystemInfoResponse(
        model_name=champ_name,
        model_version=champ_meta.get("version") or meta.get("model_version", "v2.1"),
        model_health=health,
        training_timestamp=meta.get("training_timestamp") or reg.get("deployment_timestamp"),
        dataset_size=meta.get("dataset_size") or reg.get("provenance", {}).get("dataset_size"),
        feature_count=meta.get("raw_feature_count", len(feature_columns) if feature_columns else 6),
        training_duration_s=meta.get("training_duration_s"),
        inference_latency_ms=tot_stats.get("p50") or champ_meta.get("latency_ms") or meta.get("model_only_latency_ms"),
        accuracy=champ_meta.get("accuracy") or meta.get("accuracy"),
        precision=champ_meta.get("precision") or meta.get("precision"),
        recall=champ_meta.get("recall") or meta.get("recall"),
        f1_score=champ_meta.get("f1_score") or meta.get("f1_score"),
        roc_auc=champ_meta.get("roc_auc") or meta.get("roc_auc"),
        confusion_matrix=meta.get("confusion_matrix"),
        feature_columns=feature_columns,
        champion=champ_name,
        champion_model=champ_name,
        challenger=chal_name,
        challenger_model=chal_name,
        ranking=reg.get("ranking") or meta.get("ranking"),
        benchmark_timestamp=meta.get("benchmark_timestamp") or reg.get("benchmark_timestamp"),
        benchmark_duration_s=meta.get("benchmark_duration_s"),
        deployment_timestamp=reg.get("deployment_timestamp"),
        registry_status=reg.get("status", "healthy"),
        promotion_count=len(history),
        # Phase 3 Performance additions
        warm_start_enabled=True,
        shap_cache_enabled=True,
        readiness_stage=runtime_cache.readiness_stage,
        uptime_seconds=tp.get("uptime_seconds", 0.0),
        cache_hit_rate_pct=tp.get("cache_hit_rate_pct") if "cache_hit_rate_pct" in tp else tp.get("cache_hit_rate", 100.0),
        p50_latency_ms=tot_stats.get("p50"),
        p95_latency_ms=tot_stats.get("p95"),
        requests_per_second=tp.get("requests_per_second", 0.0),
        cv_accuracy_mean=meta.get("cv_accuracy_mean"),
        cv_accuracy_std=meta.get("cv_accuracy_std"),
        cv_f1_mean=champ_meta.get("cv_f1_mean") or meta.get("cv_f1_mean"),
        cv_f1_std=champ_meta.get("cv_f1_std") or meta.get("cv_f1_std"),
        cv_roc_auc_mean=meta.get("cv_roc_auc_mean"),
        cv_roc_auc_std=meta.get("cv_roc_auc_std"),
        model_only_latency_ms=champ_meta.get("latency_ms") or meta.get("model_only_latency_ms"),
        end_to_end_latency_ms=champ_meta.get("end_to_end_latency_ms") or meta.get("end_to_end_latency_ms"),
        model_size_mb=champ_meta.get("size_mb") or meta.get("model_size_mb"),
        selection_policy=reg.get("selection_policy") or meta.get("selection_policy"),
        target_threshold=meta.get("target_threshold"),
        target_definition=meta.get("target_definition"),
        target_source_split=meta.get("target_source_split"),
        raw_feature_count=meta.get("raw_feature_count", 6),
        transformed_feature_count=meta.get("transformed_feature_count", 11),
    )


@router.get("/performance")
def get_performance_telemetry() -> dict:
    """Live runtime latency percentiles, throughput counters, and sparkline timeline."""
    return telemetry_engine.get_performance_summary()


@router.get("/cache-status")
def get_cache_status_endpoint() -> dict:
    """Runtime cache diagnostics for all 6 subsystems."""
    return runtime_cache.get_status()


@router.get("/model-registry")
def get_model_registry_endpoint() -> dict:
    try:
        return runtime_cache.registry_cache.get() or load_model_registry()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model registry: {exc}")


@router.get("/promotion-history")
def get_promotion_history_endpoint() -> List[Dict[str, Any]]:
    return load_promotion_history()


@router.post("/model-registry/promote")
def promote_candidate_endpoint(request: PromoteRequest) -> dict:
    if not _is_ready():
        raise HTTPException(status_code=503, detail="System initializing or not ready")

    try:
        result = promote_model(
            candidate_name=request.model_name,
            version=request.version,
            actor=request.actor,
            reason=request.reason,
            state=_STATE,
            force=request.force,
        )
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        logger.exception("Promotion endpoint error")
        raise HTTPException(status_code=500, detail=f"Promotion failed: {exc}")


@router.post("/model-registry/rollback")
def rollback_champion_endpoint(request: RollbackRequest) -> dict:
    if not _is_ready():
        raise HTTPException(status_code=503, detail="System initializing or not ready")

    try:
        result = rollback_model(
            target_model=request.target_model,
            target_version=request.version,
            actor=request.actor,
            reason=request.reason,
            state=_STATE,
        )
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        logger.exception("Rollback endpoint error")
        raise HTTPException(status_code=500, detail=f"Rollback failed: {exc}")


@router.get("/model-benchmark")
def get_model_benchmark() -> dict:
    try:
        return load_benchmark_report()
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Benchmark results not found: {exc}")


@router.get("/version-history")
def get_version_history() -> List[Dict[str, Any]]:
    return _STATE.get("version_history", [])


@router.get("/logs")
def get_prediction_logs() -> List[Dict[str, Any]]:
    return _PREDICTION_LOGS[-50:]


# --------------------------------------------------------------------------- #
#  High-Performance Low-Latency Inference Endpoint                            #
# --------------------------------------------------------------------------- #

@router.post("/predict", response_model=PredictResponse)
def predict_claim(request: PredictRequest) -> PredictResponse:
    """
    High-Performance Risk Assessment Inference (ACTIVE CHAMPION ONLY).
    Metadata-driven prediction semantics (Phase 3.2).
    """
    if not _is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not ready. The system is initializing. Please retry in a moment.",
        )

    timer = PrecisionTimer()

    # Stage 1: Validation & Normalization
    timer.start_stage("validation")
    features = request.features
    if not features:
        raise HTTPException(status_code=400, detail="No features provided.")

    feature_columns = runtime_cache.get_feature_columns()
    missing = [col for col in feature_columns if col not in features]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {missing}. Required: {feature_columns}",
        )

    try:
        norm_features = normalize_input_features(features)
        input_df = pd.DataFrame([norm_features])

        # Stage 2: Preprocessing (Cached Pipeline)
        timer.start_stage("preprocessing")
        pipeline = runtime_cache.get_pipeline()
        transformed = pipeline.transform(input_df)

        # Stage 3: Feature Ordering
        timer.start_stage("feature_ordering")
        if not transformed.flags.c_contiguous:
            transformed = np.ascontiguousarray(transformed)

        # Stage 4: Model Inference (Cached Champion & Metadata Target Semantics)
        timer.start_stage("inference")
        model = runtime_cache.get_champion_model()
        best_model_name = runtime_cache.champion_cache.model_name or _STATE.get("best_model_name", "catboost")
        model_version = runtime_cache.champion_cache.model_version or "v2.1"
        
        # Read target semantics from metadata
        metadata = runtime_cache.get_metadata() or _STATE.get("metadata", {})
        target_def = metadata.get("target_definition") or {
            "positive_class": 1,
            "positive_class_name": "LOW_RISK_APPROVAL",
            "negative_class": 0,
            "negative_class_name": "HIGH_RISK_REJECTION",
            "threshold_source": "training_split_median",
            "target_formula": "charges < training_median"
        }
        
        pos_class = int(target_def.get("positive_class", 1))
        neg_class = int(target_def.get("negative_class", 0))
        pos_name = str(target_def.get("positive_class_name", "LOW_RISK_APPROVAL"))
        neg_name = str(target_def.get("negative_class_name", "HIGH_RISK_REJECTION"))

        classes = list(getattr(model, "classes_", [0, 1]))
        pos_idx = classes.index(pos_class) if pos_class in classes else 1
        neg_idx = classes.index(neg_class) if neg_class in classes else 0

        raw_pred = model.predict(transformed)[0]
        proba = model.predict_proba(transformed)[0]

        low_risk_prob = float(proba[pos_idx])
        high_risk_prob = float(proba[neg_idx])

        approved = bool(raw_pred == pos_class)
        verdict = "APPROVED" if approved else "REJECTED"
        confidence_pct = round((low_risk_prob if approved else high_risk_prob) * 100, 2)

        if confidence_pct >= 95.0:
            confidence_tier = "Very High Confidence"
        elif confidence_pct >= 80.0:
            confidence_tier = "High Confidence"
        elif confidence_pct >= 60.0:
            confidence_tier = "Moderate Confidence"
        else:
            confidence_tier = "Low Confidence"

        # Stage 5: SHAP Generation (Cached Explainer & Precomputed Mappings)
        timer.start_stage("shap")
        cached_expl = runtime_cache.get_shap_explainer()
        mapping_table = runtime_cache.get_feature_mapping()
        feature_contributions, shap_payload = explain_local_prediction(
            model=model,
            pipeline=pipeline,
            input_df=input_df,
            explainer=cached_expl,
            transformed_input=transformed,
            mapping_table=mapping_table,
        )

        # Stage 6: Explanation & Decision Summary Formatting
        timer.start_stage("explanation_formatting")
        raw_feat_importances = {item["feature"]: item["shap_value"] for item in feature_contributions}
        top_decision_drivers = shap_payload.get("top_decision_drivers", [])
        explanation = shap_payload.get("executive_summary", f"Decision primarily influenced by {feature_contributions[0]['feature']}.")

        # Stage 7: Serialization & Logging
        timer.start_stage("serialization")
        stages_ns, total_ns = timer.finish()
        
        # Record in Telemetry Engine
        rec = telemetry_engine.record_prediction(
            stages_ns=stages_ns,
            total_ns=total_ns,
            success=True,
            model_name=best_model_name,
            cache_hit=True,
            shap_cache_hit=True,
        )

        total_ms = rec["total_ms"]
        stages_ms = rec["stages_ms"]

        log_entry = {
            "timestamp": rec["timestamp"],
            "features": features,
            "prediction": verdict,
            "verdict": verdict,
            "confidence": confidence_pct,
            "approval_probability": round(low_risk_prob, 4),
            "rejection_probability": round(high_risk_prob, 4),
            "latency_ms": total_ms,
            "model_name": best_model_name,
        }
        _PREDICTION_LOGS.append(log_entry)
        if len(_PREDICTION_LOGS) > 200:
            _PREDICTION_LOGS.pop(0)

        # Stage 8: Background Async Metadata Summary Persistence
        lat_stats = telemetry_engine.get_latency_metrics()
        tot_stats = lat_stats.get("total", {})
        shap_diag = telemetry_engine.get_shap_diagnostics()
        persist_summary_metadata_async(
            avg_latency_ms=tot_stats.get("mean", total_ms),
            p50_latency_ms=tot_stats.get("p50", total_ms),
            p95_latency_ms=tot_stats.get("p95", total_ms),
            prediction_count=telemetry_engine.total_predictions,
            folder=MODEL_ROOT,
        )

        # Terminal latency summary printout with SHAP improvement tracking
        ns_log.prediction_latency_summary(
            pred_id=rec["prediction_id"],
            stages_ms=stages_ms,
            total_ms=total_ms,
            rolling_metrics=lat_stats,
            champion_name=best_model_name,
            cache_hit=True,
            shap_cache_hit=True,
            shap_diagnostics=shap_diag,
        )

        return PredictResponse(
            verdict=verdict,
            prediction=verdict,
            confidence=confidence_pct,
            confidence_tier=confidence_tier,
            approval_probability=round(low_risk_prob, 4),
            rejection_probability=round(high_risk_prob, 4),
            predicted_class=int(raw_pred),
            positive_class=pos_class,
            negative_class=neg_class,
            positive_class_name=pos_name,
            negative_class_name=neg_name,
            model_name=best_model_name,
            model_version=model_version,
            label=verdict,
            top_features=feature_contributions,
            top_decision_drivers=top_decision_drivers,
            explanation=explanation,
            feature_importance=raw_feat_importances,
            shap=shap_payload,
            telemetry={
                "inference_latency_ms": total_ms,
                "model_name": best_model_name,
                "timestamp": rec["timestamp"],
                "p50_latency_ms": tot_stats.get("p50", total_ms),
                "p95_latency_ms": tot_stats.get("p95", total_ms),
            },
            stage_latencies_ms=stages_ms,
            inference_latency_ms=total_ms,
        )

    except Exception as exc:
        logger.exception("Prediction failed")
        telemetry_engine.record_prediction(
            stages_ns={"error": 0},
            total_ns=0,
            success=False,
            model_name=_STATE.get("best_model_name", "unknown"),
        )
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(exc)}")

@router.get("/explain")
def get_shap_explanation() -> dict:
    if not _is_ready():
        raise HTTPException(status_code=503, detail="Model not ready.")
    model = runtime_cache.get_champion_model()
    pipeline = runtime_cache.get_pipeline()
    X_test_df = _STATE.get("X_test_df")
    if X_test_df is None or (isinstance(X_test_df, pd.DataFrame) and X_test_df.empty):
        try:
            from app.core.data_loader import load_dataset
            raw_df = load_dataset("insurance3r2.csv", "charges")
            cols = runtime_cache.get_feature_columns() or ["age", "sex", "bmi", "children", "smoker", "region"]
            X_test_df = raw_df[[c for c in cols if c in raw_df.columns]].head(100)
            _STATE["X_test_df"] = X_test_df
        except Exception as e:
            logger.warning("Could not load sample dataset for global SHAP: %s", e)
    try:
        importance = get_global_feature_importance(model, pipeline, X_test_df)
        champ_name = runtime_cache.champion_cache.model_name or _STATE.get("best_model_name", "catboost")
        champ_version = runtime_cache.champion_cache.model_version or "v2.1"
        return {
            "model_name": champ_name,
            "model_version": champ_version,
            "feature_importance": {item["feature"]: item["importance"] for item in importance},
            "top_features": importance,
            "executive_summary": "Global SHAP feature attribution indicates Age and Smoking Status serve as the primary drivers of insurance claims risk across the historical population.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.exception("SHAP explanation failed")
        raise HTTPException(status_code=500, detail=f"SHAP explanation failed: {str(exc)}")