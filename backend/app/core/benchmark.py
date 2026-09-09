"""
benchmark.py -- Nexsure Enterprise ML Benchmark Engine (v1.1 - Validated)

Trains 5 supervised classifiers on an IDENTICAL preprocessing pipeline and
IDENTICAL train/test split, evaluates each model on a comprehensive metric set
including leakage-safe 5-fold cross-validation, selects the Champion using
a deterministic, auditable tie-breaker policy.

CHAMPION SELECTION POLICY (persisted verbatim in benchmark artifacts):
  primary:     Highest F1 Score
  tie_break_1: Higher ROC-AUC
  tie_break_2: Higher Recall
  tie_break_3: Lower model-only inference latency
  tie_break_4: Lower serialized model size (MB)

No model names are hardcoded. Champion emerges from real metric computation.
"""

import logging
import os
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

from app.core.metrics import (
    benchmark_end_to_end_latency,
    benchmark_model_only_latency,
    measure_model_size_mb,
    run_cross_validation,
    LATENCY_N_REPS,
)
from app.core.preprocess import build_preprocessing_pipeline, get_feature_groups

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Champion Selection Policy                                                   #
# --------------------------------------------------------------------------- #

CHAMPION_SELECTION_POLICY = {
    "primary":     "f1_score (higher is better)",
    "tie_break_1": "roc_auc (higher is better)",
    "tie_break_2": "recall (higher is better)",
    "tie_break_3": "model_only_latency_ms_mean (lower is better)",
    "tie_break_4": "model_size_mb (lower is better)",
}


# --------------------------------------------------------------------------- #
#  Model Registry                                                              #
# --------------------------------------------------------------------------- #

def _build_model_registry(random_state: int) -> Dict[str, Dict[str, Any]]:
    """
    Configure all 5 candidate models.

    PREPROCESSING NOTE: All models receive IDENTICAL preprocessed input
    (OHE + StandardScaler) from the shared pipeline. This ensures fair
    comparison. CatBoost does NOT use its native categorical handler.

    LIGHTGBM NOTE: n_jobs=1 is required on Windows + uvicorn due to an OpenMP
    thread-pool initialization bug in LightGBM 4.x when run inside an async
    event loop. This is documented behavior on Windows with Python 3.10.
    Additionally, OMP_NUM_THREADS=1 is set in the environment before import.
    """
    configs: Dict[str, Dict[str, Any]] = {
        "logistic_regression": {
            "class": LogisticRegression,
            "kwargs": {"solver": "liblinear", "max_iter": 1000, "random_state": random_state},
            "display_name": "Logistic Regression",
        },
        "random_forest": {
            "class": RandomForestClassifier,
            "kwargs": {
                "n_estimators": 200,
                "random_state": random_state,
                "n_jobs": -1,
                "class_weight": "balanced",
            },
            "display_name": "Random Forest",
        },
    }

    try:
        from xgboost import XGBClassifier
        configs["xgboost"] = {
            "class": XGBClassifier,
            "kwargs": {
                "n_estimators": 200,
                "random_state": random_state,
                "n_jobs": -1,
                "eval_metric": "logloss",
                "verbosity": 0,
            },
            "display_name": "XGBoost",
        }
    except ImportError:
        logger.warning("xgboost not installed -- skipping")

    try:
        # Set OMP threads before import to prevent Windows OpenMP crash
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        from lightgbm import LGBMClassifier
        configs["lightgbm"] = {
            "class": LGBMClassifier,
            "kwargs": {
                "n_estimators": 200,
                "random_state": random_state,
                "n_jobs": 1,       # Required: Windows/uvicorn OpenMP fix
                "verbosity": -1,
                "num_threads": 1,  # LightGBM-specific thread cap
            },
            "display_name": "LightGBM",
        }
    except ImportError:
        logger.warning("lightgbm not installed -- skipping")

    try:
        from catboost import CatBoostClassifier
        configs["catboost"] = {
            "class": CatBoostClassifier,
            "kwargs": {
                "iterations": 200,
                "random_seed": random_state,
                "verbose": 0,
                "thread_count": -1,
            },
            "display_name": "CatBoost",
        }
    except ImportError:
        logger.warning("catboost not installed -- skipping")

    return configs


# --------------------------------------------------------------------------- #
#  Per-Model Evaluation                                                        #
# --------------------------------------------------------------------------- #

def _evaluate_single_model(
    model_key: str,
    model_cfg: Dict[str, Any],
    X_train_t: np.ndarray,
    X_test_t: np.ndarray,
    X_test_raw: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    pipeline: Any,
    X_raw_full: pd.DataFrame,
    y_full: np.ndarray,
    random_state: int,
    preprocessing_pipeline_factory,
    dataset_size: int,
    raw_feature_count: int,
    transformed_feature_count: int,
) -> Dict[str, Any]:
    """Train, evaluate, and profile a single candidate model."""

    display_name = model_cfg["display_name"]
    ModelClass = model_cfg["class"]
    model_kwargs = model_cfg["kwargs"]

    result: Dict[str, Any] = {
        "key": model_key,
        "display_name": display_name,
        "status": "pending",
        "error": None,
        "training_duration_s": None,
    }

    try:
        # Training
        logger.info("Training %s ...", display_name)
        t_start = time.perf_counter()
        model = ModelClass(**model_kwargs)
        model.fit(X_train_t, y_train)
        training_duration_s = round(time.perf_counter() - t_start, 3)
        result["training_duration_s"] = training_duration_s

        # Held-out test set evaluation
        y_pred = model.predict(X_test_t)
        average = "binary" if len(np.unique(y_test)) == 2 else "weighted"

        accuracy  = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, average=average, zero_division=0))
        recall    = float(recall_score(y_test, y_pred, average=average, zero_division=0))
        f1        = float(f1_score(y_test, y_pred, average=average, zero_division=0))
        cm        = confusion_matrix(y_test, y_pred).tolist()
        clf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        roc_auc: Optional[float] = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test_t)
                if y_prob.shape[1] == 2:
                    roc_auc = float(roc_auc_score(y_test, y_prob[:, 1]))
            except Exception as exc:
                logger.warning("ROC-AUC failed for %s: %s", display_name, exc)

        # Dual latency measurement with identical methodology
        logger.info("Benchmarking latency for %s ...", display_name)
        lat_model = benchmark_model_only_latency(model, X_test_t, n_reps=LATENCY_N_REPS)
        lat_e2e   = benchmark_end_to_end_latency(model, pipeline, X_test_raw, n_reps=LATENCY_N_REPS)

        # Model size
        model_size_mb = measure_model_size_mb(model)

        # Leakage-safe 5-fold CV on pre-split full raw data
        logger.info("Running 5-fold CV for %s ...", display_name)
        cv_metrics = run_cross_validation(
            model_class=ModelClass,
            model_kwargs=model_kwargs,
            preprocessing_pipeline_factory=preprocessing_pipeline_factory,
            X_raw=X_raw_full,
            y=y_full,
            cv=5,
            random_state=random_state,
        )

        result.update({
            "status":                    "success",
            "model_object":              model,
            "accuracy":                  accuracy,
            "precision":                 precision,
            "recall":                    recall,
            "f1_score":                  f1,
            "roc_auc":                   roc_auc,
            "confusion_matrix":          cm,
            "classification_report":     clf_report,
            # Dual latency: mean + p50 + p95 + n_reps
            "model_only_latency_ms":     lat_model["mean_ms"],
            "model_only_p50_ms":         lat_model["p50_ms"],
            "model_only_p95_ms":         lat_model["p95_ms"],
            "end_to_end_latency_ms":     lat_e2e["mean_ms"],
            "end_to_end_p50_ms":         lat_e2e["p50_ms"],
            "end_to_end_p95_ms":         lat_e2e["p95_ms"],
            "latency_n_reps":            LATENCY_N_REPS,
            "model_size_mb":             model_size_mb,
            "dataset_size":              dataset_size,
            "raw_feature_count":         raw_feature_count,
            "transformed_feature_count": transformed_feature_count,
            "random_state":              random_state,
            **cv_metrics,
        })
        logger.info(
            "%s done -- f1=%.4f  roc=%.4f  lat_model=%.3fms  lat_e2e=%.3fms  size=%.4fMB  dur=%.3fs",
            display_name, f1, roc_auc or 0.0,
            lat_model["mean_ms"], lat_e2e["mean_ms"], model_size_mb, training_duration_s,
        )

    except Exception as exc:
        result["status"] = "failed"
        result["error"]  = f"{type(exc).__name__}: {exc}"
        logger.error("Model %s failed: %s\n%s", display_name, exc, traceback.format_exc())

    return result


# --------------------------------------------------------------------------- #
#  Champion Selection                                                          #
# --------------------------------------------------------------------------- #

def select_champion(results: Dict[str, Dict[str, Any]]) -> Tuple[str, Optional[str], List[str]]:
    """
    Select champion and runner-up from successfully benchmarked models.

    Policy (deterministic, no hardcoded model names):
      1. f1_score        descending
      2. roc_auc         descending
      3. recall          descending
      4. model_only_latency_ms ascending
      5. model_size_mb   ascending
    """
    successful = {k: v for k, v in results.items() if v["status"] == "success"}
    if not successful:
        raise RuntimeError("No models trained successfully. Cannot select a champion.")

    def sort_key(item):
        _, m = item
        return (
            -m.get("f1_score", 0.0),
            -(m.get("roc_auc") or 0.0),
            -m.get("recall", 0.0),
            m.get("model_only_latency_ms", 9999.0),
            m.get("model_size_mb", 9999.0),
        )

    ranked   = sorted(successful.items(), key=sort_key)
    ranking  = [key for key, _ in ranked]
    champion  = ranking[0]
    runner_up = ranking[1] if len(ranking) > 1 else None
    return champion, runner_up, ranking


# --------------------------------------------------------------------------- #
#  Full Benchmark Orchestrator                                                 #
# --------------------------------------------------------------------------- #

def run_benchmark(
    X_train_t: np.ndarray,
    X_test_t: np.ndarray,
    X_test_raw: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    pipeline: Any,
    X_raw_full: pd.DataFrame,
    y_full: np.ndarray,
    raw_feature_count: int,
    random_state: int = 42,
    ns_log=None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Orchestrate the full 5-model benchmark.

    Individual model failures are isolated and recorded; benchmarking
    continues for remaining candidates.

    Returns: (serializable_report, full_results_with_model_objects)
    """
    benchmark_start = time.perf_counter()
    dataset_size         = X_raw_full.shape[0]
    transformed_feature_count = X_train_t.shape[1]

    # Factory for leakage-safe CV preprocessing
    num_cols, cat_cols = get_feature_groups(X_raw_full)
    def preprocessing_pipeline_factory():
        return build_preprocessing_pipeline(num_cols, cat_cols)

    model_registry = _build_model_registry(random_state)

    if ns_log:
        ns_log.section(f"BENCHMARK ENGINE -- {len(model_registry)} CANDIDATES")

    results: Dict[str, Dict[str, Any]] = {}

    for i, (model_key, model_cfg) in enumerate(model_registry.items(), 1):
        display = model_cfg["display_name"]
        if ns_log:
            ns_log.benchmark_model_start(i, len(model_registry), display)

        result = _evaluate_single_model(
            model_key=model_key,
            model_cfg=model_cfg,
            X_train_t=X_train_t,
            X_test_t=X_test_t,
            X_test_raw=X_test_raw,
            y_train=y_train,
            y_test=y_test,
            pipeline=pipeline,
            X_raw_full=X_raw_full,
            y_full=y_full,
            random_state=random_state,
            preprocessing_pipeline_factory=preprocessing_pipeline_factory,
            dataset_size=dataset_size,
            raw_feature_count=raw_feature_count,
            transformed_feature_count=transformed_feature_count,
        )
        results[model_key] = result

        if ns_log:
            ns_log.benchmark_model_done(display, result)

    champion_key, runner_up_key, ranking = select_champion(results)
    benchmark_duration_s = round(time.perf_counter() - benchmark_start, 2)
    benchmark_timestamp  = datetime.now(timezone.utc).isoformat()

    champion_metrics = results[champion_key]

    if ns_log:
        ns_log.benchmark_table(results, ranking)
        ns_log.champion_summary(
            champion=champion_key,
            runner_up=runner_up_key,
            metrics=champion_metrics,
        )

    # Strip non-serializable model objects from report
    serializable_results = {}
    for key, r in results.items():
        serializable_results[key] = {k: v for k, v in r.items() if k != "model_object"}

    report = {
        "benchmark_timestamp":    benchmark_timestamp,
        "benchmark_duration_s":   benchmark_duration_s,
        "random_state":           random_state,
        "dataset_size":           dataset_size,
        "raw_feature_count":      raw_feature_count,
        "transformed_feature_count": transformed_feature_count,
        "latency_n_reps":         LATENCY_N_REPS,
        "champion":               champion_key,
        "runner_up":              runner_up_key,
        "ranking":                ranking,
        "selection_policy":       CHAMPION_SELECTION_POLICY,
        "model_hyperparameters":  {k: v["kwargs"] for k, v in model_registry.items()},
        "results":                serializable_results,
    }

    return report, results