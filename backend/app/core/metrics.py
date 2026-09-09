"""
metrics.py -- Nexsure Benchmark Metrics Utilities

Provides:
- Leakage-safe cross-validation using sklearn Pipeline (preprocessing + estimator)
- Two-tier latency measurement (model-only vs. end-to-end) with IDENTICAL methodology:
    * same n_repetitions for both
    * both measure single-row predictions (NOT batch)
    * both use time.perf_counter()
    * both return mean + p50 + p95
- Serialized model-size via joblib BytesIO (same codec as on-disk artifacts)

All statistics come from real computation. No fake values.
"""

import io
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Standard repetition count used by BOTH latency measurements
LATENCY_N_REPS = 200


# --------------------------------------------------------------------------- #
#  Cross-Validation                                                            #
# --------------------------------------------------------------------------- #

def run_cross_validation(
    model_class,
    model_kwargs: Dict[str, Any],
    preprocessing_pipeline_factory,
    X_raw: pd.DataFrame,
    y,
    cv: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Leakage-safe stratified K-fold cross-validation.

    Per fold:
      1. Split RAW (untransformed) X into fold_train / fold_val.
      2. Instantiate fresh sklearn Pipeline: preprocessing -> estimator.
      3. Fit pipeline on fold_train only.
      4. Evaluate on fold_val.

    Scaler means/std and OHE category sets are computed exclusively from fold
    training data -- validation data is never seen during fit. Zero leakage.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    acc_scores: List[float] = []
    f1_scores: List[float] = []
    roc_scores: List[float] = []

    y_arr = np.array(y)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_raw, y_arr)):
        X_fold_train = X_raw.iloc[train_idx]
        X_fold_val = X_raw.iloc[val_idx]
        y_fold_train = y_arr[train_idx]
        y_fold_val = y_arr[val_idx]

        prep_pipe = preprocessing_pipeline_factory()  # fresh per fold
        estimator = model_class(**model_kwargs)
        full_pipe = Pipeline([
            ("preprocessing", prep_pipe),
            ("estimator", estimator),
        ])

        full_pipe.fit(X_fold_train, y_fold_train)
        y_pred = full_pipe.predict(X_fold_val)

        acc_scores.append(float(np.mean(y_pred == y_fold_val)))
        f1_scores.append(float(f1_score(y_fold_val, y_pred, average="binary", zero_division=0)))

        if hasattr(full_pipe.named_steps["estimator"], "predict_proba"):
            try:
                y_prob = full_pipe.predict_proba(X_fold_val)[:, 1]
                roc_scores.append(float(roc_auc_score(y_fold_val, y_prob)))
            except Exception as exc:
                logger.warning("ROC-AUC failed fold %d: %s", fold_idx, exc)

    return {
        "cv_accuracy_mean": float(np.mean(acc_scores)),
        "cv_accuracy_std":  float(np.std(acc_scores)),
        "cv_f1_mean":       float(np.mean(f1_scores)),
        "cv_f1_std":        float(np.std(f1_scores)),
        "cv_roc_auc_mean":  float(np.mean(roc_scores)) if roc_scores else None,
        "cv_roc_auc_std":   float(np.std(roc_scores))  if roc_scores else None,
        "cv_folds":         cv,
    }


# --------------------------------------------------------------------------- #
#  Latency Benchmarking                                                        #
# --------------------------------------------------------------------------- #

def _measure_per_row_latencies(measure_fn, n_reps: int) -> Dict[str, float]:
    """
    Run measure_fn(i) n_reps times, collect per-iteration wall times,
    and return mean / p50 / p95 in milliseconds.

    measure_fn receives the row index so callers can cycle through rows.
    """
    times_ms: List[float] = []
    for i in range(n_reps):
        t0 = time.perf_counter()
        measure_fn(i)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    arr = np.array(times_ms)
    return {
        "mean_ms": round(float(np.mean(arr)), 4),
        "p50_ms":  round(float(np.percentile(arr, 50)), 4),
        "p95_ms":  round(float(np.percentile(arr, 95)), 4),
        "n_reps":  n_reps,
    }


def benchmark_model_only_latency(
    model,
    X_test_transformed: np.ndarray,
    n_reps: int = LATENCY_N_REPS,
) -> Dict[str, float]:
    """
    Model-only latency: single-row predict on pre-transformed data.
    Preprocessing is EXCLUDED. Measures raw estimator cost only.

    Methodology:
    - n_reps individual single-row predictions
    - Each iteration timed independently with perf_counter
    - Returns mean, p50, p95 in ms + repetition count
    """
    n_rows = X_test_transformed.shape[0]

    def measure(i: int):
        row = X_test_transformed[i % n_rows : i % n_rows + 1]
        model.predict(row)

    result = _measure_per_row_latencies(measure, n_reps)
    logger.info(
        "Model-only latency: mean=%.4fms p50=%.4fms p95=%.4fms (%d reps)",
        result["mean_ms"], result["p50_ms"], result["p95_ms"], n_reps,
    )
    return result


def benchmark_end_to_end_latency(
    model,
    pipeline,
    X_test_raw: pd.DataFrame,
    n_reps: int = LATENCY_N_REPS,
) -> Dict[str, float]:
    """
    End-to-end latency: preprocessing + single-row predict on raw data.
    Simulates the full production path: raw input -> transform -> predict.

    Methodology:
    - Identical n_reps to model_only measurement
    - Each iteration timed independently with perf_counter
    - Returns mean, p50, p95 in ms + repetition count
    """
    n_rows = X_test_raw.shape[0]

    def measure(i: int):
        row_df = X_test_raw.iloc[i % n_rows : i % n_rows + 1]
        row_t = pipeline.transform(row_df)
        model.predict(row_t)

    result = _measure_per_row_latencies(measure, n_reps)
    logger.info(
        "End-to-end latency: mean=%.4fms p50=%.4fms p95=%.4fms (%d reps)",
        result["mean_ms"], result["p50_ms"], result["p95_ms"], n_reps,
    )
    return result


# --------------------------------------------------------------------------- #
#  Model Size                                                                  #
# --------------------------------------------------------------------------- #

def measure_model_size_mb(model) -> float:
    """
    Serialize model to in-memory BytesIO buffer with joblib (identical codec
    to on-disk persistence) and return the size in MB.
    """
    buf = io.BytesIO()
    joblib.dump(model, buf)
    size_bytes = buf.tell()
    return round(size_bytes / (1024 * 1024), 4)