import logging
import time
from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def _get_average_method(y_test) -> str:
    """Choose an appropriate averaging method based on label cardinality."""
    unique_labels = np.unique(y_test)
    return "binary" if len(unique_labels) == 2 else "weighted"


def evaluate_model(model: Any, X_test, y_test) -> Dict[str, Any]:
    """
    Evaluate a single model and return standard classification metrics.
    Includes real inference latency measured over the test set.
    """
    # ── Inference Latency (single-sample average) ──────────────────────────────
    sample_size = min(50, len(y_test))
    lat_start = time.perf_counter()
    for i in range(sample_size):
        sample = X_test[i : i + 1] if isinstance(X_test, np.ndarray) else X_test[i : i + 1]
        model.predict(sample)
    lat_end = time.perf_counter()
    inference_latency_ms = round(((lat_end - lat_start) / sample_size) * 1000, 2)

    y_pred = model.predict(X_test)
    average = _get_average_method(y_test)

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average=average, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average=average, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "average_method": average,
        "inference_latency_ms": inference_latency_ms,
    }

    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
            if y_prob.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob[:, 1]))
        except Exception as exc:
            logger.warning("Could not compute roc_auc for model %s: %s", model, exc)

    logger.info(
        "Evaluation completed: accuracy=%.4f, precision=%.4f, recall=%.4f, "
        "f1=%.4f, latency=%.2fms",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        inference_latency_ms,
    )
    return metrics


def evaluate_models(models: Dict[str, Any], X_test, y_test) -> Dict[str, Dict[str, Any]]:
    """Evaluate all trained models and return a mapping of metric dictionaries."""
    results = {}
    for model_name, model in models.items():
        logger.info("Evaluating model %s", model_name)
        results[model_name] = evaluate_model(model, X_test, y_test)
    return results
