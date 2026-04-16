import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def _get_feature_names(pipeline: Any) -> List[str]:
    """Return transformed feature names after preprocessing."""
    try:
        return list(pipeline.get_feature_names_out())
    except AttributeError as exc:
        raise RuntimeError(
            "The preprocessing pipeline must support get_feature_names_out()."
        ) from exc


def _get_shap_explainer(model: Any, transformed_data: np.ndarray) -> Any:
    """Create the correct SHAP explainer for the chosen classifier."""
    if hasattr(model, "feature_importances_"):
        logger.info("Using TreeExplainer for SHAP explanations")
        return shap.TreeExplainer(model)

    if hasattr(model, "coef_"):
        logger.info("Using LinearExplainer for SHAP explanations")
        return shap.LinearExplainer(model, transformed_data, feature_perturbation="interventional")

    logger.info("Using general SHAP Explainer for model explanation")
    return shap.Explainer(model, transformed_data)


def _extract_local_shap_values(shap_values: Any) -> np.ndarray:
    """Extract a single row of SHAP values for local feature importance."""
    values = shap_values.values
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]

    if values.ndim == 2 and values.shape[0] == 1:
        return values[0]

    return values


def get_global_feature_importance(
    model: Any,
    pipeline: Any,
    X_reference: pd.DataFrame,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Compute global importance across a reference dataset using SHAP values."""
    transformed_reference = pipeline.transform(X_reference)
    explainer = _get_shap_explainer(model, transformed_reference)
    shap_values = explainer(transformed_reference)

    feature_names = _get_feature_names(pipeline)
    values = shap_values.values
    if isinstance(values, list):
        values = values[1]

    mean_importance = np.mean(np.abs(values), axis=0)
    feature_importances = [
        {"feature": name, "importance": float(score)}
        for name, score in zip(feature_names, mean_importance)
    ]

    feature_importances.sort(key=lambda item: item["importance"], reverse=True)
    return feature_importances[:top_n]


def explain_local_prediction(
    model: Any,
    pipeline: Any,
    input_df: pd.DataFrame,
    top_n: int = 3,
) -> Tuple[List[Dict[str, Any]], str]:
    """Generate local SHAP explanations and human-readable text for a prediction."""
    transformed = pipeline.transform(input_df)
    feature_names = _get_feature_names(pipeline)
    explainer = _get_shap_explainer(model, transformed)
    shap_values = explainer(transformed)

    shap_row = _extract_local_shap_values(shap_values)
    feature_contributions = []

    for name, value in zip(feature_names, shap_row):
        feature_contributions.append(
            {
                "feature": name,
                "shap_value": float(value),
                "impact": "increases" if value > 0 else "decreases",
            }
        )

    feature_contributions.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
    top_features = feature_contributions[:top_n]

    explanation_parts = []
    for feature in top_features[:2]:
        direction = "higher" if feature["shap_value"] > 0 else "lower"
        explanation_parts.append(f"{direction} {feature['feature']}"
                                 if "onehot" not in feature["feature"].lower()
                                 else f"{feature['feature']}"
                                 )

    explanation_text = (
        "This claim was influenced mainly by "
        f"{explanation_parts[0]} and {explanation_parts[1]}." if len(explanation_parts) > 1 else
        f"This claim was influenced mainly by {explanation_parts[0]}."
    )

    logger.info("Generated local SHAP explanation: %s", explanation_text)
    return top_features, explanation_text
