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


FEATURE_MAPPINGS = {
    "age": "Age",
    "bmi": "Body Mass Index",
    "children": "Dependents",
    "smoker_1": "Smoking Profile",
    "smoker_yes": "Smoking Profile",
    "smoker_0": "Smoking Profile",
    "smoker_no": "Smoking Profile",
    "sex_1": "Gender (Male)",
    "sex_0": "Gender (Female)",
    "sex_male": "Gender (Male)",
    "sex_female": "Gender (Female)",
    "region_0": "Geographic Region",
    "region_1": "Geographic Region",
    "region_2": "Geographic Region",
    "region_3": "Geographic Region",
    "region_northeast": "Geographic Region",
    "region_northwest": "Geographic Region",
    "region_southeast": "Geographic Region",
    "region_southwest": "Geographic Region",
}

def _humanize_feature(raw_name: str) -> str:
    """Map encoded ML feature names to executive-friendly labels."""
    lower_name = raw_name.lower()
    for key, mapped in FEATURE_MAPPINGS.items():
        if key in lower_name:
            return mapped
    return raw_name.replace("_", " ").title()

def _get_impact_tier(pct: float) -> str:
    if pct >= 30: return "Critical"
    if pct >= 15: return "High"
    if pct >= 5: return "Moderate"
    return "Minor"

def _generate_feature_insight(human_name: str, impact: str, tier: str, pct: float) -> str:
    direction = "increased" if impact == "increases" else "decreased"
    if human_name == "Smoking Profile":
        if impact == "increases":
            return "Historical underwriting telemetry indicates smoking behavior significantly increased projected insurance risk."
        else:
            return "Non-smoking profile favorably lowered projected insurance risk."
    
    if human_name == "Body Mass Index":
        if impact == "increases":
            return "Elevated BMI patterns influenced the model's underwriting assessment toward higher risk."
        else:
            return "Healthy BMI range positively impacted the risk evaluation."
    
    if human_name == "Age":
        if impact == "increases":
            return "Advanced age tier contributed to a higher actuarial risk profile."
        else:
            return "Younger age demographic favorably reduced the calculated risk baseline."
    
    if human_name == "Dependents":
        return f"Number of dependents {direction} the underwriting risk assessment."
    
    return f"This factor {direction} projected risk with a {tier.lower()} level of influence."


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
    
    # Handle list of values (sometimes returned for multi-class)
    if isinstance(values, list):
        # Take class 1 if available, otherwise class 0
        values = values[1] if len(values) > 1 else values[0]

    # Handle 3D arrays: (samples, features, classes)
    if values.ndim == 3:
        # Extract first sample and class 1 (Approved)
        # Check if class 1 exists, otherwise fallback to class 0
        class_idx = 1 if values.shape[2] > 1 else 0
        return values[0, :, class_idx]

    # Handle 2D arrays: (samples, features)
    if values.ndim == 2:
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
    
    total_importance = sum(mean_importance)
    if total_importance == 0:
        total_importance = 1.0

    feature_importances = []
    for name, score in zip(feature_names, mean_importance):
        human_name = _humanize_feature(name)
        pct = (score / total_importance) * 100
        # If we already have this human name, aggregate it (e.g. region_0 + region_1)
        existing = next((item for item in feature_importances if item["feature"] == human_name), None)
        if existing:
            existing["importance"] += float(score)
            existing["impact_pct"] += float(pct)
        else:
            feature_importances.append({
                "feature": human_name, 
                "importance": float(score),
                "impact_pct": float(pct)
            })

    feature_importances.sort(key=lambda item: item["importance"], reverse=True)
    return feature_importances[:top_n]


def explain_local_prediction(
    model: Any,
    pipeline: Any,
    input_df: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], str]:
    """Generate local SHAP explanations and human-readable text for a prediction."""
    transformed = pipeline.transform(input_df)
    feature_names = _get_feature_names(pipeline)
    explainer = _get_shap_explainer(model, transformed)
    shap_values = explainer(transformed)

    shap_row = _extract_local_shap_values(shap_values)
    feature_contributions = []

    # Calculate total absolute SHAP to normalize into percentages
    total_abs_shap = sum(abs(v) for v in shap_row)
    if total_abs_shap == 0:
        total_abs_shap = 1.0

    raw_contributions = []
    for name, value in zip(feature_names, shap_row):
        raw_contributions.append({
            "raw_name": name,
            "human_name": _humanize_feature(name),
            "shap_value": float(value),
        })

    # Aggregate by human name to avoid duplicate regions or one-hots splitting importance
    agg_contributions = {}
    for rc in raw_contributions:
        hname = rc["human_name"]
        if hname not in agg_contributions:
            agg_contributions[hname] = 0.0
        agg_contributions[hname] += rc["shap_value"]

    for hname, val in agg_contributions.items():
        impact_pct = (abs(val) / total_abs_shap) * 100
        impact_dir = "increases" if val > 0 else "decreases"
        tier = _get_impact_tier(impact_pct)
        insight = _generate_feature_insight(hname, impact_dir, tier, impact_pct)
        
        feature_contributions.append({
            "feature": hname,
            "shap_value": val,
            "impact_pct": impact_pct,
            "impact_dir": impact_dir,
            "impact_level": tier,
            "insight": insight,
        })

    feature_contributions.sort(key=lambda item: abs(item["shap_value"]), reverse=True)

    # We will let predict.py build the executive summary to match the prompt's required structure
    return feature_contributions, "Explanation generated by predict pipeline."
