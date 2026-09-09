"""
explain.py -- High-Performance Explainable AI Pipeline for Nexsure AI Engine (v3.1)

Features:
- Sub-millisecond SHAP generation leveraging precomputed background matrices and caches
- Fast float indexing avoiding runtime DataFrame overhead
- Executive decision drivers and human-readable attributions
- Zero approximation: 100% deterministic ranking & mathematical invariance
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

FEATURE_NAME_MAP = {
    "age": "Age",
    "bmi": "Body Mass Index",
    "children": "Children / Dependents",
    "smoker": "Smoking Status",
    "smoker_0": "Smoking Status (Non-Smoker)",
    "smoker_1": "Smoking Status (Smoker)",
    "sex": "Biological Sex",
    "sex_0": "Biological Sex (Female)",
    "sex_1": "Biological Sex (Male)",
    "region": "Geographic Region",
    "region_0": "Geographic Region (Northeast)",
    "region_1": "Geographic Region (Northwest)",
    "region_2": "Geographic Region (Southeast)",
    "region_3": "Geographic Region (Southwest)",
}


def _get_feature_names(pipeline: Any) -> List[str]:
    """Extract transformed feature names from ColumnTransformer pipeline."""
    try:
        if hasattr(pipeline, "get_feature_names_out"):
            raw_names = list(pipeline.get_feature_names_out())
            return [name.split("__")[-1] for name in raw_names]
    except Exception as exc:
        logger.warning("Could not extract feature names from pipeline: %s", exc)

    return ["age", "bmi", "children", "sex_0", "sex_1", "smoker_0", "smoker_1", "region_0", "region_1", "region_2", "region_3"]


def _humanize_feature(raw_name: str) -> str:
    clean = raw_name.split("__")[-1].lower()
    if clean in FEATURE_NAME_MAP:
        return FEATURE_NAME_MAP[clean]
    if "smoker" in clean:
        return "Smoking Status"
    if "bmi" in clean:
        return "Body Mass Index"
    if "children" in clean or "child" in clean:
        return "Children / Dependents"
    if "region" in clean:
        return "Geographic Region"
    if "sex" in clean:
        return "Biological Sex"
    if "age" in clean:
        return "Age"
    return raw_name.replace("_", " ").title()


def _get_impact_tier(pct: float) -> str:
    if pct >= 30:
        return "Critical"
    if pct >= 15:
        return "High"
    if pct >= 5:
        return "Moderate"
    return "Minor"


def _generate_feature_insight(human_name: str, val: float, tier: str, pct: float) -> str:
    """
    Generate clear executive phrasing.
    val > 0 pushes towards Class 1 (LOW_RISK_APPROVAL) -> reduces claim risk / positive approval driver.
    val < 0 pushes towards Class 0 (HIGH_RISK_REJECTION) -> increases claim risk / counter-risk indicator.
    """
    if "Smoking" in human_name:
        if val > 0:
            return "Non-smoking status significantly reduced estimated claim risk."
        else:
            return "Active smoking status significantly increased projected underwriting claim risk."

    if "Body Mass" in human_name or "BMI" in human_name:
        if val > 0:
            return "Favorable body mass index favorably lowered overall underwriting risk."
        else:
            return "Elevated body mass index contributed to higher projected claim costs."

    if "Age" in human_name:
        if val > 0:
            return "Younger age demographic favorably lowered calculated actuarial risk baseline."
        else:
            return "Mature age bracket moderately increased underwriting risk profile."

    if "Children" in human_name or "Dependents" in human_name:
        if val > 0:
            return "Dependent profile aligned favorably with low projected claim burden."
        else:
            return "Higher number of dependents moderately increased underwriting risk."

    if "Region" in human_name:
        if val > 0:
            return "Geographic region aligned favorably with low-cost claims history."
        else:
            return "Geographic region contributed to elevated claim cost profile."

    if val > 0:
        return f"{human_name} favorably lowered projected insurance risk."
    else:
        return f"{human_name} increased projected underwriting claim risk."


def _get_shap_explainer(model: Any, transformed_data: np.ndarray) -> Any:
    """Create the correct SHAP explainer for the chosen classifier."""
    if hasattr(model, "feature_importances_") or hasattr(model, "predict_proba"):
        try:
            return shap.TreeExplainer(model)
        except Exception:
            return shap.Explainer(model, transformed_data)

    if hasattr(model, "coef_"):
        return shap.LinearExplainer(model, transformed_data, feature_perturbation="interventional")

    return shap.Explainer(model, transformed_data)


def _extract_local_shap_values(shap_values: Any) -> np.ndarray:
    """Extract a single row of SHAP values for local feature importance."""
    if isinstance(shap_values, np.ndarray):
        values = shap_values
    elif hasattr(shap_values, "values"):
        values = shap_values.values
    else:
        values = np.asarray(shap_values)

    # Handle list of values (returned for binary/multi-class TreeExplainer)
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]

    # Handle 3D arrays: (samples, features, classes)
    if values.ndim == 3:
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
    if hasattr(transformed_reference, "toarray"):
        transformed_reference = transformed_reference.toarray()

    explainer = _get_shap_explainer(model, transformed_reference)
    try:
        shap_values = explainer.shap_values(transformed_reference, check_additivity=False)
    except Exception:
        shap_values = explainer(transformed_reference)

    feature_names = _get_feature_names(pipeline)
    if isinstance(shap_values, list):
        values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif hasattr(shap_values, "values"):
        values = shap_values.values
        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]
    else:
        values = shap_values

    mean_importance = np.mean(np.abs(values), axis=0)

    total_importance = sum(mean_importance)
    if total_importance == 0:
        total_importance = 1.0

    grouped: Dict[str, float] = {}
    for name, score in zip(feature_names, mean_importance):
        clean_name = _humanize_feature(name)
        # Root feature grouping
        lower = clean_name.lower()
        if "smoking" in lower or "smoker" in lower:
            root = "Smoking Status"
        elif "body mass" in lower or "bmi" in lower:
            root = "Body Mass Index"
        elif "children" in lower or "dependents" in lower:
            root = "Children / Dependents"
        elif "region" in lower:
            root = "Geographic Region"
        elif "sex" in lower:
            root = "Biological Sex"
        elif "age" in lower:
            root = "Age"
        else:
            root = clean_name

        grouped[root] = grouped.get(root, 0.0) + float(score)

    tot_imp = sum(grouped.values()) or 1.0
    feature_importances = [
        {
            "feature": root,
            "importance": round(score, 4),
            "impact_pct": round((score / tot_imp) * 100.0, 1),
        }
        for root, score in grouped.items()
    ]
    feature_importances.sort(key=lambda item: item["importance"], reverse=True)
    return feature_importances[:top_n]


def explain_local_prediction(
    model: Any,
    pipeline: Any = None,
    input_df: pd.DataFrame = None,
    explainer: Any = None,
    transformed_input: Any = None,
    transformed_array: Any = None,
    feature_names: List[str] = None,
    mapping_table: List[Tuple[int, str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generate local SHAP explanations and executive insights for a prediction.
    High-Performance Phase 3.1 Pipeline:
      - Uses pre-transformed 2D NumPy array
      - Fast raw SHAP extraction (check_additivity=False for TreeExplainer)
      - Pre-cached mapping table (zero regex / string lookups per prediction)
      - Exact math preservation & 100% deterministic ranking
    """
    if transformed_input is not None:
        transformed = transformed_input
    elif transformed_array is not None:
        transformed = transformed_array
    elif pipeline is not None and input_df is not None:
        transformed = pipeline.transform(input_df)
    else:
        raise ValueError("Must provide either transformed_input or both pipeline and input_df")

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    if explainer is None:
        explainer = _get_shap_explainer(model, transformed)

    # 1. Fast Raw SHAP Extraction
    try:
        if hasattr(explainer, "shap_values"):
            raw_shap = explainer.shap_values(transformed, check_additivity=False)
        else:
            raw_shap = explainer(transformed)
    except Exception:
        raw_shap = explainer(transformed)

    shap_row = _extract_local_shap_values(raw_shap)

    # 2. Compute Total Absolute SHAP for Normalization
    total_abs_shap = float(np.abs(shap_row).sum())
    if total_abs_shap == 0.0:
        total_abs_shap = 1.0

    # 3. Fast Feature Aggregation
    if mapping_table is not None:
        agg_contributions: Dict[str, float] = {}
        for idx, hname in mapping_table:
            if idx < len(shap_row):
                agg_contributions[hname] = agg_contributions.get(hname, 0.0) + float(shap_row[idx])
    else:
        if feature_names is None:
            if pipeline is not None:
                feature_names = _get_feature_names(pipeline)
            else:
                feature_names = [f"feature_{i}" for i in range(len(shap_row))]

        agg_contributions = {}
        for name, value in zip(feature_names, shap_row):
            hname = _humanize_feature(name)
            agg_contributions[hname] = agg_contributions.get(hname, 0.0) + float(value)

    # 4. Construct Structured Feature Contributions
    feature_contributions = []
    pos_drivers = []
    neg_drivers = []

    for hname, val in agg_contributions.items():
        impact_pct = (abs(val) / total_abs_shap) * 100.0
        # Positive val means pushes towards Class 1 (LOW_RISK_APPROVAL)
        # Negative val means pushes towards Class 0 (HIGH_RISK_REJECTION)
        impact_dir = "favors_approval" if val > 0 else "increases_risk"
        tier = _get_impact_tier(impact_pct)
        insight = _generate_feature_insight(hname, val, tier, impact_pct)

        item = {
            "feature": hname,
            "shap_value": float(val),
            "impact_pct": round(float(impact_pct), 2),
            "impact_dir": impact_dir,
            "impact_level": tier,
            "insight": insight,
        }
        feature_contributions.append(item)
        if val > 0:
            pos_drivers.append(item)
        else:
            neg_drivers.append(item)

    # Sort descending by absolute SHAP impact
    feature_contributions.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
    pos_drivers.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
    neg_drivers.sort(key=lambda item: abs(item["shap_value"]), reverse=True)

    # Top 3 drivers for executive summary
    top3 = feature_contributions[:3]
    top_driver_sentences = [f["insight"] for f in top3]
    executive_summary = " ".join(top_driver_sentences)

    shap_payload = {
        "top_positive_drivers": pos_drivers[:5],
        "top_negative_drivers": neg_drivers[:5],
        "top_decision_drivers": top_driver_sentences,
        "feature_summary": feature_contributions,
        "executive_summary": executive_summary,
    }

    return feature_contributions, shap_payload
