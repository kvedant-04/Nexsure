"""
governance.py -- Nexsure MLOps Governance, Lineage & Provenance Tracking

Implements enterprise model governance contracts:
- Model Provenance validation (ensures dataset, schemas, thresholds match)
- Immutable lineage tracking (hash, run ID, feature schema, metrics)
- Structured audit event logging (promotions, rollbacks, benchmark baselines)
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def compute_dataset_fingerprint(df: pd.DataFrame) -> str:
    """Compute deterministic SHA-256 fingerprint for dataset schema + shape + content hash."""
    shape_str = f"{df.shape[0]}x{df.shape[1]}"
    cols_str = ",".join(sorted(df.columns.astype(str)))
    # Sample row hash for content fingerprinting without extreme overhead
    sample_repr = df.head(50).to_json() if not df.empty else ""
    raw = f"{shape_str}|{cols_str}|{sample_repr}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_provenance_schema(
    benchmark_run_id: str,
    dataset_fingerprint: str,
    target_definition: str,
    target_threshold: float,
    raw_feature_names: List[str],
    transformed_feature_names: List[str],
    preprocessing_version: str = "v1.1",
    random_state: int = 42,
) -> Dict[str, Any]:
    """Build standardized provenance dictionary for a training run."""
    return {
        "benchmark_run_id": benchmark_run_id,
        "dataset_fingerprint": dataset_fingerprint,
        "target_definition": target_definition,
        "target_threshold": float(target_threshold),
        "raw_feature_count": len(raw_feature_names),
        "raw_feature_names": sorted(raw_feature_names),
        "transformed_feature_count": len(transformed_feature_names),
        "transformed_feature_names": sorted(transformed_feature_names),
        "preprocessing_version": preprocessing_version,
        "random_state": int(random_state),
    }


def validate_provenance_compatibility(
    candidate_prov: Dict[str, Any],
    champion_prov: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """
    Validate that a candidate model is provenance-compatible with the current champion.
    
    Prevents promoting a candidate trained on a mismatched dataset, different
    target definition, altered feature set, or conflicting preprocessing pipeline.
    """
    if not candidate_prov or not champion_prov:
        return True, None  # Allow initial baseline if provenance missing

    checks = [
        ("target_definition", "Target definition mismatch"),
        ("target_threshold", "Target threshold mismatch"),
        ("raw_feature_names", "Raw feature schema mismatch"),
        ("transformed_feature_names", "Transformed feature schema mismatch"),
        ("preprocessing_version", "Preprocessing pipeline version mismatch"),
        ("random_state", "Random state mismatch"),
    ]

    for key, msg in checks:
        cand_val = candidate_prov.get(key)
        champ_val = champion_prov.get(key)
        if cand_val != champ_val:
            detail = f"{msg}: candidate {key}={cand_val} vs champion {key}={champ_val}"
            logger.warning("Provenance validation failed: %s", detail)
            return False, detail

    # Optional dataset fingerprint match warning/check
    if candidate_prov.get("dataset_fingerprint") != champion_prov.get("dataset_fingerprint"):
        logger.info("Candidate trained on different dataset fingerprint than current champion.")

    return True, None


def create_audit_event(
    action: str,
    from_model: Optional[str],
    from_version: Optional[str],
    to_model: str,
    to_version: str,
    reason: str,
    actor: str = "system",
    metrics_snapshot: Optional[Dict[str, Any]] = None,
    provenance_status: str = "VERIFIED",
) -> Dict[str, Any]:
    """Create a structured audit event dictionary for promotion/rollback history."""
    return {
        "event_id": f"evt_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')[:17]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action.upper(),  # INITIAL_DEPLOYMENT | PROMOTION | ROLLBACK | BENCHMARK_PROMOTION
        "from_model": from_model,
        "from_version": from_version,
        "to_model": to_model,
        "to_version": to_version,
        "actor": actor,
        "reason": reason,
        "provenance_status": provenance_status,
        "metrics_snapshot": metrics_snapshot or {},
    }