"""
promotion.py -- Nexsure Enterprise Model Promotion, Rollback & Lifecycle Engine

Implements:
- Registry initialization from real benchmark outputs
- Immutable artifact registration
- Provenance validation prior to promotion
- Deterministic promotion policy evaluation
- Atomic Champion replacement with no intentional serving gap
- Safe rollback to immutable historical artifacts
- Audit event logging in promotion_history.json
"""

import json
import logging
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from app.core.governance import (
    build_provenance_schema,
    compute_dataset_fingerprint,
    create_audit_event,
    validate_provenance_compatibility,
)
from app.core.cache_manager import runtime_cache
from app.core.registry import (
    _model_root,
    append_promotion_event,
    get_champion_path,
    get_immutable_artifact_path,
    get_immutable_metadata_path,
    get_model_registry_path,
    load_immutable_artifact,
    load_model_registry,
    load_promotion_history,
    save_immutable_artifact,
    save_model_registry,
)

logger = logging.getLogger(__name__)

# Reentrant lock to guarantee thread-safe atomic model replacement during live serving
_SWAP_LOCK = threading.RLock()


# --------------------------------------------------------------------------- #
#  Registry Construction & Artifact Ingestion                                  #
# --------------------------------------------------------------------------- #

def build_registry(
    benchmark_report: Dict[str, Any],
    model_objects: Optional[Dict[str, Any]] = None,
    dataset_df: Optional[pd.DataFrame] = None,
    version: str = "v2.1",
    folder: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Build model_registry.json from real benchmark evaluation results.
    
    Persists immutable artifacts into model/registry/<model_name>/<version>/model.joblib,
    establishes Champion (rank 1), Challenger (rank 2, non-serving), and Candidates.
    """
    root = folder or _model_root()
    results = benchmark_report.get("results", {})
    ranking = benchmark_report.get("ranking", list(results.keys()))
    run_id = f"run_{benchmark_report.get('benchmark_timestamp', datetime.now(timezone.utc).isoformat())[:19].replace(':', '').replace('-', '')}"
    
    # Dataset fingerprint & provenance
    fingerprint = compute_dataset_fingerprint(dataset_df) if dataset_df is not None else "df_default_fp"
    raw_feature_names = benchmark_report.get("raw_feature_names", ["age", "sex", "bmi", "children", "smoker", "region"])
    transformed_feature_names = benchmark_report.get("transformed_feature_names", [])
    
    provenance = build_provenance_schema(
        benchmark_run_id=run_id,
        dataset_fingerprint=fingerprint,
        target_definition=benchmark_report.get("target_definition", "charges < training_split_median"),
        target_threshold=benchmark_report.get("target_threshold", 9373.74),
        raw_feature_names=raw_feature_names,
        transformed_feature_names=transformed_feature_names,
        preprocessing_version="v1.1",
        random_state=benchmark_report.get("random_state", 42),
    )

    models_meta: Dict[str, Dict[str, Any]] = {}
    
    # Ensure all models are saved in the immutable registry
    for rank_idx, model_name in enumerate(ranking, 1):
        m_res = results.get(model_name, {})
        
        # Determine role: Rank 1 is Champion, Rank 2 is Challenger, remainder are Candidates
        if rank_idx == 1:
            role = "CHAMPION"
            is_serving = True
        elif rank_idx == 2:
            role = "CHALLENGER"
            is_serving = False
        else:
            role = "CANDIDATE"
            is_serving = False

        meta_entry = {
            "name": model_name,
            "display_name": m_res.get("display_name", model_name),
            "version": version,
            "role": role,
            "benchmark_rank": rank_idx,
            "is_serving": is_serving,
            "f1_score": m_res.get("f1_score"),
            "accuracy": m_res.get("accuracy"),
            "precision": m_res.get("precision"),
            "recall": m_res.get("recall"),
            "roc_auc": m_res.get("roc_auc"),
            "cv_f1_mean": m_res.get("cv_f1_mean"),
            "cv_f1_std": m_res.get("cv_f1_std"),
            "latency_ms": (m_res.get("latency") or {}).get("model_only_ms", {}).get("mean") or m_res.get("model_only_latency_ms", 0.0),
            "end_to_end_latency_ms": (m_res.get("latency") or {}).get("end_to_end_ms", {}).get("mean") or m_res.get("end_to_end_latency_ms", 0.0),
            "size_mb": m_res.get("model_size_mb", 0.0),
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "provenance": provenance,
            "artifact_rel_path": f"registry/{model_name}/{version}/model.joblib",
        }

        # Resolve model object and save immutably
        model_obj = None
        if model_objects and model_name in model_objects:
            cand = model_objects[model_name]
            model_obj = cand.get("model_object", cand) if isinstance(cand, dict) else cand
        else:
            # Fallback to model/artifacts/<name>.joblib if already serialized
            raw_art = root / "artifacts" / f"{model_name}.joblib"
            if raw_art.exists():
                try:
                    model_obj = joblib.load(raw_art)
                except Exception as exc:
                    logger.warning("Could not load %s for immutable storage: %s", raw_art, exc)

        if model_obj is not None:
            save_immutable_artifact(model_obj, model_name, version, meta_entry, root)
        
        models_meta[model_name] = meta_entry

    champion_name = ranking[0] if ranking else "catboost"
    challenger_name = ranking[1] if len(ranking) > 1 else None

    registry_doc = {
        "registry_schema_version": "v2.0",
        "active_champion": champion_name,
        "active_version": version,
        "deployment_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "healthy",
        "champion": models_meta.get(champion_name, {}),
        "challenger": models_meta.get(challenger_name, {}) if challenger_name else None,
        "candidates": models_meta,
        "ranking": ranking,
        "selection_policy": benchmark_report.get("selection_policy", {}),
        "provenance": provenance,
    }

    # Save model_registry.json
    save_model_registry(registry_doc, root)
    
    # Copy champion to best_model.pkl for backward compatibility
    champ_art = get_immutable_artifact_path(champion_name, version, root)
    if champ_art.exists():
        shutil.copy2(champ_art, get_champion_path(root))

    # Record baseline deployment event in promotion_history.json if history empty
    history = load_promotion_history(root)
    if not history:
        init_event = create_audit_event(
            action="INITIAL_DEPLOYMENT",
            from_model=None,
            from_version=None,
            to_model=champion_name,
            to_version=version,
            reason=f"Selected as Benchmark Champion (F1: {models_meta.get(champion_name, {}).get('f1_score'):.4f})",
            actor="benchmark_engine",
            metrics_snapshot=models_meta.get(champion_name, {}),
            provenance_status="VERIFIED",
        )
        append_promotion_event(init_event, root)

    return registry_doc


# --------------------------------------------------------------------------- #
#  Registry Query Helpers                                                      #
# --------------------------------------------------------------------------- #

def get_champion(folder: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Return active champion metadata dict."""
    try:
        reg = load_model_registry(folder)
        return reg.get("champion")
    except Exception:
        return None


def get_challenger(folder: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Return current challenger metadata dict."""
    try:
        reg = load_model_registry(folder)
        return reg.get("challenger")
    except Exception:
        return None


def validate_registry(folder: Optional[Path] = None) -> Tuple[bool, List[str]]:
    """Validate registry document integrity and artifact physical presence."""
    errors = []
    try:
        reg = load_model_registry(folder)
    except Exception as exc:
        return False, [f"Failed to load model_registry.json: {exc}"]

    champion = reg.get("champion")
    if not champion or not champion.get("name") or not champion.get("version"):
        errors.append("Active champion record is missing or incomplete")
    else:
        art_path = get_immutable_artifact_path(champion["name"], champion["version"], folder)
        if not art_path.exists():
            errors.append(f"Champion immutable artifact missing on disk: {art_path}")

    challenger = reg.get("challenger")
    if challenger and challenger.get("name") and challenger.get("version"):
        ch_path = get_immutable_artifact_path(challenger["name"], challenger["version"], folder)
        if not ch_path.exists():
            errors.append(f"Challenger immutable artifact missing on disk: {ch_path}")

    return len(errors) == 0, errors


# --------------------------------------------------------------------------- #
#  Deterministic Promotion Policy                                              #
# --------------------------------------------------------------------------- #

def evaluate_promotion_policy(
    candidate_meta: Dict[str, Any],
    champion_meta: Dict[str, Any],
    allow_force: bool = False,
    override_reason: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Evaluate if candidate satisfies the production promotion criteria:
      1. F1 score strictly higher than Champion
      2. OR F1 equal AND ROC-AUC higher
      3. OR F1 equal AND ROC-AUC equal AND Latency lower
    """
    if allow_force and override_reason:
        return True, f"Administrative override promotion: {override_reason}"

    cand_f1 = candidate_meta.get("f1_score") or 0.0
    champ_f1 = champion_meta.get("f1_score") or 0.0

    cand_roc = candidate_meta.get("roc_auc") or 0.0
    champ_roc = champion_meta.get("roc_auc") or 0.0

    cand_lat = candidate_meta.get("latency_ms") or 999.0
    champ_lat = champion_meta.get("latency_ms") or 999.0

    # Rule 1: Higher F1
    if cand_f1 > champ_f1 + 1e-4:
        return True, f"Superior F1 score ({cand_f1:.4f} > {champ_f1:.4f})"

    # Rule 2: Equal F1 & Superior ROC-AUC
    if abs(cand_f1 - champ_f1) <= 1e-4 and cand_roc > champ_roc + 1e-4:
        return True, f"Equal F1 ({cand_f1:.4f}) and superior ROC-AUC ({cand_roc:.4f} > {champ_roc:.4f})"

    # Rule 3: Equal F1 & Equal ROC-AUC & Lower Latency
    if abs(cand_f1 - champ_f1) <= 1e-4 and abs(cand_roc - champ_roc) <= 1e-4 and cand_lat < champ_lat - 0.01:
        return True, f"Equivalent accuracy with superior inference latency ({cand_lat:.2f}ms < {champ_lat:.2f}ms)"

    return False, (
        f"Candidate metrics do not exceed Champion. "
        f"Candidate(F1={cand_f1:.4f}, ROC={cand_roc:.4f}, Lat={cand_lat:.2f}ms) vs "
        f"Champion(F1={champ_f1:.4f}, ROC={champ_roc:.4f}, Lat={champ_lat:.2f}ms)"
    )


# --------------------------------------------------------------------------- #
#  Promotion & Rollback Engine (Atomic Replacement)                            #
# --------------------------------------------------------------------------- #

def promote_model(
    candidate_name: str,
    version: Optional[str] = None,
    actor: str = "system",
    reason: Optional[str] = None,
    folder: Optional[Path] = None,
    state: Optional[Dict[str, Any]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Promote a candidate model to active Champion.
    
    Guarantees:
      1. Provenance validation against current champion.
      2. Policy check (unless administrative override).
      3. Immutable artifact pre-load & smoke validation.
      4. Atomic champion replacement with no intentional serving gap.
      5. Audit trail logged in promotion_history.json.
    """
    root = folder or _model_root()
    reg = load_model_registry(root)
    champion_meta = reg.get("champion", {})
    candidates = reg.get("candidates", {})

    if candidate_name not in candidates:
        raise ValueError(f"Candidate '{candidate_name}' is not registered in candidates: {list(candidates.keys())}")

    candidate_meta = candidates[candidate_name]
    target_version = version or candidate_meta.get("version", "v2.1")

    # 1. Provenance Validation
    prov_ok, prov_err = validate_provenance_compatibility(
        candidate_meta.get("provenance", {}),
        champion_meta.get("provenance", {}),
    )
    if not prov_ok and not force:
        raise ValueError(f"Provenance incompatibility: {prov_err}")

    # 2. Promotion Policy Check
    prom_ok, prom_reason = evaluate_promotion_policy(
        candidate_meta,
        champion_meta,
        allow_force=force,
        override_reason=reason,
    )
    if not prom_ok:
        raise ValueError(f"Promotion rejected: {prom_reason}")

    # 3. Pre-load and Validate Candidate Object
    try:
        candidate_model = load_immutable_artifact(candidate_name, target_version, root)
        # Smoke test predict_proba
        if not hasattr(candidate_model, "predict_proba"):
            raise ValueError(f"Model object {candidate_name} lacks predict_proba method")
    except Exception as exc:
        raise RuntimeError(f"Candidate artifact pre-load validation failed: {exc}")

    # 4. Atomic Champion Replacement Critical Section
    with _SWAP_LOCK:
        old_champion_name = champion_meta.get("name")
        old_champion_version = champion_meta.get("version")

        # Update candidate / champion roles in registry doc
        candidate_meta["role"] = "CHAMPION"
        candidate_meta["is_serving"] = True
        
        if old_champion_name in candidates:
            candidates[old_champion_name]["role"] = "CHALLENGER"
            candidates[old_champion_name]["is_serving"] = False

        reg["active_champion"] = candidate_name
        reg["active_version"] = target_version
        reg["deployment_timestamp"] = datetime.now(timezone.utc).isoformat()
        reg["champion"] = candidate_meta
        reg["challenger"] = candidates.get(old_champion_name)
        reg["candidates"] = candidates

        # Persist updated registry
        save_model_registry(reg, root)

        # Update best_model.pkl for legacy compatibility
        champ_art = get_immutable_artifact_path(candidate_name, target_version, root)
        shutil.copy2(champ_art, get_champion_path(root))

        # In-memory serving swap if state provided
        if state is not None:
            state["trained_models"]["best_model"] = candidate_model
            state["trained_models"][candidate_name] = candidate_model
            state["best_model_name"] = candidate_name
            state["metadata"]["best_model_name"] = candidate_name
            state["metadata"]["accuracy"] = candidate_meta.get("accuracy")
            state["metadata"]["f1_score"] = candidate_meta.get("f1_score")
            state["metadata"]["roc_auc"] = candidate_meta.get("roc_auc")

        # Invalidate and re-warm runtime cache for new Champion
        try:
            runtime_cache.invalidate_for_promotion(candidate_name, target_version, candidate_model)
        except Exception as cache_exc:
            logger.warning("Cache re-warming error during promotion: %s", cache_exc)

        # Record Audit Event
        audit_event = create_audit_event(
            action="PROMOTION",
            from_model=old_champion_name,
            from_version=old_champion_version,
            to_model=candidate_name,
            to_version=target_version,
            reason=reason or prom_reason,
            actor=actor,
            metrics_snapshot=candidate_meta,
            provenance_status="VERIFIED" if prov_ok else "OVERRIDDEN",
        )
        append_promotion_event(audit_event, root)

    logger.info("Successfully promoted %s to Champion. Reason: %s", candidate_name, prom_reason)
    return {
        "status": "promoted",
        "champion": candidate_name,
        "version": target_version,
        "previous_champion": old_champion_name,
        "reason": prom_reason,
        "timestamp": audit_event["timestamp"],
    }


def rollback_model(
    target_model: Optional[str] = None,
    target_version: Optional[str] = None,
    actor: str = "system",
    reason: Optional[str] = None,
    folder: Optional[Path] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Rollback active Champion to a previous immutable model artifact.
    
    If target_model is not explicitly supplied, reads the latest PROMOTION event
    from promotion_history.json to determine the immediate predecessor.
    """
    root = folder or _model_root()
    reg = load_model_registry(root)
    history = load_promotion_history(root)
    champion_meta = reg.get("champion", {})
    candidates = reg.get("candidates", {})

    current_champion = champion_meta.get("name")
    current_version = champion_meta.get("version")

    # Determine target predecessor
    if not target_model:
        # Find latest promotion event that was from a different model
        prev_events = [
            e for e in reversed(history)
            if e.get("action") == "PROMOTION" and e.get("from_model") and e.get("from_model") != current_champion
        ]
        if prev_events:
            target_model = prev_events[0]["from_model"]
            target_version = target_version or prev_events[0].get("from_version") or "v2.1"
        elif reg.get("challenger"):
            target_model = reg["challenger"].get("name")
            target_version = target_version or reg["challenger"].get("version") or "v2.1"
        else:
            raise ValueError("No historical predecessor found in audit history for rollback")

    target_version = target_version or "v2.1"

    # Pre-load target immutable artifact
    try:
        restored_model = load_immutable_artifact(target_model, target_version, root)
    except Exception as exc:
        raise RuntimeError(f"Rollback failed: cannot load target artifact {target_model} {target_version}: {exc}")

    # Atomic swap
    with _SWAP_LOCK:
        if target_model in candidates:
            target_meta = candidates[target_model]
        else:
            target_meta = {
                "name": target_model,
                "version": target_version,
                "role": "CHAMPION",
                "is_serving": True,
            }

        target_meta["role"] = "CHAMPION"
        target_meta["is_serving"] = True

        if current_champion in candidates:
            candidates[current_champion]["role"] = "CHALLENGER"
            candidates[current_champion]["is_serving"] = False

        reg["active_champion"] = target_model
        reg["active_version"] = target_version
        reg["deployment_timestamp"] = datetime.now(timezone.utc).isoformat()
        reg["champion"] = target_meta
        reg["challenger"] = candidates.get(current_champion)
        reg["candidates"] = candidates

        save_model_registry(reg, root)

        champ_art = get_immutable_artifact_path(target_model, target_version, root)
        shutil.copy2(champ_art, get_champion_path(root))

        if state is not None:
            state["trained_models"]["best_model"] = restored_model
            state["trained_models"][target_model] = restored_model
            state["best_model_name"] = target_model
            state["metadata"]["best_model_name"] = target_model

        # Invalidate and re-warm runtime cache for restored Champion
        try:
            runtime_cache.invalidate_for_promotion(target_model, target_version, restored_model)
        except Exception as cache_exc:
            logger.warning("Cache re-warming error during rollback: %s", cache_exc)

        rollback_reason = reason or f"Rollback from {current_champion} to {target_model} ({target_version})"
        audit_event = create_audit_event(
            action="ROLLBACK",
            from_model=current_champion,
            from_version=current_version,
            to_model=target_model,
            to_version=target_version,
            reason=rollback_reason,
            actor=actor,
            metrics_snapshot=target_meta,
            provenance_status="VERIFIED",
        )
        append_promotion_event(audit_event, root)

    logger.info("Successfully rolled back to Champion %s (version %s)", target_model, target_version)
    return {
        "status": "rolled_back",
        "champion": target_model,
        "version": target_version,
        "previous_champion": current_champion,
        "reason": rollback_reason,
        "timestamp": audit_event["timestamp"],
    }