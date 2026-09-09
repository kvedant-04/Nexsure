"""
Nexsure AI Engine -- Application Entry Point (v2.0 - Champion/Challenger Registry)

Production startup lifecycle:
  1. Validate core artifacts, model registry, and immutable champion binary.
  2. Validate target semantics and champion classes provenance.
  3. If all valid: load active Champion directly (fast path, zero retraining).
  4. If registry invalid: attempt rebuilding registry from benchmark_results.json.
  5. If rebuild fails: fallback to previous registry backup snapshot.
  6. If snapshot unavailable: run full 5-model benchmark, initialize fresh registry,
     record initial deployment in promotion_history.json.

GOVERNANCE CONTRACT:
  - Active Champion is the ONLY serving model.
  - Challenger is ranked #2 and non-serving until promoted.
  - All artifacts are versioned and immutable (model/registry/<model>/<version>/model.joblib).
  - Provenance validation prevents incompatible candidate promotions.
"""

import logging
import os
import time
from datetime import datetime, timezone

# Ensure OpenMP thread-pool safety on Windows with asyncio
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.predict import router as predict_router
import app.core.logger as ns_log

_logger = logging.getLogger(__name__)

app = FastAPI(
    title="Nexsure AI Underwriting Engine",
    version="2.0.0",
    description="Enterprise AI underwriting platform with Champion/Challenger Model Registry",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api")

DATASET_FILENAME = "insurance3r2.csv"
TARGET_COLUMN    = "charges"
TEST_SIZE        = 0.20
RANDOM_STATE     = 42
RAW_FEATURE_NAMES = ["age", "sex", "bmi", "children", "smoker", "region"]


def validate_target_semantics_or_raise(metadata: dict, champion_model: object) -> dict:
    """Validate target semantics against metadata and champion model classes."""
    target_def = metadata.get("target_definition")
    if not target_def or not isinstance(target_def, dict):
        raise RuntimeError("Target semantics missing. Refusing production inference.")

    required_fields = [
        "positive_class",
        "negative_class",
        "positive_class_name",
        "negative_class_name",
        "threshold_source",
    ]
    for field in required_fields:
        if field not in target_def:
            raise RuntimeError(f"Target semantics missing field '{field}'. Refusing production inference.")

    pos_class = target_def["positive_class"]
    neg_class = target_def["negative_class"]

    classes = list(getattr(champion_model, "classes_", [0, 1]))
    if pos_class not in classes or neg_class not in classes:
        raise RuntimeError(
            f"Champion model classes {classes} do not match metadata target semantics ({pos_class}, {neg_class}). Refusing production inference."
        )

    ns_log.target_semantics_verified(
        positive_class_name=target_def["positive_class_name"],
        negative_class_name=target_def["negative_class_name"],
        positive_class=pos_class,
        negative_class=neg_class,
        threshold_source=target_def.get("threshold_source", "training_split_median"),
        target_formula=target_def.get("target_formula", "charges < training_median"),
    )
    return target_def


@app.on_event("startup")
async def startup_lifecycle():
    """Autonomous model registry & benchmark lifecycle management."""
    startup_start = time.perf_counter()

    from app.api.predict import _set_global_state, _set_training_status
    from app.core.registry import (
        all_artifacts_valid,
        load_champion_model,
        load_metadata,
        load_model_registry,
        load_promotion_history,
        load_version_history,
        save_benchmark_report,
        save_metadata_v2,
        append_version_history_v2,
        load_benchmark_report,
        validate_registry_artifact,
        get_model_root_folder,
    )
    from app.core.promotion import build_registry, validate_registry
    from app.core.cache_manager import runtime_cache
    from app.core.train import load_pipeline, load_feature_columns
    from app.core.data_loader import load_dataset
    from app.core.preprocess import fit_preprocessing_pipeline, split_features_target
    from app.core.benchmark import run_benchmark
    import json
    import joblib
    import numpy as np

    MODEL_ROOT = get_model_root_folder()

    ns_log.banner()

    # -- Step 1: Artifact & Registry Validation --------------------------------
    ns_log.section("ARTIFACT & REGISTRY INTEGRITY CHECK")
    should_benchmark = False

    if not all_artifacts_valid(MODEL_ROOT):
        if not validate_registry_artifact(MODEL_ROOT):
            try:
                bench = load_benchmark_report(MODEL_ROOT)
                ns_log.warn("Registry missing or outdated; attempting auto-rebuild from benchmark results...")
                df_raw = load_dataset(DATASET_FILENAME, TARGET_COLUMN)
                build_registry(bench, dataset_df=df_raw, version="v2.1", folder=MODEL_ROOT)
                ns_log.success("Model registry successfully reconstructed from benchmark results")
            except Exception as reg_exc:
                ns_log.warn(f"Registry rebuild failed: {reg_exc}; triggering full benchmark")
                should_benchmark = True
        else:
            ns_log.warn("Core artifacts missing or invalid; full benchmark required")
            should_benchmark = True
    else:
        ns_log.success("All artifacts and model registry verified")

    # -- Step 2a: FAST PATH -- Load Champion from Registry --------------------
    if not should_benchmark:
        ns_log.section("LOADING ACTIVE CHAMPION & REGISTRY")
        try:
            pipeline         = load_pipeline(MODEL_ROOT)
            feature_columns  = load_feature_columns(MODEL_ROOT)
            metadata         = load_metadata(MODEL_ROOT)
            model_registry   = load_model_registry(MODEL_ROOT)
            promotion_history = load_promotion_history(MODEL_ROOT)
            benchmark_report = load_benchmark_report(MODEL_ROOT)
            version_history  = load_version_history(MODEL_ROOT)

            champion_meta = model_registry.get("champion", {})
            champion_name = champion_meta.get("name") or metadata.get("best_model_name", "catboost")
            champion_model = load_champion_model(MODEL_ROOT)

            # Target Semantics Assertion & Validation
            validate_target_semantics_or_raise(metadata, champion_model)

            _set_global_state(
                pipeline=pipeline,
                trained_models={"best_model": champion_model, champion_name: champion_model},
                feature_columns=feature_columns,
                best_model_name=champion_name,
                metadata=metadata,
                version_history=version_history,
                model_registry=model_registry,
                promotion_history=promotion_history,
                X_test_transformed=None,
                y_test=None,
                X_test_df=None,
                benchmark_report=benchmark_report,
            )

            # Initialize Multi-Layered Runtime Cache (Warm Startup)
            runtime_cache.initialize_runtime_cache(folder=MODEL_ROOT)

            _set_training_status("ready", "idle")

            elapsed = round(time.perf_counter() - startup_start, 2)
            ns_log.registry_summary(
                champion=champion_meta,
                challenger=model_registry.get("challenger"),
                status="healthy",
                promotion_count=len(promotion_history),
                active_version=model_registry.get("active_version", "v2.1"),
            )
            ns_log.system_ready(
                dataset_size=metadata.get("dataset_size", 0),
                feature_count=metadata.get("raw_feature_count", len(feature_columns)),
                duration_s=elapsed,
            )
            return

        except Exception as exc:
            ns_log.warn(f"Fast load failed: {exc}; falling back to autonomous benchmark")
            should_benchmark = True

    # -- Step 2b: BENCHMARK & REGISTRY INITIALIZATION PATH --------------------
    if should_benchmark:
        ns_log.section("AUTONOMOUS BENCHMARK & REGISTRY INITIALIZATION")
        _set_training_status("training", "initializing")

        try:
            ns_log.step("Loading dataset...")
            _set_training_status("training", "preprocessing")
            df_raw = load_dataset(DATASET_FILENAME, TARGET_COLUMN)
            ns_log.info("Dataset loaded", f"({df_raw.shape[0]} rows, {df_raw.shape[1]} columns)")

            cols_to_drop = ["charges", "steps", TARGET_COLUMN, "insuranceclaim"]
            df = df_raw.drop(columns=[c for c in cols_to_drop if c in df_raw.columns and c != "charges"])

            # Split raw features before computing target threshold
            df_features = df.drop(columns=["charges"], errors="ignore")
            charges_raw = df_raw["charges"].values

            from sklearn.model_selection import train_test_split
            train_idx, test_idx = train_test_split(
                np.arange(len(df_raw)),
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
            )

            # Compute threshold strictly from training split
            train_charges = charges_raw[train_idx]
            frozen_median_threshold = float(np.median(train_charges))
            ns_log.info(
                "Target median calculated from TRAIN SPLIT ONLY",
                f"threshold=${frozen_median_threshold:,.2f}",
            )

            y_all = (charges_raw < frozen_median_threshold).astype(int)
            y_train = y_all[train_idx]
            y_test  = y_all[test_idx]

            X_train_raw = df_features.iloc[train_idx].reset_index(drop=True)
            X_test_raw  = df_features.iloc[test_idx].reset_index(drop=True)

            ns_log.step("Fitting preprocessing pipeline on training fold...")
            pipeline, X_train_t, X_test_t = fit_preprocessing_pipeline(X_train_raw, X_test_raw)
            transformed_cols = list(pipeline.get_feature_names_out())

            raw_feature_count = X_train_raw.shape[1]
            transformed_feature_count = X_train_t.shape[1]

            ns_log.info(
                "Features schemas",
                f"raw={raw_feature_count} columns -> transformed={transformed_feature_count} columns",
            )

            # Run 5-model benchmark
            _set_training_status("training", "evaluating")
            benchmark_report, model_objects = run_benchmark(
                X_train_t=X_train_t,
                X_test_t=X_test_t,
                X_test_raw=X_test_raw,
                y_train=y_train,
                y_test=y_test,
                pipeline=pipeline,
                X_raw_full=df_features,
                y_full=y_all,
                raw_feature_count=raw_feature_count,
                random_state=RANDOM_STATE,
                ns_log=ns_log,
            )

            target_definition_dict = {
                "positive_class": 1,
                "positive_class_name": "LOW_RISK_APPROVAL",
                "negative_class": 0,
                "negative_class_name": "HIGH_RISK_REJECTION",
                "threshold_source": "training_split_median",
                "target_formula": "charges < training_median",
            }

            benchmark_report["target_threshold"] = frozen_median_threshold
            benchmark_report["target_definition"] = target_definition_dict
            benchmark_report["target_source_split"] = "train_only"
            benchmark_report["raw_feature_count"] = raw_feature_count
            benchmark_report["transformed_feature_count"] = transformed_feature_count
            benchmark_report["raw_feature_names"] = list(df_features.columns)
            benchmark_report["transformed_feature_names"] = list(transformed_cols)

            # Persist core artifacts
            ns_log.section("SAVING ARTIFACTS & INITIALIZING REGISTRY")
            MODEL_ROOT.mkdir(parents=True, exist_ok=True)
            (MODEL_ROOT / "artifacts").mkdir(parents=True, exist_ok=True)

            champion_name = benchmark_report["champion"]
            champ_entry = model_objects[champion_name]
            champion_model = champ_entry.get("model_object", champ_entry) if isinstance(champ_entry, dict) else champ_entry

            # Save preprocessing and schemas
            joblib.dump(pipeline, MODEL_ROOT / "preprocessing_pipeline.pkl")
            ns_log.success("Saved preprocessing_pipeline.pkl")

            with open(MODEL_ROOT / "feature_columns.json", "w", encoding="utf-8") as f:
                json.dump(list(df_features.columns), f, indent=2)
            ns_log.success("Saved feature_columns.json")

            with open(MODEL_ROOT / "raw_feature_schema.json", "w", encoding="utf-8") as f:
                json.dump({
                    "raw_features": list(df_features.columns),
                    "raw_feature_count": raw_feature_count,
                    "transformed_feature_count": transformed_feature_count,
                    "target": TARGET_COLUMN,
                    "target_threshold": frozen_median_threshold,
                    "target_definition": target_definition_dict,
                    "target_source_split": "train_only",
                }, f, indent=2)
            ns_log.success("Saved raw_feature_schema.json")

            save_benchmark_report(benchmark_report, MODEL_ROOT)
            ns_log.success("Saved benchmark_results.json")

            # Build Model Registry & Immutable Artifacts
            model_reg = build_registry(
                benchmark_report=benchmark_report,
                model_objects=model_objects,
                dataset_df=df_raw,
                version="v2.1",
                folder=MODEL_ROOT,
            )
            ns_log.success("Initialized model_registry.json & immutable model repository")

            # Save metadata
            champ_res = benchmark_report["results"][champion_name]
            meta = {
                "best_model_name": champion_name,
                "model_version": "v2.1",
                "training_timestamp": datetime.now(timezone.utc).isoformat(),
                "dataset_size": df_raw.shape[0],
                "raw_feature_count": raw_feature_count,
                "transformed_feature_count": transformed_feature_count,
                "accuracy": champ_res.get("accuracy"),
                "precision": champ_res.get("precision"),
                "recall": champ_res.get("recall"),
                "f1_score": champ_res.get("f1_score"),
                "roc_auc": champ_res.get("roc_auc"),
                "confusion_matrix": champ_res.get("confusion_matrix"),
                "champion": champion_name,
                "runner_up": benchmark_report.get("runner_up"),
                "ranking": benchmark_report.get("ranking"),
                "benchmark_timestamp": benchmark_report.get("benchmark_timestamp"),
                "benchmark_duration_s": benchmark_report.get("benchmark_duration_s"),
                "cv_f1_mean": champ_res.get("cv_f1_mean"),
                "cv_f1_std": champ_res.get("cv_f1_std"),
                "model_only_latency_ms": champ_res.get("model_only_latency_ms"),
                "end_to_end_latency_ms": champ_res.get("end_to_end_latency_ms"),
                "model_size_mb": champ_res.get("model_size_mb"),
                "selection_policy": benchmark_report.get("selection_policy"),
                "target_threshold": frozen_median_threshold,
                "target_definition": target_definition_dict,
                "target_source_split": "train_only",
            }
            save_metadata_v2(meta, MODEL_ROOT)
            ns_log.success("Saved metadata.json")

            append_version_history_v2({
                "version": "v2.1",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "best_model": champion_name,
                "accuracy": champ_res.get("accuracy"),
                "f1_score": champ_res.get("f1_score"),
                "roc_auc": champ_res.get("roc_auc"),
                "trigger": "benchmark_initialization",
            }, MODEL_ROOT)

            trained_models_dict = {name: model_objects[name] for name in model_objects}
            trained_models_dict["best_model"] = champion_model

            # Target Semantics Assertion & Validation
            validate_target_semantics_or_raise(meta, champion_model)

            _set_global_state(
                pipeline=pipeline,
                trained_models=trained_models_dict,
                feature_columns=list(df_features.columns),
                best_model_name=champion_name,
                metadata=meta,
                version_history=load_version_history(MODEL_ROOT),
                model_registry=model_reg,
                promotion_history=load_promotion_history(MODEL_ROOT),
                X_test_transformed=X_test_t,
                y_test=y_test,
                X_test_df=X_test_raw,
                benchmark_report=benchmark_report,
            )

            # Initialize Multi-Layered Runtime Cache (Warm Startup)
            runtime_cache.initialize_runtime_cache(folder=MODEL_ROOT)

            _set_training_status("ready", "idle")

            elapsed = round(time.perf_counter() - startup_start, 2)
            ns_log.registry_summary(
                champion=model_reg.get("champion", {}),
                challenger=model_reg.get("challenger"),
                status="healthy",
                promotion_count=len(load_promotion_history(MODEL_ROOT)),
                active_version="v2.1",
            )
            ns_log.system_ready(
                dataset_size=df_features.shape[0],
                feature_count=raw_feature_count,
                duration_s=elapsed,
            )

        except Exception as exc:
            _set_training_status("degraded", "idle")
            ns_log.error(f"Startup benchmark failed: {exc}")
            _logger.exception("Startup benchmark failed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
