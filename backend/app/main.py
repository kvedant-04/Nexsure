"""
Nexsure AI Engine — Application Entry Point

Implements a production-safe startup lifecycle:
  1. Check if all model artifacts exist and metadata is valid.
  2. If valid: load artifacts (fast startup).
  3. If missing or corrupted: auto-train the full ML pipeline.

Manual training via API is never required.
"""

import logging
import time
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.predict import router as predict_router
from app.core import logger as ns_log  # enterprise terminal logger

logging.basicConfig(
    level=logging.WARNING,  # suppress noisy library logs; structured output via ns_log
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
# Our own modules can remain at INFO level
logging.getLogger("app").setLevel(logging.INFO)

_logger = logging.getLogger(__name__)

app = FastAPI(
    title="Nexsure API",
    description=(
        "Production-grade AI health insurance risk assessment platform. "
        "Autonomous ML lifecycle, SHAP explainability, and real-time observability."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api")


@app.on_event("startup")
async def startup_lifecycle():
    """
    Autonomous model lifecycle management.

    Checks artifact integrity before deciding whether to load or train.
    This guarantees the backend is ALWAYS ready after startup — no manual
    API calls required.
    """
    startup_start = time.perf_counter()

    # ── Lazy imports to avoid circular dependencies ────────────────────────────
    from app.api.predict import (
        _set_global_state,
        _get_training_status,
        _set_training_status,
    )
    from app.core.train import (
        artifacts_exist,
        validate_metadata,
        load_metadata,
        load_pipeline,
        load_model,
        load_feature_columns,
        load_version_history,
        get_model_root_folder,
        get_artifacts_folder,
        save_metadata,
        save_pipeline,
        save_feature_columns,
        append_version_history,
        train_and_select_best_model,
    )
    from app.core.data_loader import load_dataset
    from app.core.preprocess import fit_preprocessing_pipeline, split_features_target
    from app.core.evaluate import evaluate_models
    import joblib

    MODEL_ROOT = get_model_root_folder()
    ARTIFACTS_ROOT = get_artifacts_folder()

    ns_log.banner()

    # ── Step 1: Check artifacts ────────────────────────────────────────────────
    ns_log.section("ARTIFACT INTEGRITY CHECK")

    should_train = False
    if not artifacts_exist(MODEL_ROOT):
        ns_log.warn("One or more required artifacts are missing — auto-training required")
        should_train = True
    elif not validate_metadata(MODEL_ROOT):
        ns_log.warn("Metadata validation failed — auto-training required")
        should_train = True
    else:
        ns_log.success("All artifacts present and valid")

    # ── Step 2: Load OR Train ─────────────────────────────────────────────────
    if not should_train:
        # ── FAST PATH: Load existing artifacts ──────────────────────────────
        ns_log.section("LOADING SAVED ARTIFACTS")
        try:
            pipeline = load_pipeline(MODEL_ROOT)
            ns_log.artifact_loaded("preprocessing_pipeline.pkl", str(MODEL_ROOT))

            feature_columns = load_feature_columns(MODEL_ROOT)
            ns_log.artifact_loaded("feature_columns.json", str(MODEL_ROOT))

            metadata = load_metadata(MODEL_ROOT)
            best_model_name = metadata.get("best_model_name")
            ns_log.artifact_loaded("metadata.json", str(MODEL_ROOT))

            # Load best model
            best_model = load_model("best_model", folder=MODEL_ROOT, extension="pkl")
            ns_log.artifact_loaded("best_model.pkl", str(MODEL_ROOT))

            # Load all model variants
            trained_models = {"best_model": best_model}
            for mname in metadata.get("trained_models", []):
                try:
                    trained_models[mname] = load_model(mname, folder=ARTIFACTS_ROOT, extension="joblib")
                except FileNotFoundError:
                    pass

            # If best_model_name key is in trained_models, also map by real name
            if best_model_name and best_model_name not in trained_models:
                trained_models[best_model_name] = best_model

            version_history = load_version_history(MODEL_ROOT)

            _set_global_state(
                pipeline=pipeline,
                trained_models=trained_models,
                feature_columns=feature_columns,
                best_model_name=best_model_name,
                metadata=metadata,
                version_history=version_history,
                X_test_transformed=None,
                y_test=None,
                X_test_df=None,
            )
            _set_training_status("ready", "idle")

            ns_log.success("All artifacts loaded successfully")
            elapsed = round(time.perf_counter() - startup_start, 2)
            ns_log.system_ready(
                dataset_size=metadata.get("dataset_size", 0),
                feature_count=len(feature_columns),
                duration_s=elapsed,
            )
            return

        except Exception as exc:
            ns_log.warn(f"Artifact loading failed: {exc} — falling back to auto-training")
            should_train = True

    if should_train:
        # ── AUTO-TRAIN PATH ──────────────────────────────────────────────────
        ns_log.section("AUTO-TRAINING PIPELINE")
        _set_training_status("training", "initializing")

        DATASET_FILENAME = "insurance3r2.csv"
        TARGET_COLUMN = "insuranceclaim"
        TEST_SIZE = 0.20
        RANDOM_STATE = 42

        try:
            # ── Load Dataset ──────────────────────────────────────────────
            ns_log.step("Loading dataset...")
            _set_training_status("training", "preprocessing")
            df = load_dataset(DATASET_FILENAME, TARGET_COLUMN)
            ns_log.info("Dataset loaded", f"({df.shape[0]} rows × {df.shape[1]} columns)")

            # ── Feature Engineering ───────────────────────────────────────
            ns_log.step("Engineering target variable from charges median...")
            threshold = df["charges"].median()
            df["target"] = (df["charges"] < threshold).astype(int)
            ns_log.info("Target threshold (charges median)", f"${threshold:,.2f}")

            cols_to_drop = ["charges", "steps"]
            if TARGET_COLUMN in df.columns:
                cols_to_drop.append(TARGET_COLUMN)
            if "insuranceclaim" in df.columns and "insuranceclaim" not in cols_to_drop:
                cols_to_drop.append("insuranceclaim")
            df = df.drop(columns=cols_to_drop, errors="ignore")
            ns_log.info("Dropped leakage columns", str([c for c in cols_to_drop if c != "target"]))

            # ── Train/Test Split ──────────────────────────────────────────
            ns_log.step("Splitting train/test sets...")
            X_train, X_test, y_train, y_test = split_features_target(
                df, target_column="target", test_size=TEST_SIZE, random_state=RANDOM_STATE
            )
            feature_columns = X_train.columns.tolist()
            ns_log.info("Train size", str(X_train.shape[0]))
            ns_log.info("Test  size", str(X_test.shape[0]))
            ns_log.info("Features ", str(feature_columns))

            # ── Preprocessing ─────────────────────────────────────────────
            ns_log.step("Fitting preprocessing pipeline...")
            _set_training_status("training", "preprocessing")
            pipeline, X_train_t, X_test_t = fit_preprocessing_pipeline(X_train, X_test)
            ns_log.success("Preprocessing pipeline fitted")

            # ── Training Models ───────────────────────────────────────────
            ns_log.section("TRAINING MODELS")
            _set_training_status("training", "training")
            train_result = train_and_select_best_model(
                X_train_t, X_test_t, y_train, y_test, random_state=RANDOM_STATE
            )
            trained_models = train_result["models"]
            all_metrics = train_result["metrics"]
            best_model_name = train_result["best_model_name"]
            inference_latency_ms = train_result["inference_latency_ms"]
            training_duration_s = train_result["training_duration_s"]

            for mname in trained_models:
                ns_log.success(f"Trained: {mname}")

            # ── Evaluation ────────────────────────────────────────────────
            ns_log.section("EVALUATING MODELS")
            _set_training_status("training", "evaluating")
            ns_log.model_table(all_metrics)

            # ── Best Model Selection ──────────────────────────────────────
            _set_training_status("training", "selecting")
            best_metrics = all_metrics[best_model_name]
            ns_log.best_model_announcement(best_model_name, best_metrics)

            # ── Save Artifacts ────────────────────────────────────────────
            ns_log.section("SAVING ARTIFACTS")
            _set_training_status("training", "saving")

            save_pipeline(pipeline)
            ns_log.success("Saved preprocessing_pipeline.pkl")

            save_feature_columns(feature_columns)
            ns_log.success("Saved feature_columns.json")

            # Build comprehensive metadata
            now_iso = datetime.now(timezone.utc).isoformat()
            # Derive version number from history
            history = load_version_history(MODEL_ROOT)
            version_num = len(history) + 1
            version_tag = f"v{version_num}.0"

            full_metadata = {
                "best_model_name": best_model_name,
                "feature_columns": feature_columns,
                "trained_models": list(trained_models.keys()),
                "model_version": version_tag,
                "training_timestamp": now_iso,
                "dataset_size": df.shape[0],
                "feature_count": len(feature_columns),
                "training_duration_s": training_duration_s,
                "inference_latency_ms": inference_latency_ms,
                "model_health": "healthy",
                # Best model metrics (flat for quick access)
                "accuracy": best_metrics.get("accuracy"),
                "precision": best_metrics.get("precision"),
                "recall": best_metrics.get("recall"),
                "f1_score": best_metrics.get("f1_score"),
                "roc_auc": best_metrics.get("roc_auc"),
                "confusion_matrix": best_metrics.get("confusion_matrix"),
                # All model metrics (for comparison)
                "all_model_metrics": all_metrics,
            }
            save_metadata(full_metadata)
            ns_log.success("Saved metadata.json (with full metrics)")

            # Append to version history
            version_entry = {
                "version": version_tag,
                "timestamp": now_iso,
                "model": best_model_name,
                "accuracy": best_metrics.get("accuracy"),
                "f1_score": best_metrics.get("f1_score"),
                "dataset_size": df.shape[0],
                "training_duration_s": training_duration_s,
            }
            append_version_history(version_entry)
            ns_log.success(f"Recorded version {version_tag} in version_history.json")

            # Map best_model_name so prediction works by both keys
            trained_models["best_model"] = trained_models.get(best_model_name)
            if best_model_name not in trained_models:
                trained_models[best_model_name] = trained_models["best_model"]

            version_history = load_version_history(MODEL_ROOT)

            _set_global_state(
                pipeline=pipeline,
                trained_models=trained_models,
                feature_columns=feature_columns,
                best_model_name=best_model_name,
                metadata=full_metadata,
                version_history=version_history,
                X_test_transformed=X_test_t,
                y_test=y_test,
                X_test_df=X_test,
            )
            _set_training_status("ready", "idle")

            elapsed = round(time.perf_counter() - startup_start, 2)
            ns_log.system_ready(
                dataset_size=df.shape[0],
                feature_count=len(feature_columns),
                duration_s=elapsed,
            )

        except Exception as exc:
            _set_training_status("degraded", "idle")
            ns_log.error(f"Auto-training failed: {exc}")
            _logger.exception("Startup training failed with exception")
