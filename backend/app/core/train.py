import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from app.core.evaluate import evaluate_model

logger = logging.getLogger(__name__)

# ─── Path Helpers ──────────────────────────────────────────────────────────────

def get_model_root_folder() -> Path:
    """Return the root model folder under backend/model."""
    current_file = Path(__file__).resolve()
    return current_file.parents[3] / "model"


def get_artifacts_folder() -> Path:
    """Return the artifacts folder path for saving trained model variants."""
    return get_model_root_folder() / "artifacts"


def get_model_file_path(name: str = "best_model", folder: Path = None, extension: str = "pkl") -> Path:
    folder = folder or get_model_root_folder()
    return folder / f"{name}.{extension}"


def get_metadata_path(folder: Path = None, filename: str = "metadata.json") -> Path:
    folder = folder or get_model_root_folder()
    return folder / filename


def get_feature_columns_path(folder: Path = None, filename: str = "feature_columns.json") -> Path:
    folder = folder or get_model_root_folder()
    return folder / filename


def get_version_history_path(folder: Path = None) -> Path:
    folder = folder or get_model_root_folder()
    return folder / "version_history.json"


# ─── Artifact I/O ─────────────────────────────────────────────────────────────

def save_feature_columns(feature_columns: List[str], folder: Path = None, filename: str = "feature_columns.json") -> Path:
    folder = folder or get_model_root_folder()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    logger.info("Saving feature columns metadata to %s", path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(feature_columns, handle)
    return path


def load_feature_columns(folder: Path = None, filename: str = "feature_columns.json") -> List[str]:
    folder = folder or get_model_root_folder()
    path = folder / filename
    if not path.exists():
        raise FileNotFoundError(f"Feature columns metadata not found at {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_metadata(metadata: Dict[str, Any], folder: Path = None, filename: str = "metadata.json") -> Path:
    """Save extended metadata including all ML metrics, timestamps, and version info."""
    folder = folder or get_model_root_folder()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    logger.info("Saving model metadata to %s", path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    return path


def load_metadata(folder: Path = None, filename: str = "metadata.json") -> Dict[str, Any]:
    folder = folder or get_model_root_folder()
    path = folder / filename
    if not path.exists():
        raise FileNotFoundError(f"Model metadata not found at {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_model(name: str = "best_model", folder: Path = None, extension: str = "pkl") -> Any:
    path = get_model_file_path(name=name, folder=folder, extension=extension)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")
    logger.info("Loading model %s from %s", name, path)
    return joblib.load(path)


def load_pipeline(folder: Path = None, filename: str = "preprocessing_pipeline.pkl") -> Any:
    folder = folder or get_model_root_folder()
    path = folder / filename
    if not path.exists():
        raise FileNotFoundError(f"Pipeline file not found at {path}")
    logger.info("Loading preprocessing pipeline from %s", path)
    return joblib.load(path)


# ─── Version History ──────────────────────────────────────────────────────────

def load_version_history(folder: Path = None) -> List[Dict[str, Any]]:
    """Load the model version history (retraining audit trail)."""
    path = get_version_history_path(folder)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def append_version_history(entry: Dict[str, Any], folder: Path = None) -> None:
    """Append a new version record to the version history file."""
    folder = folder or get_model_root_folder()
    folder.mkdir(parents=True, exist_ok=True)
    history = load_version_history(folder)
    history.append(entry)
    path = get_version_history_path(folder)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)
    logger.info("Appended version record %s to history", entry.get("version"))


# ─── Model Building ───────────────────────────────────────────────────────────

def build_classifiers(random_state: int = 42) -> Dict[str, Any]:
    """
    Create the classifier objects for training.

    Logistic Regression is a strong, interpretable baseline.
    Random Forest handles nonlinear feature interactions in tabular claim data.
    """
    return {
        "logistic_regression": LogisticRegression(
            solver="liblinear", max_iter=1000, random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        ),
    }


def train_models(X_train, y_train, random_state: int = 42) -> Dict[str, Any]:
    """Train both classifier models and return them."""
    models = build_classifiers(random_state=random_state)
    for model_name, model in models.items():
        logger.info("Training %s", model_name)
        model.fit(X_train, y_train)
        logger.info("Finished training %s", model_name)
    return models


def save_model(model: Any, name: str, folder: Path = None, extension: str = "joblib") -> Path:
    """Save a trained model object to disk."""
    folder = folder or get_artifacts_folder()
    folder.mkdir(parents=True, exist_ok=True)
    model_path = folder / f"{name}.{extension}"
    logger.info("Saving model %s to %s", name, model_path)
    joblib.dump(model, model_path)
    return model_path


def save_pipeline(pipeline: Any, folder: Path = None, filename: str = "preprocessing_pipeline.pkl") -> Path:
    """Save the preprocessing pipeline used for feature transformation."""
    folder = folder or get_model_root_folder()
    folder.mkdir(parents=True, exist_ok=True)
    pipeline_path = folder / filename
    logger.info("Saving preprocessing pipeline to %s", pipeline_path)
    joblib.dump(pipeline, pipeline_path)
    return pipeline_path


def select_best_model(metrics: Dict[str, Dict[str, Any]]) -> str:
    """Choose the best model using F1-score with accuracy as a tie breaker."""
    best_name = None
    best_f1 = -1.0
    best_accuracy = -1.0

    for model_name, values in metrics.items():
        f1 = float(values.get("f1_score", 0.0))
        accuracy = float(values.get("accuracy", 0.0))
        if f1 > best_f1 or (f1 == best_f1 and accuracy > best_accuracy):
            best_name = model_name
            best_f1 = f1
            best_accuracy = accuracy

    if best_name is None:
        raise RuntimeError("Could not select a best model because no metrics were available.")

    logger.info(
        "Selected best model '%s' with f1_score=%s and accuracy=%s",
        best_name, best_f1, best_accuracy,
    )
    return best_name


# ─── Full Training Orchestrator ───────────────────────────────────────────────

def train_and_select_best_model(
    X_train, X_test, y_train, y_test, random_state: int = 42
) -> Dict[str, Any]:
    """
    Train both models, compare on standard metrics, save the best, and return
    a full result dict including per-model metrics and timing.
    """
    training_start = time.perf_counter()

    models = train_models(X_train, y_train, random_state=random_state)
    metrics: Dict[str, Dict[str, Any]] = {}

    for model_name, model in models.items():
        metrics[model_name] = evaluate_model(model, X_test, y_test)

    best_model_name = select_best_model(metrics)
    best_model = models[best_model_name]

    # ── Inference latency sample (average over 50 predictions on test set) ─────
    import numpy as np
    sample_size = min(50, X_test.shape[0])
    sample_X = X_test[:sample_size] if isinstance(X_test, np.ndarray) else X_test[:sample_size]
    lat_start = time.perf_counter()
    for _ in range(sample_size):
        best_model.predict(sample_X[:1])
    lat_end = time.perf_counter()
    avg_latency_ms = round(((lat_end - lat_start) / sample_size) * 1000, 2)

    training_end = time.perf_counter()
    duration_s = round(training_end - training_start, 2)

    # ── Persist best model ────────────────────────────────────────────────────
    save_model(best_model, "best_model", folder=get_model_root_folder(), extension="pkl")
    for model_name, model in models.items():
        save_model(model, model_name)

    return {
        "models": models,
        "metrics": metrics,
        "best_model_name": best_model_name,
        "inference_latency_ms": avg_latency_ms,
        "training_duration_s": duration_s,
    }


# ─── Artifact Validation ──────────────────────────────────────────────────────

REQUIRED_ARTIFACTS = [
    "best_model.pkl",
    "preprocessing_pipeline.pkl",
    "metadata.json",
    "feature_columns.json",
]


def artifacts_exist(folder: Optional[Path] = None) -> bool:
    """Return True only if all required model artifacts are present."""
    root = folder or get_model_root_folder()
    for artifact in REQUIRED_ARTIFACTS:
        if not (root / artifact).exists():
            logger.info("Missing artifact: %s", artifact)
            return False
    return True


def validate_metadata(folder: Optional[Path] = None) -> bool:
    """
    Validate that metadata.json contains the required fields.
    Returns False if the file is missing, corrupted, or incomplete.
    """
    try:
        meta = load_metadata(folder)
        required_keys = {"best_model_name", "feature_columns", "trained_models", "accuracy"}
        missing = required_keys - set(meta.keys())
        if missing:
            logger.warning("Metadata missing required keys: %s", missing)
            return False
        return True
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        logger.warning("Metadata validation failed: %s", exc)
        return False
