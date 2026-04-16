import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from app.core.evaluate import evaluate_model

logger = logging.getLogger(__name__)


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
    folder = folder or get_model_root_folder()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    logger.info("Saving model metadata to %s", path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle)
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


def build_classifiers(random_state: int = 42) -> Dict[str, Any]:
    """Create the classifier objects used for training.

    Logistic Regression is a strong baseline for binary insurance decisions because it is
    interpretable and fast. Random Forest is added for robustness and nonlinear feature
    interactions, which often improve classification on tabular claim data.
    """
    return {
        "logistic_regression": LogisticRegression(
            solver="liblinear", max_iter=1000, random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def train_models(X_train, y_train, random_state: int = 42) -> Dict[str, Any]:
    """Train both logistic regression and random forest models."""
    models = build_classifiers(random_state=random_state)

    for model_name, model in models.items():
        logger.info("Training %s", model_name)
        model.fit(X_train, y_train)
        logger.info("Finished training %s", model_name)

    return models


def save_model(model: Any, name: str, folder: Path = None, extension: str = "joblib") -> Path:
    """Save a trained model object to disk with a configurable extension."""
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
        best_name,
        best_f1,
        best_accuracy,
    )
    return best_name


def train_and_select_best_model(
    X_train, X_test, y_train, y_test, random_state: int = 42
) -> Dict[str, Any]:
    """Train both models, compare them on standard metrics, and save the best model."""
    models = train_models(X_train, y_train, random_state=random_state)
    metrics = {}

    for model_name, model in models.items():
        metrics[model_name] = evaluate_model(model, X_test, y_test)

    best_model_name = select_best_model(metrics)
    best_model = models[best_model_name]

    # Save the selected best model to the required backend/model path.
    save_model(best_model, "best_model", folder=get_model_root_folder(), extension="pkl")
    # Also preserve each trained model variant in artifacts for future analysis.
    for model_name, model in models.items():
        save_model(model, model_name)

    return {
        "models": models,
        "metrics": metrics,
        "best_model_name": best_model_name,
    }
