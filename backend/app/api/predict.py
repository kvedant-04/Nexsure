import logging
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.data_loader import load_dataset
from app.core.evaluate import evaluate_models
from app.core.explain import explain_local_prediction, get_global_feature_importance
from app.core.preprocess import fit_preprocessing_pipeline, split_features_target
from app.core.train import (
    save_feature_columns,
    save_metadata,
    save_pipeline,
    train_and_select_best_model,
    load_model,
    load_pipeline,
    load_metadata,
    get_model_root_folder,
    get_artifacts_folder,
)

logger = logging.getLogger(__name__)
router = APIRouter()

MODEL_ROOT = get_model_root_folder()
ARTIFACTS_ROOT = get_artifacts_folder()
PIPELINE_FILENAME = "preprocessing_pipeline.pkl"
METADATA_FILENAME = "metadata.json"


def _load_saved_artifacts() -> None:
    global pipeline, trained_models, feature_columns, best_model_name

    if pipeline is not None and trained_models and feature_columns is not None:
        return

    logger.info("Attempting to load saved model artifacts from disk")

    if not MODEL_ROOT.exists():
        logger.info("Model root directory does not exist yet: %s", MODEL_ROOT)
        return

    if (MODEL_ROOT / PIPELINE_FILENAME).exists():
        try:
            pipeline = load_pipeline(folder=MODEL_ROOT, filename=PIPELINE_FILENAME)
        except FileNotFoundError:
            logger.warning("Preprocessing pipeline not found during artifact load")

    if (MODEL_ROOT / METADATA_FILENAME).exists():
        try:
            metadata = load_metadata(folder=MODEL_ROOT, filename=METADATA_FILENAME)
            feature_columns = metadata.get("feature_columns")
            best_model_name = metadata.get("best_model_name")
            trained_model_names = metadata.get("trained_models", [])
        except FileNotFoundError:
            logger.warning("Model metadata failed to load from %s", MODEL_ROOT / METADATA_FILENAME)
            trained_model_names = []
    else:
        trained_model_names = []

    trained_models = {}
    if trained_model_names and ARTIFACTS_ROOT.exists():
        for model_name in trained_model_names:
            try:
                trained_models[model_name] = load_model(model_name, folder=ARTIFACTS_ROOT, extension="joblib")
            except FileNotFoundError:
                logger.warning("Saved model %s not found in artifacts folder", model_name)

    if not trained_models and ARTIFACTS_ROOT.exists():
        for model_path in ARTIFACTS_ROOT.glob("*.joblib"):
            try:
                trained_models[model_path.stem] = joblib.load(model_path)
            except Exception as exc:
                logger.warning("Unable to load artifact model %s: %s", model_path, exc)

    best_model_path = MODEL_ROOT / "best_model.pkl"
    if best_model_path.exists() and "best_model" not in trained_models:
        try:
            trained_models["best_model"] = load_model("best_model", folder=MODEL_ROOT, extension="pkl")
        except FileNotFoundError:
            logger.warning("Best model file found but failed to load: %s", best_model_path)

    if feature_columns is None and pipeline is not None and hasattr(pipeline, "feature_names_in_"):
        feature_columns = list(pipeline.feature_names_in_)


@router.get("/health")
def health_check() -> dict:
    """Simple health endpoint to verify the service is running."""
    logger.info("Health check requested")
    return {"status": "ok"}

# Global state is intentionally kept simple for this scaffold.
pipeline = None
trained_models: Dict[str, Any] = {}
feature_columns: Optional[list] = None
X_test_transformed = None
y_test = None
X_test_df = None
best_model_name: Optional[str] = None
prediction_logs: List[Dict[str, Any]] = []
training_history: Dict[str, List[float]] = {"epochs": [], "accuracy": []}


class TrainRequest(BaseModel):
    dataset_filename: str = Field(..., description="CSV filename stored in backend/data/")
    target_column: str = Field(..., description="Name of the target column in the dataset")
    test_size: float = Field(0.20, gt=0.0, lt=0.5, description="Fraction of data to reserve for testing")
    random_state: int = Field(42, description="Random seed for reproducible splits")


class TrainResponse(BaseModel):
    models_trained: Dict[str, str]
    best_model: str
    dataset_rows: int
    dataset_columns: int


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Feature values for a single claim example")
    model_name: str = Field("logistic_regression", description="Model to use for prediction")


class PredictResponse(BaseModel):
    model_name: str
    prediction: str
    confidence: float
    label: str
    top_features: List[Dict[str, Any]]
    explanation: str
    feature_importance: Optional[Dict[str, float]] = None


class GlobalExplanationResponse(BaseModel):
    top_features: List[Dict[str, Any]]
    feature_importance: Optional[Dict[str, float]] = None


class TrainStatusResponse(BaseModel):
    best_model: str
    accuracy: Optional[float] = None


class EvaluateResponse(BaseModel):
    evaluations: Dict[str, Dict[str, Any]]


@router.post("/train", response_model=TrainResponse)
def train_model(request: TrainRequest) -> TrainResponse:
    """Train machine learning models using the provided dataset and return training results."""
    global pipeline, trained_models, feature_columns, X_test_transformed, y_test, X_test_df, best_model_name

    try:
        # Load and validate dataset
        df = load_dataset(request.dataset_filename, request.target_column)

        # Split data
        X_train, X_test, y_train, y_test_local = split_features_target(
            df,
            target_column=request.target_column,
            test_size=request.test_size,
            random_state=request.random_state,
        )

        # Preprocess data
        pipeline, X_train_transformed, X_test_transformed_local = fit_preprocessing_pipeline(
            X_train, X_test
        )

        # Train models and select best
        logger.info("Starting training for all models")
        train_result = train_and_select_best_model(
            X_train_transformed,
            X_test_transformed_local,
            y_train,
            y_test_local,
            random_state=request.random_state,
        )

        # Update global state
        trained_models = train_result["models"]
        best_model_name_local = train_result["best_model_name"]
        save_pipeline(pipeline)
        save_feature_columns(X_train.columns.tolist())
        save_metadata(
            {
                "best_model_name": best_model_name_local,
                "feature_columns": X_train.columns.tolist(),
                "trained_models": list(trained_models.keys()),
            }
        )

        feature_columns = X_train.columns.tolist()
        X_test_transformed = X_test_transformed_local
        y_test = y_test_local
        X_test_df = X_test
        best_model_name = best_model_name_local

        # Mock training history (in a real system, this would come from the training loop)
        global training_history
        training_history = {
            "epochs": [1, 2, 3, 4],
            "accuracy": [0.7, 0.8, 0.85, 0.9]
        }

        logger.info("Training completed successfully for models: %s", list(trained_models.keys()))

        return TrainResponse(
            models_trained={name: "trained" for name in trained_models.keys()},
            best_model=best_model_name,
            dataset_rows=df.shape[0],
            dataset_columns=df.shape[1],
        )

    except FileNotFoundError as exc:
        logger.error("Dataset file not found: %s", exc)
        raise HTTPException(
            status_code=404,
            detail=f"Dataset file not found: {request.dataset_filename}. Please ensure it exists in backend/data/"
        )
    except ValueError as exc:
        logger.error("Invalid input parameters: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input parameters: {str(exc)}"
        )
    except Exception as exc:
        logger.error("Unexpected error during training: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during model training. Please check logs for details."
        )


@router.post("/predict", response_model=PredictResponse)
def predict_claim(request: PredictRequest) -> PredictResponse:
    """Generate a prediction for a single insurance claim with SHAP explanations."""
    _load_saved_artifacts()

    # Validate model availability
    if pipeline is None or not trained_models:
        raise HTTPException(
            status_code=400,
            detail="Models are not trained yet. Please train models first using POST /train."
        )

    # Validate requested model
    if request.model_name not in trained_models:
        available_models = list(trained_models.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model name '{request.model_name}'. Available models: {available_models}"
        )

    # Validate feature columns
    if feature_columns is None:
        raise HTTPException(
            status_code=500,
            detail="Feature column metadata is missing. Please retrain models using POST /train."
        )

    # Validate input features
    features = request.features
    if not features:
        raise HTTPException(
            status_code=400,
            detail="No features provided. Please provide feature values for prediction."
        )

    missing_columns = [col for col in feature_columns if col not in features]
    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {missing_columns}. Required features: {feature_columns}"
        )

    try:
        row = {col: features[col] for col in feature_columns}
        input_df = pd.DataFrame([row], columns=feature_columns)

        print("INPUT FEATURES:", features)  # DEBUG

        transformed = pipeline.transform(input_df)
        model = trained_models[request.model_name]

        # Use predict_proba correctly
        proba = model.predict_proba(transformed)[0]
        print("PROBABILITIES:", proba)  # DEBUG

        # Assuming class 1 = APPROVED
        approval_prob = proba[1]
        print("APPROVAL PROB:", approval_prob)  # DEBUG

        if approval_prob >= 0.5:
            prediction = "APPROVED"
        else:
            prediction = "REJECTED"

        confidence = round(max(proba) * 100, 2)
        print("FINAL:", prediction)  # DEBUG

        top_features, explanation_text = explain_local_prediction(model, pipeline, input_df, top_n=5)

        feature_importance = {
            item["feature"]: float(abs(item.get("shap_value", 0.0)))
            for item in top_features
            if isinstance(item, dict) and "feature" in item
        }

        label = prediction  # prediction is already the label
        explanation = (
            f"This claim was {label} mainly due to {top_features[0]['feature']} "
            f"and {top_features[1]['feature']}."
        ) if len(top_features) > 1 else (
            f"This claim was {label} mainly due to {top_features[0]['feature']}."
        )

        # Store prediction log
        global prediction_logs
        prediction_logs.append({
            "id": len(prediction_logs) + 1,
            "age": int(features.get("age", 0)),
            "result": label,
            "confidence": float(confidence),
        })

        return PredictResponse(
            model_name=request.model_name,
            prediction=prediction,
            confidence=confidence,
            label=label,
            top_features=top_features,
            explanation=explanation,
            feature_importance=feature_importance,
        )

    except Exception as exc:
        logger.error("Error during prediction: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An error occurred during prediction. Please check logs for details."
        )


@router.get("/train-status", response_model=TrainStatusResponse)
def get_train_status() -> TrainStatusResponse:
    _load_saved_artifacts()

    if not trained_models:
        return TrainStatusResponse(best_model="Random Forest", accuracy=None)

    best_name = best_model_name or next(iter(trained_models.keys()), "Random Forest")
    accuracy = None

    if X_test_transformed is not None and y_test is not None:
        try:
            evaluations = evaluate_models(trained_models, X_test_transformed, y_test)
            if best_name in evaluations:
                accuracy = float(evaluations[best_name].get("accuracy", 0.0))
        except Exception:
            accuracy = None

    return TrainStatusResponse(best_model=best_name, accuracy=accuracy)


@router.get("/metrics", response_model=EvaluateResponse)
def get_model_metrics() -> EvaluateResponse:
    """Return performance metrics for all trained models on the test set."""
    _load_saved_artifacts()

    if pipeline is None or not trained_models:
        raise HTTPException(
            status_code=400,
            detail="Models are not trained yet. Please train models first using POST /train."
        )

    if X_test_transformed is None or y_test is None:
        raise HTTPException(
            status_code=500,
            detail="Test data not available. Please retrain models using POST /train.",
        )

    try:
        evaluations = evaluate_models(trained_models, X_test_transformed, y_test)
        return EvaluateResponse(evaluations=evaluations)
    except Exception as exc:
        logger.error("Error evaluating models: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to evaluate models. Please check logs for details."
        )


@router.get("/explain", response_model=GlobalExplanationResponse)
def get_shap_explanation() -> GlobalExplanationResponse:
    """Return global SHAP feature importance for the best trained model."""
    _load_saved_artifacts()

    if pipeline is None or not trained_models or X_test_df is None:
        raise HTTPException(
            status_code=400,
            detail="Models are not trained yet. Please train models first using POST /train."
        )

    if best_model_name is None or best_model_name not in trained_models:
        raise HTTPException(
            status_code=500,
            detail="Best model is not available for SHAP explanation.",
        )

    try:
        model = trained_models[best_model_name]
        top_features = get_global_feature_importance(model, pipeline, X_test_df)
        feature_importance = {
            item["feature"]: float(item.get("importance", 0.0))
            for item in top_features
            if isinstance(item, dict) and "feature" in item
        }
        return GlobalExplanationResponse(top_features=top_features, feature_importance=feature_importance)
    except Exception as exc:
        logger.error("Error generating SHAP explanation: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate SHAP explanation. Please check logs for details."
        )


@router.get("/history")
def get_training_history() -> Dict[str, List[float]]:
    """Return training history with epochs and accuracy."""
    return training_history


@router.get("/logs")
def get_prediction_logs() -> List[Dict[str, Any]]:
    """Return recent prediction logs."""
    return prediction_logs[-10:]  # Return last 10 logs