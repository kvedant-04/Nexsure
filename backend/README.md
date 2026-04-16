# Insurance Claim Approval Prediction Backend

FastAPI backend service for training ML models and predicting insurance claim approvals with SHAP explainability.

## Features

- **Model Training**: Train Logistic Regression and Random Forest models
- **Model Selection**: Automatically select best model based on F1-score
- **Prediction**: Generate predictions with probability scores
- **Explainability**: SHAP-based feature importance and explanations
- **Metrics**: Comprehensive model evaluation metrics

## API Endpoints

### POST /api/train
Train machine learning models on the provided dataset.

**Request Body:**
```json
{
  "dataset_filename": "insurance_claims.csv",
  "target_column": "approved",
  "test_size": 0.2,
  "random_state": 42
}
```

**Response:**
```json
{
  "models_trained": {
    "logistic_regression": "trained",
    "random_forest": "trained"
  },
  "best_model": "random_forest",
  "dataset_rows": 1000,
  "dataset_columns": 15
}
```

### POST /api/predict
Generate prediction for a single insurance claim.

**Request Body:**
```json
{
  "features": {
    "age": 35,
    "income": 50000,
    "claim_amount": 2500
  },
  "model_name": "logistic_regression"
}
```

**Response:**
```json
{
  "model_name": "logistic_regression",
  "prediction": 1,
  "probability": 0.78,
  "top_features": [
    {
      "feature": "income",
      "shap_value": 0.45,
      "impact": "increases"
    }
  ],
  "explanation": "This claim was approved mainly due to income and claim_amount."
}
```

### GET /api/metrics
Get performance metrics for all trained models.

**Response:**
```json
{
  "evaluations": {
    "logistic_regression": {
      "accuracy": 0.85,
      "precision": 0.82,
      "recall": 0.88,
      "f1_score": 0.85,
      "confusion_matrix": [[45, 8], [6, 41]],
      "roc_auc": 0.91
    }
  }
}
```

### GET /api/explain
Get global SHAP feature importance for the best model.

**Response:**
```json
{
  "top_features": [
    {
      "feature": "income",
      "importance": 0.35
    },
    {
      "feature": "claim_amount",
      "importance": 0.28
    }
  ]
}
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
uvicorn app.main:app --reload
```

## Data Requirements

- Dataset must be placed in `backend/data/` directory
- CSV format with headers
- Target column should be binary (0/1) for classification
- Features can be numeric or categorical

## Model Selection Criteria

Models are compared using:
1. **F1-score** (primary metric for imbalanced classification)
2. **Accuracy** (tie-breaker)

## Error Handling

The API provides comprehensive error handling with appropriate HTTP status codes:
- `400`: Bad request (missing models, invalid input)
- `404`: Dataset file not found
- `500`: Internal server errors

## Production Considerations

- Global state is used for simplicity in this scaffold
- For production, consider using a database for model storage
- Add authentication and rate limiting
- Implement model versioning and rollback capabilities