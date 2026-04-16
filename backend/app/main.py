import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.predict import router as predict_router, _load_saved_artifacts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI-Based Insurance Claim Approval Prediction API",
    description=(
        "Backend API for training models, evaluating performance, and predicting "
        "insurance claim approvals using dataset data loaded from backend/data/."
    ),
    version="0.1.0",
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api")


@app.on_event("startup")
def startup_event() -> None:
    try:
        _load_saved_artifacts()
        logger.info("Loaded saved model artifacts on startup.")
    except Exception as exc:
        logger.warning("No saved model artifacts loaded on startup: %s", exc)
