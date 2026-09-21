"""
paths.py -- Centralized Cross-Platform Path Management for Nexsure AI Engine

Ensures consistent filesystem resolution across:
- Windows localhost development
- Linux Render containers
- Git clone environments
- Custom environment variable overrides (MODEL_DIR, DATA_DIR, BASE_DIR)
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Centralized Base Directory Resolution                                       #
# --------------------------------------------------------------------------- #

# Current file is backend/app/core/paths.py
# .parents[0] = backend/app/core
# .parents[1] = backend/app
# .parents[2] = backend
# .parents[3] = repository root
_CURRENT_FILE = Path(__file__).resolve()
BACKEND_DIR = _CURRENT_FILE.parents[2]
REPO_ROOT = _CURRENT_FILE.parents[3]

# Optional override from environment
_CUSTOM_BASE_DIR = os.getenv("BASE_DIR") or os.getenv("NEXSURE_BASE_DIR")
BASE_DIR = Path(_CUSTOM_BASE_DIR).resolve() if _CUSTOM_BASE_DIR else REPO_ROOT


def get_base_dir() -> Path:
    """Return the resolved base repository root directory."""
    return BASE_DIR


def get_backend_dir() -> Path:
    """Return the resolved backend directory."""
    return BACKEND_DIR


# --------------------------------------------------------------------------- #
#  Model & Data Directory Resolution                                           #
# --------------------------------------------------------------------------- #

def get_model_root_folder() -> Path:
    """
    Resolve the model root folder.
    Checks:
      1. Environment variables (MODEL_DIR or NEXSURE_MODEL_DIR)
      2. <REPO_ROOT>/model
      3. <BACKEND_DIR>/model
      4. <CWD>/model
      5. Fallback to <REPO_ROOT>/model
    """
    env_model_dir = os.getenv("MODEL_DIR") or os.getenv("NEXSURE_MODEL_DIR")
    if env_model_dir:
        p = Path(env_model_dir).resolve()
        if p.exists():
            return p

    candidates = [
        BASE_DIR / "model",
        REPO_ROOT / "model",
        BACKEND_DIR / "model",
        Path.cwd() / "model",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Default to BASE_DIR / "model"
    return BASE_DIR / "model"


def get_data_folder() -> Path:
    """
    Resolve the backend data directory.
    Checks:
      1. Environment variables (DATA_DIR or NEXSURE_DATA_DIR)
      2. <BACKEND_DIR>/data
      3. <BASE_DIR>/backend/data
      4. <BASE_DIR>/data
      5. <CWD>/backend/data or <CWD>/data
    """
    env_data_dir = os.getenv("DATA_DIR") or os.getenv("NEXSURE_DATA_DIR")
    if env_data_dir:
        p = Path(env_data_dir).resolve()
        if p.exists():
            return p

    candidates = [
        BACKEND_DIR / "data",
        BASE_DIR / "backend" / "data",
        BASE_DIR / "data",
        Path.cwd() / "backend" / "data",
        Path.cwd() / "data",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return BACKEND_DIR / "data"


def get_dataset_path(filename: str = "insurance3r2.csv") -> Path:
    """
    Resolve a specific dataset file.
    Searches across candidate data directories.
    """
    if not filename:
        raise ValueError("Dataset filename cannot be empty")

    # If an absolute path is passed
    direct_path = Path(filename)
    if direct_path.is_absolute() and direct_path.exists():
        return direct_path

    # Check data folder first
    data_dir = get_data_folder()
    primary = data_dir / filename
    if primary.exists():
        return primary

    # Search other candidate locations
    search_dirs = [
        BACKEND_DIR / "data",
        BASE_DIR / "backend" / "data",
        BASE_DIR / "data",
        Path.cwd() / "backend" / "data",
        Path.cwd() / "data",
    ]
    for directory in search_dirs:
        candidate = directory / filename
        if candidate.exists():
            return candidate

    # Return primary even if not yet found (will fail gracefully with informative error)
    return primary


# --------------------------------------------------------------------------- #
#  Model Artifact Paths Helpers                                                #
# --------------------------------------------------------------------------- #

def get_artifacts_folder(folder: Optional[Path] = None) -> Path:
    """Return the artifacts subfolder for model variants: model/artifacts"""
    return (folder or get_model_root_folder()) / "artifacts"


def get_model_file_path(
    name: str = "best_model",
    folder: Optional[Path] = None,
    extension: str = "pkl",
) -> Path:
    return (folder or get_model_root_folder()) / f"{name}.{extension}"


def get_champion_path(folder: Optional[Path] = None) -> Path:
    """Path to active champion model: model/best_model.pkl"""
    return (folder or get_model_root_folder()) / "best_model.pkl"


def get_pipeline_path(
    folder: Optional[Path] = None,
    filename: str = "preprocessing_pipeline.pkl",
) -> Path:
    """Path to preprocessing pipeline: model/preprocessing_pipeline.pkl"""
    return (folder or get_model_root_folder()) / filename


def get_metadata_path(
    folder: Optional[Path] = None,
    filename: str = "metadata.json",
) -> Path:
    """Path to metadata: model/metadata.json"""
    return (folder or get_model_root_folder()) / filename


def get_benchmark_results_path(
    folder: Optional[Path] = None,
    filename: str = "benchmark_results.json",
) -> Path:
    """Path to benchmark results: model/benchmark_results.json"""
    return (folder or get_model_root_folder()) / filename


def get_model_registry_path(
    folder: Optional[Path] = None,
    filename: str = "model_registry.json",
) -> Path:
    """Path to model registry: model/model_registry.json"""
    return (folder or get_model_root_folder()) / filename


def get_model_registry_backup_path(
    folder: Optional[Path] = None,
    filename: str = "model_registry.json.bak",
) -> Path:
    """Path to model registry backup: model/model_registry.json.bak"""
    return (folder or get_model_root_folder()) / filename


def get_promotion_history_path(
    folder: Optional[Path] = None,
    filename: str = "promotion_history.json",
) -> Path:
    """Path to promotion history: model/promotion_history.json"""
    return (folder or get_model_root_folder()) / filename


def get_feature_columns_path(
    folder: Optional[Path] = None,
    filename: str = "feature_columns.json",
) -> Path:
    """Path to feature columns: model/feature_columns.json"""
    return (folder or get_model_root_folder()) / filename


def get_raw_feature_schema_path(
    folder: Optional[Path] = None,
    filename: str = "raw_feature_schema.json",
) -> Path:
    """Path to raw feature schema: model/raw_feature_schema.json"""
    return (folder or get_model_root_folder()) / filename


def get_version_history_path(
    folder: Optional[Path] = None,
    filename: str = "version_history.json",
) -> Path:
    """Path to version history: model/version_history.json"""
    return (folder or get_model_root_folder()) / filename


def get_model_registry_dir(folder: Optional[Path] = None) -> Path:
    """Path to versioned registry directory: model/registry"""
    return (folder or get_model_root_folder()) / "registry"


def get_immutable_artifact_path(
    model_name: str,
    version: str,
    folder: Optional[Path] = None,
) -> Path:
    """Path to an immutable, versioned model artifact: model/registry/<model_name>/<version>/model.joblib"""
    return get_model_registry_dir(folder) / model_name / version / "model.joblib"


def get_immutable_metadata_path(
    model_name: str,
    version: str,
    folder: Optional[Path] = None,
) -> Path:
    """Path to immutable metadata: model/registry/<model_name>/<version>/metadata.json"""
    return get_model_registry_dir(folder) / model_name / version / "metadata.json"


REQUIRED_PRODUCTION_ARTIFACTS = [
    "best_model.pkl",
    "preprocessing_pipeline.pkl",
    "metadata.json",
    "benchmark_results.json",
    "model_registry.json",
    "promotion_history.json",
    "feature_columns.json",
]


def validate_production_artifacts(folder: Optional[Path] = None) -> tuple[bool, list[str]]:
    """
    Validate that all 7 required production artifacts exist on disk.
    Returns (is_valid, list_of_missing_artifacts).
    """
    root = folder or get_model_root_folder()
    missing = [artifact for artifact in REQUIRED_PRODUCTION_ARTIFACTS if not (root / artifact).exists()]
    return len(missing) == 0, missing

