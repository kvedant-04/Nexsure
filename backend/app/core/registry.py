"""
registry.py -- Nexsure Model Registry, Lineage & Artifact Persistence

Responsibilities:
- Manage versioned immutable model storage: model/registry/<model_name>/<version>/model.joblib
- Manage model_registry.json (Champion, Challenger, Candidate metadata)
- Manage promotion_history.json (Audit log of all lifecycle events)
- Atomic persistence with automatic backup snapshots (model_registry.json.bak)
- Legacy best_model.pkl compatibility
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Path Helpers                                                                #
# --------------------------------------------------------------------------- #

def _model_root() -> Path:
    """Resolve the model/ directory relative to this file."""
    return Path(__file__).resolve().parents[3] / "model"


def get_model_root_folder() -> Path:
    """Public alias for model root."""
    return _model_root()


def get_model_registry_dir(folder: Optional[Path] = None) -> Path:
    """Return the base registry directory for immutable versioned artifacts."""
    return (folder or _model_root()) / "registry"


def get_immutable_artifact_path(
    model_name: str,
    version: str,
    folder: Optional[Path] = None,
) -> Path:
    """Path to an immutable, versioned model artifact: model/registry/<name>/<version>/model.joblib"""
    return get_model_registry_dir(folder) / model_name / version / "model.joblib"


def get_immutable_metadata_path(
    model_name: str,
    version: str,
    folder: Optional[Path] = None,
) -> Path:
    """Path to metadata alongside the immutable artifact."""
    return get_model_registry_dir(folder) / model_name / version / "metadata.json"


def get_model_registry_path(folder: Optional[Path] = None) -> Path:
    return (folder or _model_root()) / "model_registry.json"


def get_model_registry_backup_path(folder: Optional[Path] = None) -> Path:
    return (folder or _model_root()) / "model_registry.json.bak"


def get_promotion_history_path(folder: Optional[Path] = None) -> Path:
    return (folder or _model_root()) / "promotion_history.json"


def get_benchmark_results_path(folder: Optional[Path] = None) -> Path:
    return (folder or _model_root()) / "benchmark_results.json"


def get_metadata_path(folder: Optional[Path] = None) -> Path:
    return (folder or _model_root()) / "metadata.json"


def get_version_history_path(folder: Optional[Path] = None) -> Path:
    return (folder or _model_root()) / "version_history.json"


def get_champion_path(folder: Optional[Path] = None) -> Path:
    return (folder or _model_root()) / "best_model.pkl"


# --------------------------------------------------------------------------- #
#  Atomic JSON Persistence                                                     #
# --------------------------------------------------------------------------- #

def _atomic_write_json(file_path: Path, data: Any, create_backup: bool = True) -> None:
    """Write JSON data to a file atomically via a temporary file and replace."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = file_path.with_suffix(".tmp")
    
    if create_backup and file_path.exists():
        backup_path = file_path.with_suffix(".json.bak")
        try:
            shutil.copy2(file_path, backup_path)
        except Exception as exc:
            logger.warning("Failed to create backup for %s: %s", file_path.name, exc)

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    
    os.replace(temp_path, file_path)


# --------------------------------------------------------------------------- #
#  Immutable Artifact Storage                                                  #
# --------------------------------------------------------------------------- #

def save_immutable_artifact(
    model_obj: Any,
    model_name: str,
    version: str,
    metadata: Dict[str, Any],
    folder: Optional[Path] = None,
) -> Path:
    """Persist a model artifact into the immutable versioned hierarchy."""
    artifact_path = get_immutable_artifact_path(model_name, version, folder)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model_obj, artifact_path)
    
    meta_path = get_immutable_metadata_path(model_name, version, folder)
    _atomic_write_json(meta_path, metadata, create_backup=False)
    
    logger.info("Saved immutable artifact: %s (version %s)", model_name, version)
    return artifact_path


def load_immutable_artifact(
    model_name: str,
    version: str,
    folder: Optional[Path] = None,
) -> Any:
    """Load a model artifact from the immutable versioned hierarchy."""
    path = get_immutable_artifact_path(model_name, version, folder)
    if not path.exists():
        raise FileNotFoundError(f"Immutable model artifact not found: {path}")
    logger.info("Loading immutable artifact from %s", path)
    return joblib.load(path)


# --------------------------------------------------------------------------- #
#  Registry & Promotion History Persistence                                    #
# --------------------------------------------------------------------------- #

def save_model_registry(registry_data: Dict[str, Any], folder: Optional[Path] = None) -> None:
    path = get_model_registry_path(folder)
    _atomic_write_json(path, registry_data, create_backup=True)


def load_model_registry(folder: Optional[Path] = None) -> Dict[str, Any]:
    path = get_model_registry_path(folder)
    if not path.exists():
        backup = get_model_registry_backup_path(folder)
        if backup.exists():
            logger.warning("model_registry.json missing; loading from backup %s", backup)
            path = backup
        else:
            raise FileNotFoundError("model_registry.json not found")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("model_registry.json corrupt: %s; trying backup", exc)
        backup = get_model_registry_backup_path(folder)
        if backup.exists() and backup != path:
            with open(backup, "r", encoding="utf-8") as f:
                return json.load(f)
        raise exc


def save_promotion_history(history: List[Dict[str, Any]], folder: Optional[Path] = None) -> None:
    path = get_promotion_history_path(folder)
    _atomic_write_json(path, history, create_backup=False)


def load_promotion_history(folder: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = get_promotion_history_path(folder)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning("Error reading promotion_history.json: %s", exc)
        return []


def append_promotion_event(event: Dict[str, Any], folder: Optional[Path] = None) -> None:
    history = load_promotion_history(folder)
    history.append(event)
    save_promotion_history(history, folder)


# --------------------------------------------------------------------------- #
#  Artifact Validation                                                         #
# --------------------------------------------------------------------------- #

REQUIRED_CORE_ARTIFACTS = [
    "best_model.pkl",
    "preprocessing_pipeline.pkl",
    "metadata.json",
    "feature_columns.json",
]


def validate_core_artifacts(folder: Optional[Path] = None) -> bool:
    root = folder or _model_root()
    for name in REQUIRED_CORE_ARTIFACTS:
        if not (root / name).exists():
            return False
    return True


def validate_registry_artifact(folder: Optional[Path] = None) -> bool:
    try:
        reg = load_model_registry(folder)
        champion = reg.get("champion")
        if not champion or not champion.get("name") or not champion.get("version"):
            return False
        champ_path = get_immutable_artifact_path(champion["name"], champion["version"], folder)
        if not champ_path.exists():
            if not get_champion_path(folder).exists():
                return False
        return True
    except Exception:
        return False


def validate_benchmark_artifact(folder: Optional[Path] = None) -> bool:
    root = folder or _model_root()
    path = root / "benchmark_results.json"
    if not path.exists():
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return bool(data.get("champion") and data.get("ranking"))
    except Exception:
        return False


def validate_metadata(folder: Optional[Path] = None) -> bool:
    path = get_metadata_path(folder)
    if not path.exists():
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return bool(data.get("best_model_name") and data.get("accuracy") is not None)
    except Exception:
        return False


def all_artifacts_valid(folder: Optional[Path] = None) -> bool:
    return (
        validate_core_artifacts(folder)
        and validate_benchmark_artifact(folder)
        and validate_metadata(folder)
        and validate_registry_artifact(folder)
    )


# --------------------------------------------------------------------------- #
#  Loaders                                                                     #
# --------------------------------------------------------------------------- #

def load_champion_model(folder: Optional[Path] = None) -> Any:
    try:
        reg = load_model_registry(folder)
        champ = reg.get("champion", {})
        if champ.get("name") and champ.get("version"):
            try:
                return load_immutable_artifact(champ["name"], champ["version"], folder)
            except Exception:
                pass
    except Exception:
        pass
    
    path = get_champion_path(folder)
    return joblib.load(path)


def save_benchmark_report(report: Dict[str, Any], folder: Optional[Path] = None) -> None:
    folder = folder or _model_root()
    folder.mkdir(parents=True, exist_ok=True)
    path = get_benchmark_results_path(folder)
    _atomic_write_json(path, report, create_backup=False)


def load_benchmark_report(folder: Optional[Path] = None) -> Dict[str, Any]:
    path = get_benchmark_results_path(folder)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_metadata_v2(metadata: Dict[str, Any], folder: Optional[Path] = None) -> None:
    path = get_metadata_path(folder)
    _atomic_write_json(path, metadata, create_backup=False)


def load_metadata(folder: Optional[Path] = None) -> Dict[str, Any]:
    path = get_metadata_path(folder)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_version_history_v2(entry: Dict[str, Any], folder: Optional[Path] = None) -> None:
    path = get_version_history_path(folder)
    history = []
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(entry)
    _atomic_write_json(path, history, create_backup=False)


def load_version_history(folder: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = get_version_history_path(folder)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []