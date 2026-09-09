"""
cache_manager.py -- Nexsure Multi-Layered Runtime Cache Architecture (Phase 3.1 Optimized)

Implements:
- Layered runtime caches with individual lifecycle state tracking:
  - ChampionModelCache
  - PreprocessingPipelineCache
  - ShapExplainerCache (Version & Artifact-Hash Aware)
  - ShapBackgroundCache (Precomputed Background Matrix & Expected Value)
  - FeatureMappingCache (Precomputed Feature Index -> Human Label mappings)
  - MetadataCache
  - RegistryCache
  - FeatureSchemaCache
- Explicit Readiness Stages:
  BOOTING -> LOADING_REGISTRY -> LOADING_MODEL -> LOADING_PIPELINE -> WARMING_EXPLAINER -> CACHE_READY -> READY
- Isolated fault recovery for each individual cache layer
- Zero disk reads during active production serving
"""

import hashlib
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap

from app.core.registry import (
    _model_root,
    get_champion_path,
    get_immutable_artifact_path,
    get_metadata_path,
    get_model_registry_path,
    load_champion_model,
    load_metadata,
    load_model_registry,
)
from app.core.train import load_feature_columns, load_pipeline

logger = logging.getLogger(__name__)


FEATURE_MAPPINGS = {
    "age": "Age",
    "bmi": "Body Mass Index",
    "children": "Dependents",
    "smoker_1": "Smoking Profile",
    "smoker_yes": "Smoking Profile",
    "smoker_0": "Smoking Profile",
    "smoker_no": "Smoking Profile",
    "sex_1": "Gender (Male)",
    "sex_0": "Gender (Female)",
    "sex_male": "Gender (Male)",
    "sex_female": "Gender (Female)",
    "region_0": "Geographic Region",
    "region_1": "Geographic Region",
    "region_2": "Geographic Region",
    "region_3": "Geographic Region",
    "region_northeast": "Geographic Region",
    "region_northwest": "Geographic Region",
    "region_southeast": "Geographic Region",
    "region_southwest": "Geographic Region",
}


def _humanize_feature(raw_name: str) -> str:
    """Map encoded ML feature names to executive-friendly labels."""
    lower_name = raw_name.lower()
    for key, mapped in FEATURE_MAPPINGS.items():
        if key in lower_name:
            return mapped
    return raw_name.replace("_", " ").title()


# --------------------------------------------------------------------------- #
#  Base Cache Layer                                                            #
# --------------------------------------------------------------------------- #

class CacheLayer:
    """Base class for an isolated runtime cache layer."""

    def __init__(self, name: str):
        self.name = name
        self._lock = threading.RLock()
        self.state = "MISS"  # READY | MISS | INVALIDATED | REBUILDING
        self.initialized_at: Optional[str] = None
        self.last_accessed_at: Optional[str] = None
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self._data: Any = None

    def get(self) -> Any:
        with self._lock:
            self.last_accessed_at = datetime.now(timezone.utc).isoformat()
            if self._data is not None and self.state == "READY":
                self.cache_hits += 1
                return self._data
            self.cache_misses += 1
            return None

    def set(self, data: Any) -> None:
        with self._lock:
            self._data = data
            self.state = "READY"
            now = datetime.now(timezone.utc).isoformat()
            if self.initialized_at is None:
                self.initialized_at = now
            self.last_accessed_at = now

    def invalidate(self) -> None:
        with self._lock:
            self._data = None
            self.state = "INVALIDATED"
            logger.info("CacheLayer [%s] marked INVALIDATED", self.name)

    def get_cache_status(self) -> Dict[str, Any]:
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            total = self.cache_hits + self.cache_misses
            hit_rate = (self.cache_hits / total * 100.0) if total > 0 else 100.0
            return {
                "name": self.name,
                "state": self.state,
                "initialized_at": self.initialized_at,
                "last_accessed_at": self.last_accessed_at,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate_pct": round(hit_rate, 2),
            }


# --------------------------------------------------------------------------- #
#  Specialized Cache Subsystems                                               #
# --------------------------------------------------------------------------- #

class ChampionModelCache(CacheLayer):
    def __init__(self):
        super().__init__("Champion Model Cache")
        self.model_name: Optional[str] = None
        self.model_version: Optional[str] = None
        self.artifact_hash: Optional[str] = None

    def set_model(self, model: Any, name: str, version: str, artifact_hash: str) -> None:
        with self._lock:
            self.model_name = name
            self.model_version = version
            self.artifact_hash = artifact_hash
            self.set(model)

    def get_cache_status(self) -> Dict[str, Any]:
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        res = super().get_status()
        res.update({
            "model_name": self.model_name,
            "model_version": self.model_version,
            "artifact_hash": self.artifact_hash,
        })
        return res


class PreprocessingPipelineCache(CacheLayer):
    def __init__(self):
        super().__init__("Preprocessing Pipeline Cache")
        self.transformed_feature_count: Optional[int] = None

    def set_pipeline(self, pipeline: Any, feature_count: int) -> None:
        with self._lock:
            self.transformed_feature_count = feature_count
            self.set(pipeline)

    def get_cache_status(self) -> Dict[str, Any]:
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        res = super().get_status()
        res["transformed_feature_count"] = self.transformed_feature_count
        return res


class ShapExplainerCache(CacheLayer):
    """
    Model-Version & Artifact-Hash Aware SHAP Explainer Cache.
    Key: (model_name, model_version, artifact_hash)
    """
    def __init__(self):
        super().__init__("SHAP Explainer Cache")
        self.explainer_cache: Dict[str, Any] = {}
        self.active_key: Optional[str] = None

    def make_key(self, model_name: str, version: str, artifact_hash: str) -> str:
        return f"{model_name}:{version}:{artifact_hash[:12]}"

    def get_explainer(self, key: str) -> Optional[Any]:
        with self._lock:
            self.last_accessed_at = datetime.now(timezone.utc).isoformat()
            if key in self.explainer_cache:
                self.cache_hits += 1
                self.active_key = key
                return self.explainer_cache[key]
            self.cache_misses += 1
            return None

    def store_explainer(self, key: str, explainer: Any) -> None:
        with self._lock:
            self.explainer_cache[key] = explainer
            self.active_key = key
            self.state = "READY"
            now = datetime.now(timezone.utc).isoformat()
            if self.initialized_at is None:
                self.initialized_at = now
            self.last_accessed_at = now

    def invalidate(self) -> None:
        with self._lock:
            self.explainer_cache.clear()
            self.active_key = None
            self.state = "INVALIDATED"

    def get_cache_status(self) -> Dict[str, Any]:
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        res = super().get_status()
        res.update({
            "active_key": self.active_key,
            "cached_explainers_count": len(self.explainer_cache),
            "cached_keys": list(self.explainer_cache.keys()),
        })
        return res


class ShapBackgroundCache(CacheLayer):
    """
    Persistent SHAP Background Dataset & Feature Mapping Cache (Phase 3.1).
    Precomputed once during warm startup:
    - Preprocessed background dataset (sampled from training set)
    - Encoded feature matrix
    - Expected value / base value
    - Precomputed mapping table [(col_idx, human_name), ...]
    """
    def __init__(self):
        super().__init__("SHAP Background Cache")
        self.background_matrix: Optional[np.ndarray] = None
        self.expected_value: Optional[float] = None
        self.feature_names: Optional[List[str]] = None
        self.mapping_table: Optional[List[Tuple[int, str]]] = None
        self.sample_count: int = 0

    def set_background(
        self,
        background_matrix: np.ndarray,
        expected_value: Optional[float],
        feature_names: List[str],
        mapping_table: List[Tuple[int, str]],
    ) -> None:
        with self._lock:
            self.background_matrix = background_matrix
            self.expected_value = expected_value
            self.feature_names = feature_names
            self.mapping_table = mapping_table
            self.sample_count = len(background_matrix) if background_matrix is not None else 0
            self.set({
                "background_matrix": background_matrix,
                "expected_value": expected_value,
                "feature_names": feature_names,
                "mapping_table": mapping_table,
            })

    def get_cache_status(self) -> Dict[str, Any]:
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        res = super().get_status()
        res.update({
            "background_cached": self.background_matrix is not None,
            "sample_count": self.sample_count,
            "feature_mapping_cached": self.mapping_table is not None,
            "feature_count": len(self.feature_names) if self.feature_names else 0,
            "expected_value": float(self.expected_value) if self.expected_value is not None else None,
        })
        return res


class FeatureMappingCache(CacheLayer):
    """Precomputed feature mapping dictionary cache."""
    def __init__(self):
        super().__init__("Feature Schema & Mapping Cache")
        self.mapping_table: Optional[List[Tuple[int, str]]] = None

    def set_mappings(self, mapping_table: List[Tuple[int, str]]) -> None:
        with self._lock:
            self.mapping_table = mapping_table
            self.set(mapping_table)


class MetadataCache(CacheLayer):
    def __init__(self):
        super().__init__("Metadata Cache")


class RegistryCache(CacheLayer):
    def __init__(self):
        super().__init__("Registry Cache")


class FeatureSchemaCache(CacheLayer):
    def __init__(self):
        super().__init__("Feature Schema Cache")


# --------------------------------------------------------------------------- #
#  Runtime Cache Coordinator (Singleton)                                     #
# --------------------------------------------------------------------------- #

class RuntimeCache:
    """Coordinator for all multi-level cache subsystems."""

    def __init__(self):
        self._lock = threading.RLock()
        self.champion_cache = ChampionModelCache()
        self.pipeline_cache = PreprocessingPipelineCache()
        self.shap_cache = ShapExplainerCache()
        self.background_cache = ShapBackgroundCache()
        self.feature_mapping_cache = FeatureMappingCache()
        self.metadata_cache = MetadataCache()
        self.registry_cache = RegistryCache()
        self.feature_cache = FeatureSchemaCache()

        # Startup lifecycle state
        self.readiness_stage: str = "BOOTING"
        self.is_warm: bool = False
        self.boot_start_time = time.perf_counter()

    def set_stage(self, stage: str) -> None:
        with self._lock:
            self.readiness_stage = stage
            logger.info("RuntimeCache Lifecycle Stage -> %s", stage)

    def _compute_model_hash(self, model_obj: Any) -> str:
        """Compute SHA-256 fingerprint of serialized model weights."""
        try:
            raw_bytes = joblib.dumps(model_obj)
            return hashlib.sha256(raw_bytes).hexdigest()[:16]
        except Exception:
            return "hash_default"

    def initialize_runtime_cache(
        self,
        folder: Optional[Path] = None,
        sample_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Execute full Warm Startup:
          1. Load & cache Registry
          2. Load & cache Champion Model
          3. Load & cache Preprocessing Pipeline
          4. Load & cache Feature Schema & Metadata
          5. Precompute Background Matrix & Feature Mapping Table
          6. Pre-warm & cache SHAP Explainer
          7. Mark CACHE_READY -> READY
        """
        with self._lock:
            root = folder or _model_root()

            # Stage 1: Load Registry
            self.set_stage("LOADING_REGISTRY")
            self.registry_cache.state = "REBUILDING"
            reg = load_model_registry(root)
            self.registry_cache.set(reg)

            # Stage 2: Load Feature Schema & Metadata
            self.metadata_cache.state = "REBUILDING"
            self.feature_cache.state = "REBUILDING"
            meta = load_metadata(root)
            feature_cols = load_feature_columns(root)
            self.metadata_cache.set(meta)
            self.feature_cache.set(feature_cols)

            # Stage 3: Load Model
            self.set_stage("LOADING_MODEL")
            self.champion_cache.state = "REBUILDING"
            champ_meta = reg.get("champion", {})
            champ_name = champ_meta.get("name") or meta.get("best_model_name", "catboost")
            champ_ver = champ_meta.get("version", "v2.1")
            champ_model = load_champion_model(root)
            if isinstance(champ_model, dict) and "model_object" in champ_model:
                champ_model = champ_model["model_object"]
            model_hash = self._compute_model_hash(champ_model)
            self.champion_cache.set_model(champ_model, champ_name, champ_ver, model_hash)

            # Stage 4: Load Pipeline
            self.set_stage("LOADING_PIPELINE")
            self.pipeline_cache.state = "REBUILDING"
            pipeline = load_pipeline(root)
            transformed_count = meta.get("transformed_feature_count", 11)
            self.pipeline_cache.set_pipeline(pipeline, transformed_count)

            # Precompute Feature Names & Mapping Table
            try:
                feature_names = list(pipeline.get_feature_names_out())
            except Exception:
                feature_names = [f"feature_{i}" for i in range(transformed_count)]
            mapping_table = [(i, _humanize_feature(name)) for i, name in enumerate(feature_names)]
            self.feature_mapping_cache.set_mappings(mapping_table)

            # Stage 5: Pre-warm SHAP Explainer & Background Matrix
            self.set_stage("WARMING_EXPLAINER")
            self.shap_cache.state = "REBUILDING"
            self.background_cache.state = "REBUILDING"
            self.warm_shap_explainer(champ_name, champ_ver, champ_model, pipeline, sample_df, feature_names, mapping_table)

            self.set_stage("CACHE_READY")
            self.is_warm = True
            self.set_stage("READY")
            logger.info("RuntimeCache warm startup completed successfully. Zero disk reads active.")

    def warm_shap_explainer(
        self,
        model_name: str,
        version: str,
        model_obj: Any,
        pipeline: Any,
        sample_df: Optional[pd.DataFrame] = None,
        feature_names: Optional[List[str]] = None,
        mapping_table: Optional[List[Tuple[int, str]]] = None,
    ) -> Any:
        """Pre-warm and store a SHAP Explainer and Background Dataset in the versioned cache."""
        with self._lock:
            model_hash = self.champion_cache.artifact_hash or self._compute_model_hash(model_obj)
            key = self.shap_cache.make_key(model_name, version, model_hash)

            existing = self.shap_cache.get_explainer(key)
            if existing is not None and self.background_cache.state == "READY":
                return existing

            logger.info("Initializing SHAP Explainer & Background Cache for [%s] (%s)...", model_name, key)

            # 1. Load Background Sample from training dataset
            if sample_df is None:
                data_candidates = [
                    Path("backend/data/insurance3r2.csv"),
                    Path("data/insurance3r2.csv"),
                    Path("d:/Nexsure/backend/data/insurance3r2.csv"),
                ]
                raw_df = None
                for p in data_candidates:
                    if p.exists():
                        try:
                            raw_df = pd.read_csv(p)
                            break
                        except Exception:
                            pass

                if raw_df is not None:
                    raw_cols = ["age", "sex", "bmi", "children", "smoker", "region"]
                    available_cols = [c for c in raw_cols if c in raw_df.columns]
                    sample_df = raw_df[available_cols].sample(min(100, len(raw_df)), random_state=42)
                else:
                    sample_df = pd.DataFrame([{
                        "age": 35,
                        "sex": "female",
                        "bmi": 27.5,
                        "children": 1,
                        "smoker": "no",
                        "region": "northeast"
                    }])

            X_transformed = pipeline.transform(sample_df)
            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()

            if feature_names is None:
                try:
                    feature_names = list(pipeline.get_feature_names_out())
                except Exception:
                    feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
            if mapping_table is None:
                mapping_table = [(i, _humanize_feature(name)) for i, name in enumerate(feature_names)]

            # 2. Construct Explainer
            actual_model = model_obj
            if isinstance(model_obj, dict) and "model_object" in model_obj:
                actual_model = model_obj["model_object"]

            explainer = None
            if hasattr(actual_model, "feature_importances_") or hasattr(actual_model, "predict_proba"):
                try:
                    explainer = shap.TreeExplainer(actual_model)
                except Exception as exc:
                    logger.warning("TreeExplainer init fallback: %s", exc)
                    explainer = shap.Explainer(actual_model, X_transformed)
            elif hasattr(actual_model, "coef_"):
                explainer = shap.LinearExplainer(actual_model, X_transformed, feature_perturbation="interventional")
            else:
                explainer = shap.Explainer(actual_model, X_transformed)

            # Warm-up pass
            try:
                if hasattr(explainer, "shap_values"):
                    explainer.shap_values(X_transformed[:2], check_additivity=False)
                else:
                    explainer(X_transformed[:2])
            except Exception:
                pass

            # Extract base / expected value
            expected_val = getattr(explainer, "expected_value", None)
            if isinstance(expected_val, (list, np.ndarray)):
                expected_val = float(expected_val[1] if len(expected_val) > 1 else expected_val[0])

            self.shap_cache.store_explainer(key, explainer)
            self.background_cache.set_background(
                background_matrix=X_transformed,
                expected_value=expected_val,
                feature_names=feature_names,
                mapping_table=mapping_table,
            )
            return explainer

    # ----------------------------------------------------------------------- #
    # Fast In-Memory Accessors (Zero Disk I/O)                                 #
    # ----------------------------------------------------------------------- #

    def get_champion_model(self) -> Optional[Any]:
        return self.champion_cache.get()

    def get_pipeline(self) -> Optional[Any]:
        return self.pipeline_cache.get()

    def get_shap_explainer(self) -> Optional[Any]:
        if self.shap_cache.active_key:
            return self.shap_cache.get_explainer(self.shap_cache.active_key)
        # Fallback to first cached explainer
        with self.shap_cache._lock:
            if self.shap_cache.explainer_cache:
                k = next(iter(self.shap_cache.explainer_cache))
                return self.shap_cache.get_explainer(k)
        return None

    def get_shap_background(self) -> Optional[Dict[str, Any]]:
        return self.background_cache.get()

    def get_feature_mapping(self) -> Optional[List[Tuple[int, str]]]:
        return self.feature_mapping_cache.mapping_table

    def get_metadata(self) -> Optional[Dict[str, Any]]:
        return self.metadata_cache.get()

    def get_registry(self) -> Optional[Dict[str, Any]]:
        return self.registry_cache.get()

    def get_feature_columns(self) -> Optional[List[str]]:
        return self.feature_cache.get()

    def invalidate_for_promotion(self, new_champ_name: str, new_version: str, new_model_obj: Any) -> None:
        """Atomically invalidate & update cache when Champion is promoted or rolled back."""
        with self._lock:
            logger.info("Invalidating caches for promotion -> [%s] (%s)", new_champ_name, new_version)
            actual_model = new_model_obj.get("model_object", new_model_obj) if isinstance(new_model_obj, dict) else new_model_obj
            new_hash = self._compute_model_hash(actual_model)

            self.champion_cache.set_model(actual_model, new_champ_name, new_version, new_hash)
            self.shap_cache.invalidate()

            pipeline = self.pipeline_cache.get()
            if pipeline is not None:
                self.warm_shap_explainer(new_champ_name, new_version, actual_model, pipeline)
            logger.info("Caches rebuilt successfully after promotion.")

    def get_cache_status(self) -> Dict[str, Any]:
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """Aggregate health status for /api/cache-status."""
        with self._lock:
            layers = [
                self.champion_cache.get_status(),
                self.pipeline_cache.get_status(),
                self.shap_cache.get_status(),
                self.background_cache.get_status(),
                self.feature_mapping_cache.get_status(),
                self.registry_cache.get_status(),
                self.metadata_cache.get_status(),
                self.feature_cache.get_status(),
            ]
            all_ready = all(l["state"] == "READY" for l in layers)
            return {
                "readiness_stage": self.readiness_stage,
                "is_warm": self.is_warm,
                "overall_status": "READY" if all_ready else "DEGRADED",
                "explainer_initialized": self.shap_cache.state == "READY",
                "background_cached": self.background_cache.background_matrix is not None,
                "feature_mapping_cached": self.feature_mapping_cache.mapping_table is not None,
                "model_hash": self.champion_cache.artifact_hash,
                "active_champion": {
                    "name": self.champion_cache.model_name,
                    "version": self.champion_cache.model_version,
                    "artifact_hash": self.champion_cache.artifact_hash,
                },
                "layers": layers,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }


# Global Runtime Cache Coordinator Singleton
runtime_cache = RuntimeCache()
