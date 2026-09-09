"""
performance.py -- Precision Timing Instrumentation & Non-Blocking Summary Persistence

Implements:
- PrecisionTimer measuring nanoseconds across all 8 inference lifecycle stages:
  1. validation
  2. preprocessing
  3. feature_ordering
  4. inference
  5. shap
  6. explanation_formatting
  7. serialization
  8. response_build
- Non-blocking metadata summary updates (average_latency, P50, P95, prediction_count, etc.)
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from app.core.registry import _model_root, get_metadata_path, load_metadata, save_metadata_v2

logger = logging.getLogger(__name__)


class PrecisionTimer:
    """Nanosecond-precision inference stage timer."""

    def __init__(self):
        self.start_ns = time.perf_counter_ns()
        self.stages_ns: Dict[str, int] = {}
        self._current_stage: Optional[str] = None
        self._stage_start_ns: Optional[int] = None

    def start_stage(self, stage_name: str) -> None:
        """Begin measuring a specific lifecycle stage."""
        now = time.perf_counter_ns()
        if self._current_stage is not None and self._stage_start_ns is not None:
            elapsed = now - self._stage_start_ns
            self.stages_ns[self._current_stage] = elapsed
        self._current_stage = stage_name
        self._stage_start_ns = now

    def end_stage(self) -> None:
        """Stop the currently active stage."""
        now = time.perf_counter_ns()
        if self._current_stage is not None and self._stage_start_ns is not None:
            elapsed = now - self._stage_start_ns
            self.stages_ns[self._current_stage] = elapsed
            self._current_stage = None
            self._stage_start_ns = None

    def finish(self) -> Tuple[Dict[str, int], int]:
        """Finish all timing and return (stages_dict, total_nanoseconds)."""
        self.end_stage()
        total_ns = time.perf_counter_ns() - self.start_ns
        return self.stages_ns, total_ns


_PERSIST_LOCK = threading.Lock()


def persist_summary_metadata_async(
    avg_latency_ms: float,
    p50_latency_ms: float,
    p95_latency_ms: float,
    prediction_count: int,
    folder: Optional[Path] = None,
) -> None:
    """Non-blocking background update of summary metadata in metadata.json."""
    def _worker():
        with _PERSIST_LOCK:
            try:
                root = folder or _model_root()
                meta = load_metadata(root)
                meta["average_latency_ms"] = avg_latency_ms
                meta["p50_latency_ms"] = p50_latency_ms
                meta["p95_latency_ms"] = p95_latency_ms
                meta["prediction_count"] = prediction_count
                meta["last_prediction_timestamp"] = datetime.now(timezone.utc).isoformat()
                meta["cache_enabled"] = True
                save_metadata_v2(meta, root)
            except Exception as exc:
                logger.warning("Background metadata persistence error: %s", exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()