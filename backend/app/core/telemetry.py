"""
telemetry.py -- Nexsure Enterprise Runtime Telemetry & Performance Engine (Phase 3.1 Optimized)

Provides:
- High-precision nanosecond stage timing
- Bounded rolling percentile calculations (P50, P90, P95, P99, Mean, Std, Min, Max)
- Dedicated SHAP explainability telemetry (SHAP P50, P95, Mean, Cache hit rates, Reduction %)
- Real-time throughput metrics (RPS, predictions/min, uptime, success/error rates)
- Rolling sparkline history (last 30 measurements)
"""

import collections
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Bounded ring buffers to prevent memory leaks
MAX_HISTORY_LEN = 500
TIMELINE_LEN = 30

# Baseline reference benchmarks before Phase 3.1 optimization (recorded in Phase 3)
BASELINE_SHAP_LATENCY_MS = 9.87
BASELINE_TOTAL_LATENCY_MS = 14.01


class TelemetryEngine:
    """Thread-safe singleton for inference latency, SHAP profiling, and throughput tracking."""

    def __init__(self):
        self._lock = threading.RLock()
        self._boot_time = time.perf_counter()
        self._boot_timestamp = datetime.now(timezone.utc).isoformat()

        # Prediction counters
        self.total_predictions = 0
        self.successful_predictions = 0
        self.failed_predictions = 0

        # Cache counters
        self.cache_hits = 0
        self.cache_misses = 0
        self.shap_cache_hits = 0
        self.shap_cache_misses = 0

        # Latency buffers (in milliseconds)
        self._total_latencies = collections.deque(maxlen=MAX_HISTORY_LEN)
        self._model_latencies = collections.deque(maxlen=MAX_HISTORY_LEN)
        self._shap_latencies = collections.deque(maxlen=MAX_HISTORY_LEN)
        self._preprocess_latencies = collections.deque(maxlen=MAX_HISTORY_LEN)
        self._validation_latencies = collections.deque(maxlen=MAX_HISTORY_LEN)
        self._ordering_latencies = collections.deque(maxlen=MAX_HISTORY_LEN)
        self._formatting_latencies = collections.deque(maxlen=MAX_HISTORY_LEN)
        self._serialization_latencies = collections.deque(maxlen=MAX_HISTORY_LEN)

        # Recent timestamps for rolling RPS / RPM computation
        self._request_timestamps = collections.deque(maxlen=MAX_HISTORY_LEN)

        # Sparkline timeline events (last 30)
        self._timeline = collections.deque(maxlen=TIMELINE_LEN)

    def record_prediction(
        self,
        stages_ns: Dict[str, int],
        total_ns: int,
        success: bool = True,
        model_name: str = "best_model",
        cache_hit: bool = True,
        shap_cache_hit: bool = True,
    ) -> Dict[str, Any]:
        """Record an inference event with nanosecond-precision stage timings."""
        now = time.perf_counter()
        now_iso = datetime.now(timezone.utc).isoformat()

        total_ms = round(total_ns / 1_000_000.0, 3)
        val_ms = round(stages_ns.get("validation", 0) / 1_000_000.0, 3)
        pre_ms = round(stages_ns.get("preprocessing", 0) / 1_000_000.0, 3)
        ord_ms = round(stages_ns.get("feature_ordering", 0) / 1_000_000.0, 3)
        inf_ms = round(stages_ns.get("inference", 0) / 1_000_000.0, 3)
        shp_ms = round(stages_ns.get("shap", 0) / 1_000_000.0, 3)
        fmt_ms = round(stages_ns.get("explanation_formatting", 0) / 1_000_000.0, 3)
        ser_ms = round(stages_ns.get("serialization", 0) / 1_000_000.0, 3)

        with self._lock:
            self.total_predictions += 1
            if success:
                self.successful_predictions += 1
            else:
                self.failed_predictions += 1

            if cache_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1

            if shap_cache_hit:
                self.shap_cache_hits += 1
            else:
                self.shap_cache_misses += 1

            # Append to bounded rolling windows
            self._total_latencies.append(total_ms)
            self._model_latencies.append(inf_ms)
            self._shap_latencies.append(shp_ms)
            self._preprocess_latencies.append(pre_ms)
            self._validation_latencies.append(val_ms)
            self._ordering_latencies.append(ord_ms)
            self._formatting_latencies.append(fmt_ms)
            self._serialization_latencies.append(ser_ms)

            self._request_timestamps.append(now)

            stages_ms = {
                "validation": val_ms,
                "preprocessing": pre_ms,
                "feature_ordering": ord_ms,
                "inference": inf_ms,
                "shap": shp_ms,
                "explanation_formatting": fmt_ms,
                "serialization": ser_ms,
            }
            event = {
                "prediction_id": self.total_predictions,
                "timestamp": now_iso,
                "model_name": model_name,
                "total_ms": total_ms,
                "total_latency_ms": total_ms,
                "stages_ms": stages_ms,
                "inference_latency_ms": inf_ms,
                "shap_latency_ms": shp_ms,
                "cache_hit": cache_hit,
                "shap_cache_hit": shap_cache_hit,
            }
            self._timeline.append(event)
            return event

    def _compute_stats(self, values: collections.deque) -> Dict[str, float]:
        """Compute robust percentiles and summary statistics on a bounded float deque."""
        if not values:
            return {
                "mean": 0.0, "median": 0.0, "p50": 0.0, "p90": 0.0,
                "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0,
                "std": 0.0, "count": 0,
            }

        arr = np.array(values, dtype=np.float64)
        return {
            "mean": round(float(np.mean(arr)), 3),
            "median": round(float(np.median(arr)), 3),
            "p50": round(float(np.percentile(arr, 50)), 3),
            "p90": round(float(np.percentile(arr, 90)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
            "p99": round(float(np.percentile(arr, 99)), 3),
            "min": round(float(np.min(arr)), 3),
            "max": round(float(np.max(arr)), 3),
            "std": round(float(np.std(arr)), 3),
            "count": len(arr),
        }

    def get_throughput(self) -> Dict[str, Any]:
        """Derive real-time throughput metrics from rolling request window."""
        with self._lock:
            now = time.perf_counter()
            uptime = max(0.001, now - self._boot_time)

            # Count requests in last 60 seconds
            cutoff_1m = now - 60.0
            reqs_1m = sum(1 for ts in self._request_timestamps if ts >= cutoff_1m)

            # Requests in last 1 second
            cutoff_1s = now - 1.0
            reqs_1s = sum(1 for ts in self._request_timestamps if ts >= cutoff_1s)

            total_cache = self.cache_hits + self.cache_misses
            hit_rate = (self.cache_hits / total_cache * 100.0) if total_cache > 0 else 100.0
            miss_rate = 100.0 - hit_rate

            return {
                "total_predictions": self.total_predictions,
                "successful_predictions": self.successful_predictions,
                "failed_predictions": self.failed_predictions,
                "requests_per_second": round(float(reqs_1s), 2),
                "requests_per_minute": round(float(reqs_1m), 2),
                "uptime_seconds": round(uptime, 2),
                "requests_last_minute": reqs_1m,
                "cache_hit_rate": round(hit_rate, 2),
                "cache_miss_rate": round(miss_rate, 2),
                "average_requests_per_second_since_boot": round(self.total_predictions / uptime, 4),
            }

    def get_shap_diagnostics(self) -> Dict[str, Any]:
        """Return dedicated SHAP performance metrics and optimization comparison."""
        with self._lock:
            shap_stats = self._compute_stats(self._shap_latencies)
            total_stats = self._compute_stats(self._total_latencies)

            cur_shap_avg = shap_stats["mean"] if shap_stats["count"] > 0 else BASELINE_SHAP_LATENCY_MS
            cur_total_avg = total_stats["mean"] if total_stats["count"] > 0 else BASELINE_TOTAL_LATENCY_MS

            shap_reduction = max(0.0, round(((BASELINE_SHAP_LATENCY_MS - cur_shap_avg) / BASELINE_SHAP_LATENCY_MS) * 100.0, 1))
            total_reduction = max(0.0, round(((BASELINE_TOTAL_LATENCY_MS - cur_total_avg) / BASELINE_TOTAL_LATENCY_MS) * 100.0, 1))

            tot_shap = self.shap_cache_hits + self.shap_cache_misses
            hit_rate = (self.shap_cache_hits / tot_shap * 100.0) if tot_shap > 0 else 100.0

            return {
                "current_shap_average_ms": cur_shap_avg,
                "current_shap_p50_ms": shap_stats["p50"],
                "current_shap_p95_ms": shap_stats["p95"],
                "baseline_shap_latency_ms": BASELINE_SHAP_LATENCY_MS,
                "shap_latency_reduction_pct": shap_reduction,
                "current_total_average_ms": cur_total_avg,
                "baseline_total_latency_ms": BASELINE_TOTAL_LATENCY_MS,
                "total_latency_reduction_pct": total_reduction,
                "shap_cache_hit_rate": round(hit_rate, 2),
                "shap_cache_hits": self.shap_cache_hits,
                "shap_cache_misses": self.shap_cache_misses,
                "sample_count": shap_stats["count"],
            }

    def get_summary_snapshot(self) -> Dict[str, Any]:
        return self.get_performance_summary()

    def get_latency_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total": self._compute_stats(self._total_latencies),
                "inference": self._compute_stats(self._model_latencies),
                "shap": self._compute_stats(self._shap_latencies),
            }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Aggregate full latency percentiles, throughput, stages, and timeline."""
        with self._lock:
            tot = self._compute_stats(self._total_latencies)
            val = self._compute_stats(self._validation_latencies)
            pre = self._compute_stats(self._preprocess_latencies)
            ord_ = self._compute_stats(self._ordering_latencies)
            inf = self._compute_stats(self._model_latencies)
            shp = self._compute_stats(self._shap_latencies)
            fmt = self._compute_stats(self._formatting_latencies)
            ser = self._compute_stats(self._serialization_latencies)

            return {
                "throughput": self.get_throughput(),
                "shap_diagnostics": self.get_shap_diagnostics(),
                "percentiles": {
                    "mean_ms": tot["mean"],
                    "median_ms": tot["median"],
                    "p50_ms": tot["p50"],
                    "p90_ms": tot["p90"],
                    "p95_ms": tot["p95"],
                    "p99_ms": tot["p99"],
                    "min_ms": tot["min"],
                    "max_ms": tot["max"],
                    "std_dev_ms": tot["std"],
                    "sample_count": tot["count"],
                },
                "stage_latencies_ms": {
                    "validation": {"mean_ms": val["mean"], "p50_ms": val["p50"], "p95_ms": val["p95"]},
                    "preprocessing": {"mean_ms": pre["mean"], "p50_ms": pre["p50"], "p95_ms": pre["p95"]},
                    "feature_ordering": {"mean_ms": ord_["mean"], "p50_ms": ord_["p50"], "p95_ms": ord_["p95"]},
                    "model_inference": {"mean_ms": inf["mean"], "p50_ms": inf["p50"], "p95_ms": inf["p95"]},
                    "shap_generation": {"mean_ms": shp["mean"], "p50_ms": shp["p50"], "p95_ms": shp["p95"]},
                    "explanation_formatting": {"mean_ms": fmt["mean"], "p50_ms": fmt["p50"], "p95_ms": fmt["p95"]},
                    "serialization": {"mean_ms": ser["mean"], "p50_ms": ser["p50"], "p95_ms": ser["p95"]},
                },
                "timeline": list(self._timeline),
            }


# Global Telemetry Engine Singleton
telemetry_engine = TelemetryEngine()
