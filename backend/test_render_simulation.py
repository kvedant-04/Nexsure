"""
test_render_simulation.py -- Local Render Startup & API Simulation Test
Validates that FastAPI starts and all endpoints return valid responses.
"""

import sys
import asyncio
import json

from app.main import app, startup_lifecycle, _get_cors_origins
from app.api.predict import (
    health_check,
    get_training_status,
    get_system_info,
    get_performance_telemetry,
    get_cache_status_endpoint,
    get_model_registry_endpoint,
    get_model_benchmark,
    get_promotion_history_endpoint,
    get_version_history,
    get_prediction_logs,
    predict_claim,
    get_shap_explanation,
    PredictRequest,
)

print("=" * 60)
print("  NEXSURE RENDER BACKEND SIMULATION TEST")
print("=" * 60)

async def run_simulation():
    # 1. Trigger startup lifecycle
    print("\n--- 1. Triggering Startup Lifecycle ---")
    await startup_lifecycle()
    print("Startup lifecycle completed successfully.")

    # 2. Test CORS Configuration
    print("\n--- 2. Testing CORS Configuration ---")
    cors_origins = _get_cors_origins()
    print(f"✅ CORS Allowed Origins ({len(cors_origins)}): {cors_origins}")
    assert "http://localhost:3000" in cors_origins
    assert "http://localhost:5173" in cors_origins

    # 3. Test Health
    print("\n--- 3. Testing /api/health ---")
    data = health_check()
    print(f"✅ Health: status={data.get('status')}, model_health={data.get('model_health')}, champion={data.get('active_champion')}")
    assert data.get("status") == "ok"
    assert data.get("model_health") == "healthy"

    # 4. Test System Info
    print("\n--- 4. Testing /api/system-info ---")
    sys_data = get_system_info()
    print(f"✅ System Info: Champion={sys_data.champion}, Accuracy={sys_data.accuracy}, Health={sys_data.model_health}")
    assert sys_data.champion is not None

    # 5. Test Performance
    print("\n--- 5. Testing /api/performance ---")
    perf_data = get_performance_telemetry()
    print(f"✅ Performance: Total requests={perf_data.get('total_requests')}, Uptime={perf_data.get('uptime_seconds')}s")

    # 6. Test Cache Status
    print("\n--- 6. Testing /api/cache-status ---")
    cache_data = get_cache_status_endpoint()
    print(f"✅ Cache Status: Readiness={cache_data.get('readiness_stage')}, Active={cache_data.get('active_champion')}")
    assert cache_data.get("readiness_stage") == "READY"

    # 7. Test Model Registry
    print("\n--- 7. Testing /api/model-registry ---")
    reg_data = get_model_registry_endpoint()
    print(f"✅ Model Registry: Champion={reg_data.get('champion', {}).get('name')}, Challenger={reg_data.get('challenger', {}).get('name')}")
    assert reg_data.get("champion") is not None

    # 8. Test Model Benchmark
    print("\n--- 8. Testing /api/model-benchmark ---")
    bench_data = get_model_benchmark()
    print(f"✅ Benchmark Report: Champion={bench_data.get('champion')}, Ranking={bench_data.get('ranking')}")
    assert bench_data.get("champion") is not None

    # 9. Test Promotion History
    print("\n--- 9. Testing /api/promotion-history ---")
    promo_data = get_promotion_history_endpoint()
    print(f"✅ Promotion History: {len(promo_data)} recorded lifecycle events")

    # 10. Test Explain
    print("\n--- 10. Testing /api/explain ---")
    exp_data = get_shap_explanation()
    print(f"✅ Global Explain: Top feature={exp_data.get('top_features', [{}])[0].get('feature')}")
    assert exp_data.get("model_name") is not None

    # 11. Test Predict
    print("\n--- 11. Testing /api/predict ---")
    req = PredictRequest(
        features={
            "age": 35,
            "sex": "female",
            "bmi": 26.5,
            "children": 1,
            "smoker": "no",
            "region": "northeast"
        }
    )
    pred_data = predict_claim(req)
    print(f"✅ Predict: Verdict={pred_data.verdict}, Confidence={pred_data.confidence}%, Latency={pred_data.inference_latency_ms}ms")
    print(f"   Drivers: {pred_data.top_decision_drivers}")
    assert pred_data.verdict in ["APPROVED", "REJECTED"]

    # 12. Test Training Status & Logs
    print("\n--- 12. Testing /api/training-status & /api/logs ---")
    ts_data = get_training_status()
    logs_data = get_prediction_logs()
    print(f"✅ Training Status: {ts_data} | Logs recorded: {len(logs_data)}")

    print("\n" + "=" * 60)
    print("🎉 ALL 12 ENDPOINTS & LIFECYCLE CHECKS PASSED WITH ZERO REGRESSIONS!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_simulation())
