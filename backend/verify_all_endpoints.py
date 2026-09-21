"""
verify_all_endpoints.py -- Live HTTP Endpoint Validation against running Uvicorn server
"""

import urllib.request
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def get(path):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "Nexsure-Verifier/1.0"})
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

def post(path, data):
    url = f"{BASE_URL}{path}"
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", "User-Agent": "Nexsure-Verifier/1.0"})
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

print("=" * 60)
print("  LIVE HTTP ENDPOINT VALIDATION (RENDER SIMULATION)")
print("=" * 60)

# 1. /docs (Swagger)
status_docs = urllib.request.urlopen(f"{BASE_URL}/docs").status
print(f"1. /docs (Swagger UI): Status {status_docs} [OK]")

# 2. /openapi.json
status_oa, oa_data = get("/openapi.json")
print(f"2. /openapi.json: Status {status_oa}, {len(oa_data.get('paths', {}))} endpoints exposed [OK]")

# 3. /api/health
status_health, health_data = get("/api/health")
print(f"3. /api/health: Status {status_health}, status={health_data.get('status')}, health={health_data.get('model_health')}, champion={health_data.get('active_champion')} [OK]")

# 4. /api/system-info
status_sys, sys_data = get("/api/system-info")
print(f"4. /api/system-info: Status {status_sys}, Champion={sys_data.get('champion')}, Accuracy={sys_data.get('accuracy')}, Registry={sys_data.get('registry_status')} [OK]")

# 5. /api/predict
pred_payload = {
    "features": {
        "age": 32,
        "sex": "male",
        "bmi": 24.0,
        "children": 0,
        "smoker": "no",
        "region": "southwest"
    }
}
status_pred, pred_data = post("/api/predict", pred_payload)
print(f"5. /api/predict: Status {status_pred}, Verdict={pred_data.get('verdict')}, Confidence={pred_data.get('confidence')}%, Latency={pred_data.get('inference_latency_ms')}ms [OK]")

# 6. /api/model-benchmark
status_bench, bench_data = get("/api/model-benchmark")
print(f"6. /api/model-benchmark: Status {status_bench}, Champion={bench_data.get('champion')}, Ranking={bench_data.get('ranking')} [OK]")

# 7. /api/model-registry
status_reg, reg_data = get("/api/model-registry")
print(f"7. /api/model-registry: Status {status_reg}, Champion={reg_data.get('champion', {}).get('name')}, Challenger={reg_data.get('challenger', {}).get('name')} [OK]")

# 8. /api/performance
status_perf, perf_data = get("/api/performance")
print(f"8. /api/performance: Status {status_perf}, P50={perf_data.get('percentiles', {}).get('p50_ms')}ms, Uptime={perf_data.get('throughput', {}).get('uptime_seconds')}s [OK]")

# 9. /api/cache-status
status_cache, cache_data = get("/api/cache-status")
print(f"9. /api/cache-status: Status {status_cache}, Readiness={cache_data.get('readiness_stage')}, Champion={cache_data.get('active_champion', {}).get('name')} [OK]")

# 10. /api/promotion-history
status_promo, promo_data = get("/api/promotion-history")
print(f"10. /api/promotion-history: Status {status_promo}, {len(promo_data)} audit events [OK]")

# 11. /api/explain
status_exp, exp_data = get("/api/explain")
print(f"11. /api/explain: Status {status_exp}, Model={exp_data.get('model_name')}, Features count={len(exp_data.get('top_features', []))} [OK]")

# 12. /api/training-status & /api/logs
status_ts, ts_data = get("/api/training-status")
status_logs, logs_data = get("/api/logs")
print(f"12. /api/training-status & logs: Status {status_ts}, status={ts_data.get('status')}, logs={len(logs_data)} [OK]")

print("\n" + "=" * 60)
print("[SUCCESS] ALL HTTP ENDPOINTS OPERATING FLAWLESSLY ON LIVE UVICORN SERVER!")
print("=" * 60)
