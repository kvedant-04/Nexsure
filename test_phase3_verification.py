import urllib.request, json, time

time.sleep(2)

def get(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def post(url, data):
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

print("=== 1. TEST GET /api/cache-status ===")
cs = get("http://127.0.0.1:8000/api/cache-status")
print("Readiness Stage:", cs.get("readiness_stage"))
print("Is Warm:", cs.get("is_warm"))
print("Overall Status:", cs.get("overall_status"))
for layer in cs.get("layers", []):
    print(f"  - {layer.get('name'):<30}: State={layer.get('state'):<8} | Hits={layer.get('cache_hits')} | Misses={layer.get('cache_misses')} | HitRate={layer.get('hit_rate_pct')}%")

print("\n=== 2. RUN INFERENCE BURST (10 PREDICTIONS) ===")
patient_cases = [
    {"age": 28, "sex": "male", "bmi": 24.5, "children": 0, "smoker": "no", "region": "northwest"},
    {"age": 58, "sex": "female", "bmi": 34.2, "children": 2, "smoker": "yes", "region": "southeast"},
    {"age": 42, "sex": "male", "bmi": 27.8, "children": 1, "smoker": "no", "region": "northeast"},
    {"age": 31, "sex": "female", "bmi": 22.1, "children": 0, "smoker": "no", "region": "southwest"},
    {"age": 60, "sex": "male", "bmi": 38.5, "children": 3, "smoker": "yes", "region": "southeast"},
]

for i in range(10):
    case = patient_cases[i % len(patient_cases)]
    res = post("http://127.0.0.1:8000/api/predict", {"features": case})
    stages = res.get("stage_latencies_ms", {})
    tot = res.get("telemetry", {}).get("inference_latency_ms", 0.0)
    print(f"  Prediction #{i+1:02d} -> Verdict={res.get('prediction'):<8} (Conf={res.get('confidence'):.1f}%) | Total={tot:.3f}ms | Val={stages.get('validation',0):.3f}ms Pre={stages.get('preprocessing',0):.3f}ms Inf={stages.get('inference',0):.3f}ms SHAP={stages.get('shap',0):.3f}ms")

print("\n=== 3. TEST GET /api/performance ===")
perf = get("http://127.0.0.1:8000/api/performance")
tp = perf.get("throughput", {})
lat = perf.get("latency", {})
tot = lat.get("total", {})

print("Throughput:")
print(f"  Total Predictions: {tp.get('total_predictions')}")
print(f"  Requests/sec: {tp.get('requests_per_second')}")
print(f"  Uptime (seconds): {tp.get('uptime_seconds')}")
print(f"  Cache Hit Rate: {tp.get('cache_hit_rate_pct')}%")

print("\nPercentile Latency (ms):")
print(f"  Mean   : {tot.get('mean')} ms")
print(f"  Median : {tot.get('median')} ms")
print(f"  P50    : {tot.get('p50')} ms")
print(f"  P90    : {tot.get('p90')} ms")
print(f"  P95    : {tot.get('p95')} ms")
print(f"  P99    : {tot.get('p99')} ms")
print(f"  Min    : {tot.get('min')} ms")
print(f"  Max    : {tot.get('max')} ms")

print("\nStage Latency Averages (ms):")
for st_name, st_val in lat.get("stages", {}).items():
    print(f"  - {st_name:<25}: Mean={st_val.get('mean'):.3f}ms | P50={st_val.get('p50'):.3f}ms | P95={st_val.get('p95'):.3f}ms")

print("\n=== 4. TEST GET /api/system-info ===")
sys_info = get("http://127.0.0.1:8000/api/system-info")
print("Champion Model:", sys_info.get("champion_model"))
print("Warm Start Enabled:", sys_info.get("warm_start_enabled"))
print("SHAP Cache Enabled:", sys_info.get("shap_cache_enabled"))
print("Readiness Stage:", sys_info.get("readiness_stage"))
print("Cache Hit Rate:", sys_info.get("cache_hit_rate_pct"), "%")
print("P50 Latency (ms):", sys_info.get("p50_latency_ms"))
print("P95 Latency (ms):", sys_info.get("p95_latency_ms"))