import urllib.request, json, time

time.sleep(1)

def get(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))

def post(url, data):
    payload = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))

print("=== 1. TEST GET /api/model-registry ===")
reg = get("http://127.0.0.1:8000/api/model-registry")
print("Active Champion:", reg.get("active_champion"), "| Version:", reg.get("active_version"))
print("Champion Role:", reg.get("champion", {}).get("role"), "| Is Serving:", reg.get("champion", {}).get("is_serving"))
print("Challenger Role:", reg.get("challenger", {}).get("role"), "| Is Serving:", reg.get("challenger", {}).get("is_serving"))
print("Registered Candidates:", list(reg.get("candidates", {}).keys()))

print("\n=== 2. TEST GET /api/system-info ===")
info = get("http://127.0.0.1:8000/api/system-info")
print("Champion Model:", info.get("champion_model"))
print("Challenger Model:", info.get("challenger_model"))
print("Registry Status:", info.get("registry_status"))
print("Promotion Count:", info.get("promotion_count"))

print("\n=== 3. TEST INFERENCE WITH CHAMPION BEFORE PROMOTION ===")
pred_before = post("http://127.0.0.1:8000/api/predict", {
    "features": {"age": 45, "sex": "male", "bmi": 28.5, "children": 2, "smoker": "no", "region": "northwest"}
})
print("Prediction using:", pred_before.get("model_name"), "| Verdict:", pred_before.get("prediction"), "| Confidence:", pred_before.get("confidence"), "%")

print("\n=== 4. TEST PROMOTION: PROMOTE LOGISTIC REGRESSION ===")
prom_resp = post("http://127.0.0.1:8000/api/model-registry/promote", {
    "model_name": "logistic_regression",
    "version": "v2.1",
    "reason": "Equal accuracy with superior ROC-AUC (0.9449 > 0.9123) and lower inference latency (0.03ms)",
    "actor": "governance_admin",
    "force": True
})
print("Promotion Response:", prom_resp)

print("\n=== 5. VERIFY REGISTRY AFTER PROMOTION ===")
reg_after = get("http://127.0.0.1:8000/api/model-registry")
print("New Champion:", reg_after.get("active_champion"), "| Is Serving:", reg_after.get("champion", {}).get("is_serving"))
print("New Challenger:", reg_after.get("challenger", {}).get("name"), "| Is Serving:", reg_after.get("challenger", {}).get("is_serving"))

print("\n=== 6. TEST INFERENCE AFTER PROMOTION (MUST USE NEW CHAMPION) ===")
pred_promoted = post("http://127.0.0.1:8000/api/predict", {
    "features": {"age": 45, "sex": "male", "bmi": 28.5, "children": 2, "smoker": "no", "region": "northwest"}
})
print("Prediction using:", pred_promoted.get("model_name"), "| Verdict:", pred_promoted.get("prediction"), "| Confidence:", pred_promoted.get("confidence"), "%")

print("\n=== 7. TEST ROLLBACK: REVERT TO PREVIOUS CHAMPION (CATBOOST) ===")
roll_resp = post("http://127.0.0.1:8000/api/model-registry/rollback", {
    "reason": "Automated fault recovery rollback to baseline CatBoost",
    "actor": "governance_admin"
})
print("Rollback Response:", roll_resp)

print("\n=== 8. VERIFY REGISTRY AFTER ROLLBACK ===")
reg_rollback = get("http://127.0.0.1:8000/api/model-registry")
print("Restored Champion:", reg_rollback.get("active_champion"), "| Is Serving:", reg_rollback.get("champion", {}).get("is_serving"))
print("Challenger:", reg_rollback.get("challenger", {}).get("name"), "| Is Serving:", reg_rollback.get("challenger", {}).get("is_serving"))

print("\n=== 9. TEST INFERENCE AFTER ROLLBACK (MUST USE RESTORED CHAMPION) ===")
pred_restored = post("http://127.0.0.1:8000/api/predict", {
    "features": {"age": 45, "sex": "male", "bmi": 28.5, "children": 2, "smoker": "no", "region": "northwest"}
})
print("Prediction using:", pred_restored.get("model_name"), "| Verdict:", pred_restored.get("prediction"), "| Confidence:", pred_restored.get("confidence"), "%")

print("\n=== 10. TEST PROMOTION HISTORY AUDIT TRAIL ===")
history = get("http://127.0.0.1:8000/api/promotion-history")
print(f"Total Audit Events Recorded: {len(history)}")
for i, evt in enumerate(history, 1):
    action = evt.get("action", "")
    from_m = str(evt.get("from_model"))
    to_m = evt.get("to_model", "")
    to_v = evt.get("to_version", "")
    act = evt.get("actor", "")
    rsn = evt.get("reason", "")
    print(f"  [{i}] {action:<18} | {from_m} -> {to_m} ({to_v}) | Actor: {act} | Reason: {rsn}")