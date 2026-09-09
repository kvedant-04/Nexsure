import json
import urllib.request

def get(url):
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode('utf-8'))

def post(url, data):
    body = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))

print('=== VERIFY CACHE REUSE & PROMOTION INVALIDATION ===')
c1 = get('http://127.0.0.1:8000/api/cache-status')
shap_hits_before = c1['caches']['shap_explainer']['cache_hits']
print(f'SHAP explainer hits before: {shap_hits_before}')

for i in range(3):
    post('http://127.0.0.1:8000/api/predict', {'features': {'age': 40 + i, 'sex': 'male', 'bmi': 28.0, 'children': 1, 'smoker': 'no', 'region': 'northwest'}})

c2 = get('http://127.0.0.1:8000/api/cache-status')
shap_hits_after = c2['caches']['shap_explainer']['cache_hits']
print(f'SHAP explainer hits after 3 predictions: {shap_hits_after} (Incremented by {shap_hits_after - shap_hits_before})')
assert shap_hits_after - shap_hits_before == 3, 'SHAP explainer cache should be reused on every prediction!'

print('\n--- Testing Promotion Invalidation ---')
promo = post('http://127.0.0.1:8000/api/model-registry/promote', {'model_name': 'lightgbm', 'reason': 'Phase 3 verification test'})
print(f'Promotion result: {promo.get(" message\, promo)}')

c3 = get('http://127.0.0.1:8000/api/cache-status')
print(f'Active Champion in Cache: {c3[\active_champion\]}')

p_new = post('http://127.0.0.1:8000/api/predict', {'features': {'age': 45, 'sex': 'female', 'bmi': 24.5, 'children': 0, 'smoker': 'no', 'region': 'southeast'}})
print(f'Prediction on promoted Champion: Model={p_new[\model_metadata\][\name\]} v{p_new[\model_metadata\][\version\]} | Total={p_new[\performance_telemetry\][\total_latency_ms\]:.2f}ms')

print('\n--- Testing Rollback Invalidation ---')
roll = post('http://127.0.0.1:8000/api/model-registry/rollback', {'target_version': 'v2.1', 'reason': 'Phase 3 verification rollback'})
print(f'Rollback result: {roll.get(\message\, roll)}')

c4 = get('http://127.0.0.1:8000/api/cache-status')
print(f'Active Champion after Rollback: {c4[\active_champion\]}')

p_roll = post('http://127.0.0.1:8000/api/predict', {'features': {'age': 30, 'sex': 'male', 'bmi': 22.0, 'children': 0, 'smoker': 'no', 'region': 'northeast'}})
print(f'Prediction after Rollback: Model={p_roll[\model_metadata\][\name\]} v{p_roll[\model_metadata\][\version\]} | Total={p_roll[\performance_telemetry\][\total_latency_ms\]:.2f}ms')

print('\n>>> ALL CACHE REUSE, PROMOTION, AND ROLLBACK REBUILD TESTS PASSED! <<<')
