#!/usr/bin/env python3
"""
Simple test script to validate the backend API endpoints.
Run this after starting the FastAPI server with: uvicorn app.main:app --reload
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint."""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_train_endpoint():
    """Test the training endpoint (will fail without dataset)."""
    print("\nTesting /api/train endpoint...")
    payload = {
        "dataset_filename": "insurance_claims.csv",
        "target_column": "approved",
        "test_size": 0.2,
        "random_state": 42
    }

    try:
        response = requests.post(f"{BASE_URL}/api/train", json=payload)
        if response.status_code == 404:
            print("✅ Train endpoint responded (expected 404 without dataset)")
            return True
        elif response.status_code == 200:
            print("✅ Train endpoint succeeded")
            return True
        else:
            print(f"❌ Train endpoint failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Train endpoint error: {e}")
        return False

def test_predict_endpoint():
    """Test the prediction endpoint (will fail without trained models)."""
    print("\nTesting /api/predict endpoint...")
    payload = {
        "features": {
            "age": 35,
            "income": 50000,
            "claim_amount": 2500
        },
        "model_name": "logistic_regression"
    }

    try:
        response = requests.post(f"{BASE_URL}/api/predict", json=payload)
        if response.status_code == 400:
            print("✅ Predict endpoint responded (expected 400 without trained models)")
            return True
        else:
            print(f"❌ Predict endpoint unexpected response: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Predict endpoint error: {e}")
        return False

def test_metrics_endpoint():
    """Test the metrics endpoint (will fail without trained models)."""
    print("\nTesting /api/metrics endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/metrics")
        if response.status_code == 400:
            print("✅ Metrics endpoint responded (expected 400 without trained models)")
            return True
        else:
            print(f"❌ Metrics endpoint unexpected response: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Metrics endpoint error: {e}")
        return False

def test_explain_endpoint():
    """Test the explain endpoint (will fail without trained models)."""
    print("\nTesting /api/explain endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/explain")
        if response.status_code == 400:
            print("✅ Explain endpoint responded (expected 400 without trained models)")
            return True
        else:
            print(f"❌ Explain endpoint unexpected response: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Explain endpoint error: {e}")
        return False

def main():
    """Run all API tests."""
    print("🚀 Testing Insurance Claim Prediction API")
    print("=" * 50)

    # Wait a moment for server to start
    time.sleep(2)

    tests = [
        test_health,
        test_train_endpoint,
        test_predict_endpoint,
        test_metrics_endpoint,
        test_explain_endpoint,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All API endpoints are responding correctly!")
    else:
        print("⚠️  Some tests failed. Check the server logs for details.")

if __name__ == "__main__":
    main()