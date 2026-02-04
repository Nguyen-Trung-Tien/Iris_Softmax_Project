import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    try:
        r = requests.get(f"{BASE_URL}/health")
        print(f"[HEALTH] {r.status_code} - {r.json()}")
    except Exception as e:
        print(f"[HEALTH] Failed: {e}")

def test_monitor():
    try:
        r = requests.get(f"{BASE_URL}/api/monitoring")
        print(f"[MONITOR] {r.status_code} - {r.json()}")
    except Exception as e:
        print(f"[MONITOR] Failed: {e}")

def test_history():
    try:
        r = requests.get(f"{BASE_URL}/api/training_history")
        data = r.json()
        print(f"[HISTORY] {r.status_code} - Count: {len(data)}")
    except Exception as e:
        print(f"[HISTORY] Failed: {e}")

def test_auto_tune():
    try:
        # Mock auto tune
        print("[AUTO_TUNE] Sending request (this might take a few seconds)...")
        r = requests.post(f"{BASE_URL}/api/lab/auto_tune", json={})
        print(f"[AUTO_TUNE] {r.status_code} - {r.json().get('msg')}")
    except Exception as e:
        print(f"[AUTO_TUNE] Failed: {e}")

def test_models():
    try:
        r = requests.get(f"{BASE_URL}/api/models")
        print(f"[MODELS] {r.status_code} - Count: {len(r.json())}")
    except Exception as e:
        print(f"[MODELS] Failed: {e}")

if __name__ == "__main__":
    print("Verifying MLOps Endpoints...")
    test_health()
    test_models()
    test_monitor()
    test_history()
    # test_auto_tune() # Uncomment if server is running and you want to wait
    print("Done.")
