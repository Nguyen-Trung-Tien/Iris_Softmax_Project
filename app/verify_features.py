import requests
import time
import os

BASE_URL = "http://localhost:5000"

def test_api():
    print(">>> Testing APIs...")

    # 1. Train to ensure model exists
    print("1. Triggering Training...")
    r = requests.post(f"{BASE_URL}/train")
    print(f"   Train Trigger: {r.status_code}")
    
    # Wait for training
    for _ in range(10):
        time.sleep(2)
        status = requests.get(f"{BASE_URL}/train_status").json()
        print(f"   Status: {status['running']}, Epoch: {status.get('epoch')}")
        if not status['running'] and status.get('total', 0) > 0:
            break
            
    # 2. List Models
    print("2. Listing Models...")
    r = requests.get(f"{BASE_URL}/api/models")
    if r.status_code == 200:
        models = r.json()
        print(f"   Found {len(models)} models.")
        if len(models) > 0:
            print(f"   Latest: {models[0]['filename']}")
    else:
        print(f"   FAILED: {r.text}")

    # 3. Explain
    print("3. Explain Model...")
    r = requests.get(f"{BASE_URL}/api/explain")
    if r.status_code == 200:
        print(f"   Explanation: {r.json()}")
    else:
        print(f"   FAILED: {r.text}")

    # 4. Lab Train
    print("4. Lab Training...")
    payload = {"alpha": 0.01, "epochs": 5}
    r = requests.post(f"{BASE_URL}/api/lab/train", json=payload)
    if r.status_code == 200:
        res = r.json()
        print(f"   Lab Result: Accuracy={res.get('new_accuracy')}, Epochs={len(res.get('loss', []))}")
    else:
        print(f"   FAILED: {r.text}")

if __name__ == "__main__":
    try:
        test_api()
    except Exception as e:
        print(f"Test Failed: {e}")
