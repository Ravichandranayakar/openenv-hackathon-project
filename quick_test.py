import requests
import json

# Test reset
print("Testing reset...")
r = requests.post("http://127.0.0.1:8000/reset", json={})
print(f"RESET status: {r.status_code}")

# Test step
print("\nTesting step after reset...")
body = {"action": {"action_type": "classify_issue", "classification": "billing"}}
s = requests.post("http://127.0.0.1:8000/step", json=body)
print(f"STEP status: {s.status_code}")
msg = s.json()["observation"]["resolution_message"]
print(f"Response: {msg}")
