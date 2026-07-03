import requests
import json
import os
import time

with open('config.json', 'r') as f:
    config = json.load(f)

model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])

# Allow overriding host/port via environment so client and server can run in
# separate terminals without editing code.
host = os.environ.get('HOST', '127.0.0.1')
port = os.environ.get('PORT', '8000')
URL = f"http://{host}:{port}/"

# Basic readiness check with retries to avoid racing the server startup
max_retries = int(os.environ.get('API_RETRIES', 10))
for attempt in range(1, max_retries + 1):
    try:
        resp = requests.get(URL, timeout=2)
        break
    except requests.exceptions.RequestException:
        if attempt == max_retries:
            print(f"ERROR: Could not reach server at {URL} after {max_retries} attempts")
            raise
        time.sleep(1)

# Call each API endpoint and store the responses
response1 = requests.post(
    URL + "prediction",
    json={"filepath": os.path.join(test_data_path, "testdata.csv")},
    timeout=30,
)

response2 = requests.get(URL + "scoring", timeout=30)

response3 = requests.get(URL + "summarystats", timeout=30)

response4 = requests.get(URL + "diagnostics", timeout=30)

# Combine all API responses
responses = {
    "predictions": response1.json(),
    "scoring": response2.json(),
    "summary_statistics": response3.json(),
    "diagnostics": response4.json()
}

# Write the responses to workspace
os.makedirs(model_path, exist_ok=True)
output_path = os.path.join(model_path, 'apireturns.txt')
with open(output_path, 'w') as f:
    f.write(json.dumps(responses, indent=2))

print(f"API responses written to {output_path}")
print(json.dumps(responses, indent=2))
