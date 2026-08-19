import os
import requests
import json

API_KEY = os.environ["DISGENET_API_KEY"]

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
}

url = "https://api.disgenet.com/api/v1/embeddings/normalize"

params = {
    "text": "Alzheimer disease"
}

r = requests.get(url, headers=headers, params=params)

print("Status:", r.status_code)
print(json.dumps(r.json(), indent=2))