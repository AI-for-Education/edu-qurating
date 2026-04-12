# %%
import json

import requests

# %%
endpoint = "http://localhost:8000/score"

body = {
    "texts": ["ab " * 1000],
    "model_type": "core_ed",
}
response = requests.post(endpoint, json=body)

scores = json.loads(response.content.decode())