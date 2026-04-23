# %%
import json

from fdllm import register_models, get_caller, LLMMessage
import requests
from pydantic import BaseModel

from qurating.constants import ROOT

# %%
endpoint = "http://localhost:8000/score"

body = {
    "texts": ["ab " * 1000],
    "model_type": "core_ed",
}
response = requests.post(endpoint, json=body)

scores = json.loads(response.content.decode())

# %%
# test llama.cpp server
register_models(ROOT / "custom_models.yaml")

class ResponseFormat(BaseModel):
    thinking: str
    answer: str

caller = get_caller("gemma-4-E4B")

message = LLMMessage(Role="user", Message="this is a test")

out = caller.call(message, response_schema=ResponseFormat)
print(out)
print(ResponseFormat.model_validate_json(out.Message))