# %%
from fdllm import LLMMessage, get_caller, OpenAICaller, GoogleGenAICaller
from fdllm.llmtypes import LLMCaller
from fdllm import register_models
from dotenv import load_dotenv
# import nest_asyncio

from qurating.constants import DATASETS_DIR, TEMPLATES_DIR, RESULTS_DIR, ROOT

# nest_asyncio.apply()

load_dotenv(override=True)

register_models(ROOT / "custom_models.yaml")

# %%
caller = get_caller("gemini-2.5-flash-vertex")
message = LLMMessage(Role="user", Message="hi")

# %%
caller.call(message, logprobs=True, top_logprobs=19)