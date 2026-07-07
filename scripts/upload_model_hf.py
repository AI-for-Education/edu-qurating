# %%
import os

from dotenv import load_dotenv
from huggingface_hub import HfApi

from qurating.constants import ROOT

load_dotenv(override=True)

# %%
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path=str(ROOT / "checkpoints_preferences/checkpoint-352"),
    repo_id="AI-for-Education/qurater_Qwen3-Reranker-4B-seq-cls_ds-ours_v2-200000",
    repo_type="model",
)
