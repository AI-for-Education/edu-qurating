# %%
from dotenv import load_dotenv
from transformers import AutoModel

from qurating.scoring_projects.fwe_fortified.model_manager import QuratingModelManager

load_dotenv(override=True)

UPLOAD = False

# %%
modman = QuratingModelManager()

if UPLOAD:
    modman.upload_all()

# %%
base_model = "qwen-3-4b"
model_type = "core_ed"

modman._download_s3(base_model, model_type)

model = AutoModel.from_pretrained(
    modman.model_folder(base_model, model_type, local_path=True)
)

# %%
modman.download_all()
