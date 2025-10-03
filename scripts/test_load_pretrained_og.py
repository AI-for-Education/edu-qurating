# %%
from pathlib import Path

from dotenv import load_dotenv
import torch

from qurating.modeling.model_factory import create_model

load_dotenv(override=True)

# %%
torch.cuda.is_available()

torch.cuda.device_count()

print(torch.cuda.get_device_name(0))
print("__CUDNN VERSION:", torch.backends.cudnn.version())
print("__Number CUDA Devices:", torch.cuda.device_count())
print("__CUDA Device Name:", torch.cuda.get_device_name(0))
print(
    "__CUDA Device Total Memory [GB]:",
    torch.cuda.get_device_properties(0).total_memory / 1e9,
)
print("Memory Usage:")
print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")

# %%
ROOT = Path(__file__).resolve().parents[1]
cpdir = ROOT / "test_training_output_2/checkpoint-360"
print(cpdir.exists())

model = create_model(cpdir)

# model = AutoModelForSequenceClassification.from_pretrained(cpdir)

# %%
model = model.to(torch.device("cuda:0"))
