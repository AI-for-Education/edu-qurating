# %%
from pathlib import Path
import re

from datasets import Dataset, load_dataset
from dotenv import load_dotenv

from qurating.constants import (
    RESULTS_DIR,
    VALIDATION_DATA_DATASETS_DIR,
    VALIDATION_DATA_RESULTS_DIR,
    ROOT,
)
from qurating.inference import ModelAnnotator, TokenizeAndChunk

load_dotenv(override=True)

# %%
dataset_name = "HuggingFaceTB/smollm-corpus"
dataset_subset = "cosmopedia-v2"
dataset_streaming = True
dataset_split = "train"
dataset_nrows = 50000

try:
    texts_dataset = load_dataset(
        dataset_name,
        name=dataset_subset,
        streaming=dataset_streaming,
        split=dataset_split,
    )
except Exception:
    raise ValueError(f"{dataset_name} doesn't exist or is not a valid format")

if dataset_nrows is not None:
    texts_dataset = Dataset.from_list(list(texts_dataset.take(dataset_nrows)))

# %%
texts_dataset.to_parquet(
    VALIDATION_DATA_DATASETS_DIR / f"cosmopedia-v2_sample_{dataset_nrows}.parquet"
)