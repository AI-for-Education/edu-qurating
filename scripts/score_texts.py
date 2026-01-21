""" """

# %%
import time
from functools import reduce
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import json

from datasets import load_dataset, get_dataset_config_names, Dataset, IterableDataset
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd

from qurating.constants import DATASETS_DIR, RESULTS_DIR, VALIDATION_DATA_DATASETS_DIR
from qurating.inference import ModelAnnotator, TokenizeAndChunk

load_dotenv(override=True)

# %%
# load annotator model
print("Loading pairwise dataset...")

dataset_file = (
    RESULTS_DIR
    / "tokens_max_512"
    / "fwe-fortified_sampled-500000_seed-72353534"
    / "ours_v2"
    / "combined_gpt-4.1-mini_nexamples-200000_use-logprobs"
)
parquetf = Path(dataset_file).with_suffix(".parquet")
if parquetf.exists():
    ds = Dataset.from_parquet(str(dataset_file))
elif Path(dataset_file).is_dir():
    ds = Dataset.load_from_disk(dataset_file)
else:
    raise ValueError(f"{dataset_file} doesn't exist or is not a valid format")
labels = [
    "_".join(col.split("_")[:-1]) for col in ds.column_names if col.endswith("_average")
]
print(f"Labels: {labels}")

model = "AI-for-Education/qurater_gemma-3-4b-pt_ds-ours_v2-200000"
# model = "AI-for-Education/qurater_Qwen3-Reranker-4B-seq-cls_ds-ours_v2-200000"

annotator_batch_size = 400

annotator = ModelAnnotator(str(model), labels, annotator_batch_size)

tokenizer = TokenizeAndChunk(str(model), "text", 512)

# %%
### load texts to score
texts_dataset = Dataset.from_parquet(
    str(VALIDATION_DATA_DATASETS_DIR / "bottom_up_sample_english.parquet")
).rename_column("full_text", "text")

processed_ds = texts_dataset.map(
    tokenizer, batched=True, remove_columns=["text"], batch_size=4000
)

results = processed_ds.map(
    annotator,
    batched=True,
    with_indices=True,
    batch_size=annotator_batch_size,
    remove_columns=processed_ds.column_names,
)

# %%
