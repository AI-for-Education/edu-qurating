""" """

# %%
from pathlib import Path
import re

from datasets import Dataset
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
# load annotator model
print("Loading pairwise dataset...")

dataset_file = (
    RESULTS_DIR
    / "tokens_max_512"
    / "fwe-fortified_sampled-pedagogical-5-500000_seed-274634520"
    / "FLN_teacher-facing"
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

# model = "AI-for-Education/qurater_gemma-3-4b-pt_ds-ours_v2-200000"

# model = "AI-for-Education/qurater_Qwen3-Reranker-4B-seq-cls_ds-ours_v2-200000"
# model = str(
#     ROOT
#     / "checkpoints-preferences"
#     / "qurater_Qwen3-Reranker-4B-seq-cls_bsz512_lr5e-5_epochs2_warmup0.1_conf0.5_labeltemp1.0_ds-fwe-fortified_sampled-primary-5-pedagogical-5-FLN_student-facing-500000-200000-512-274634520-gpt-4.1-mini-logprobs"
#     / "checkpoint-340"
# )
model = str(
    ROOT
    / "checkpoints-preferences"
    / "qurater_Qwen3-Reranker-4B-seq-cls_bsz512_lr5e-5_epochs2_warmup0.1_conf0.5_labeltemp1.0_ds-fwe-fortified_sampled-pedagogical-5-FLN_teacher-facing-500000-200000-512-274634520-gpt-4.1-mini-logprobs"
    / "checkpoint-312"
)

annotator_batch_size = 2000

annotator = ModelAnnotator(str(model), labels, annotator_batch_size)

tokenizer = TokenizeAndChunk(str(model), "text", 512)

# %%
### load texts to score
texts_dataset = Dataset.from_parquet(
    str(VALIDATION_DATA_DATASETS_DIR / "bottom_up_sample_english.parquet")
).rename_column("full_text", "text")

texts_dataset.add_column("record_id", texts_dataset["index"])


def remove_single_newline(row):
    row["text"] = re.sub("(.)\n(?!\n)", r"\1 ", row["text"])
    return row


texts_dataset = texts_dataset.map(remove_single_newline)

processed_ds = texts_dataset.map(
    tokenizer,
    batched=True,
    remove_columns=["text"],
    batch_size=4000,
    load_from_cache_file=False,
)

results = processed_ds.map(
    annotator,
    batched=True,
    with_indices=True,
    batch_size=annotator_batch_size,
    remove_columns=[col for col in processed_ds.column_names if col != "record_id"],
    load_from_cache_file=False,
)

# %%
model_string = [sub for sub in model.split("/") if sub.startswith("qurater_")][0]

outfile = (
    VALIDATION_DATA_RESULTS_DIR / model_string / "bottom_up_sample_english.parquet"
)

outfile.parent.mkdir(exist_ok=True, parents=True)
results.to_parquet(outfile)

# %%
