# %%
import re
from itertools import chain

import numpy as np
from datasets import Dataset
import pandas as pd

from qurating.inference import TokenizeAndChunk
from qurating.constants import VALIDATION_DATA_DATASETS_DIR, TEMPLATES_DIR, DATA_DIR

# %%
model = "AI-for-Education/qurater_gemma-3-4b-pt_ds-ours_v2-200000"

tokenizer = TokenizeAndChunk(str(model), "text", 512)

dataset_name = "bottom_up_sample_english.parquet"

dataset_file = VALIDATION_DATA_DATASETS_DIR / dataset_name

texts_dataset = Dataset.from_parquet(
    str(dataset_file),
).rename_column("full_text", "text")


def remove_single_newline(row):
    text = row["text"]
    # replace newline followed by whitespace with newline
    text = re.sub(r"\n\s", r"\n", text)
    # replace 3 or more newlines with 2
    text = re.sub(r"(\n{3,})|\n\n", lambda x: "\n\n" if x.group(1) else "\n", text)
    # # # replace single newline with nothing
    # text = re.sub(r"(.)\n(?!\n)", r"\1 ", text)
    row["text"] = text
    return row


texts_dataset = texts_dataset.map(remove_single_newline, load_from_cache_file=False)

processed_ds = texts_dataset.map(
    tokenizer,
    batched=True,
    remove_columns=["text"],
    batch_size=500,
    load_from_cache_file=False,
)

# %%
rng = np.random.default_rng(seed=823533744)
docidx = rng.permutation(len(processed_ds))
max_chunks = 80
text_chunks = []
indices = []
for idx in docidx:
    chunk_list = processed_ds["chunks_token_ids"][idx]
    docrng = np.random.default_rng(seed=docidx)
    chunkidx = docrng.permutation(len(chunk_list))[0]
    text_chunk = tokenizer.tokenizer.decode(chunk_list[chunkidx])
    text_chunks.append(text_chunk)
    indices.append({"id": processed_ds["id"][idx], "chunk": chunkidx})
    if len(text_chunks) >= max_chunks:
        break

text_pairs = [
    {
        "option_a_text": text_chunks[i],
        "option_b_text": text_chunks[i + 1],
        "option_a_id": indices[i]["id"],
        "option_b_id": indices[i + 1]["id"],
        "option_a_chunk": indices[i]["chunk"],
        "option_b_chunk": indices[i + 1]["chunk"],
    }
    for i in range(0, max_chunks - 1, 2)
]

# %%
template_folders = ["ours_v2", "FLN_student-facing", "FLN_teacher-facing"]

template_files = chain.from_iterable(
    (TEMPLATES_DIR / tfolder).glob("*.txt") for tfolder in template_folders
)

template_text = {}
for fl in template_files:
    with open(fl, "r") as f:
        template_text[f"{fl.parent.name}/{fl.stem}"] = f.read()

rng = np.random.default_rng(seed=936635)

for tp in text_pairs:
    use_template = rng.choice(list(template_text))
    tp["judgement_dimension"] = use_template
    tp["judgement_text"] = template_text[use_template]

# %%
example_data_df = pd.DataFrame(text_pairs)
example_data_df.to_csv(DATA_DIR / "example_pairwise_data.csv", encoding="utf_8_sig")
