# %%
import re

import numpy as np
from datasets import Dataset
import pandas as pd

from qurating.inference import TokenizeAndChunk
from qurating.constants import (
    VALIDATION_DATA_DATASETS_DIR,
    TEMPLATES_DIR,
    DATA_DIR,
    VALIDATION_DATA_RESULTS_DIR,
)

RESULTS_MAPPING = {
    "general_educational": {
        "folder": "ours_v2",
        "ratings_folder": "qurater_Qwen3-Reranker-4B-seq-cls_ds-ours_v2-200000",
    },
    "FLN_student-facing": {
        "folder": None,
        "ratings_folder": "qurater_Qwen3-Reranker-4B-seq-cls_bsz512_lr5e-5_epochs2_warmup0.1_conf0.5_labeltemp1.0_ds-fwe-fortified_sampled-primary-5-pedagogical-5-FLN_student-facing-500000-200000-512-274634520-gpt-4.1-mini-logprobs",
    },
    "FLN_teacher-facing": {
        "folder": None,
        "ratings_folder": "qurater_Qwen3-Reranker-4B-seq-cls_bsz512_lr5e-5_epochs2_warmup0.1_conf0.5_labeltemp1.0_ds-fwe-fortified_sampled-pedagogical-5-FLN_teacher-facing-500000-200000-512-274634520-gpt-4.1-mini-logprobs",
    },
}


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


# %%
tokenizer_model = "AI-for-Education/qurater_Qwen3-Reranker-4B-seq-cls_ds-ours_v2-200000"
tokenizer = TokenizeAndChunk(str(tokenizer_model), "text", 512)

dataset_name = "bottom_up_sample_english_markdown.parquet"
dataset_file = VALIDATION_DATA_DATASETS_DIR / dataset_name
texts_dataset = Dataset.from_parquet(
    str(dataset_file),
).rename_column("full_text", "text")

# texts_dataset = texts_dataset.map(remove_single_newline, load_from_cache_file=False)

processed_ds = texts_dataset.map(
    tokenizer,
    batched=True,
    remove_columns=["text"],
    batch_size=500,
    load_from_cache_file=False,
)

lang_filt = np.array(processed_ds["language"]) == "English"

# %%
results_data_file = (
    VALIDATION_DATA_RESULTS_DIR
    / RESULTS_MAPPING["general_educational"]["ratings_folder"]
    / dataset_name
)
results_dataset = Dataset.from_parquet(str(results_data_file))
pedagogical_scores = results_dataset["pedagogical_structure_chunks"]

nsamples_per_col = 60
min_chunks = 3
alpha = 10

rng = np.random.default_rng(seed=823533744)

text_pairs = []
for model_type, res_info in RESULTS_MAPPING.items():
    print(model_type)

    # if res_info folder is None, folder is model type
    template_folder = res_info["folder"] or model_type

    results_data_file = (
        VALIDATION_DATA_RESULTS_DIR / res_info["ratings_folder"] / dataset_name
    )
    results_dataset = Dataset.from_parquet(str(results_data_file))

    nchunks = np.array(
        [len(chunk_list) for chunk_list in results_dataset["chunk_lengths"]]
    )

    ## document scores percentiles
    doc_score_cols = [
        col for col in results_dataset.column_names if col.endswith("_average")
    ]
    percentiles = {
        col: np.percentile(results_dataset[col], [alpha, 100 - alpha])
        for col in doc_score_cols
    }
    sample_idxs = {
        col: [
            np.flatnonzero((results_dataset[col] < prct[0]) & (nchunks >= min_chunks) & lang_filt),
            np.flatnonzero((results_dataset[col] > prct[1]) & (nchunks >= min_chunks) & lang_filt),
        ]
        for col, prct in percentiles.items()
    }

    ## sample from lower and upper percentiles per criterion
    text_chunks = []
    dimensions = []
    template_texts = []
    order = []
    indices = []
    for col in doc_score_cols:
        *prefix_parts, suffix = col.split("_")
        dimension_prefix = "_".join(prefix_parts)

        template_file = (
            TEMPLATES_DIR / template_folder / f"pairwise_{dimension_prefix}.txt"
        )
        with open(template_file, "r") as f:
            template_text = f.read()

        chunk_scores_col = f"{dimension_prefix}_chunks"

        paired_samples = []
        for doc_idx_array in sample_idxs[col]:
            samples = rng.choice(
                doc_idx_array, size=nsamples_per_col // 2, replace=False
            )
            paired_samples.append(samples)
        chunk_lists = []
        doc_idx_list = []
        for s1, s2 in zip(*paired_samples):
            if rng.uniform() >= 0.5:
                order.append("A")
                first = s1
                second = s2
            else:
                order.append("B")
                first = s2
                second = s1
            chunk_lists.append(processed_ds["chunks_token_ids"][first])
            chunk_lists.append(processed_ds["chunks_token_ids"][second])
            doc_idx_list.append(first)
            doc_idx_list.append(second)
        for doc_idx, chunk_list in zip(doc_idx_list, chunk_lists):
            chunk_scores = results_dataset[chunk_scores_col][doc_idx]
            chunk_scores_pedagogical = pedagogical_scores[doc_idx]
            doc_score = results_dataset[col][doc_idx]
            # print(doc_score)
            # print(np.mean(chunk_scores))
            if len(chunk_scores) > 2:
                use_chunk_scores = chunk_scores[1:-1]
                offset = 1
            else:
                use_chunk_scores = chunk_scores
                offset = 0
            # chunk_list_idx = (
            #     np.argmin(np.abs(np.array(use_chunk_scores) - doc_score)) + offset
            # )
            chunkidx = np.argmax(chunk_scores_pedagogical).item()
            token_chunk = chunk_list[chunkidx]
            text_chunk = tokenizer.tokenizer.decode(token_chunk)
            text_chunks.append(text_chunk)
            dimensions.append(dimension_prefix)
            template_texts.append(template_text)
            indices.append({"id": processed_ds["id"][doc_idx], "chunk": chunkidx})

    text_pairs.extend(
        [
            {
                "option_a_text": text_chunks[i],
                "option_b_text": text_chunks[i + 1],
                "option_a_id": indices[i]["id"],
                "option_b_id": indices[i + 1]["id"],
                "option_a_chunk": indices[i]["chunk"],
                "option_b_chunk": indices[i + 1]["chunk"],
                "judgement_dimension": f"{model_type}/{dimensions[i]}",
                "judgement_text": template_texts[i],
            }
            for i in range(0, len(text_chunks), 2)
        ]
    )

# # %%
# rng = np.random.default_rng(seed=823533744)
# docidx = rng.permutation(len(processed_ds))
# max_chunks = 80
# text_chunks = []
# indices = []
# for doc_idx in docidx:
#     chunk_lists = processed_ds["chunks_token_ids"][doc_idx]
#     docrng = np.random.default_rng(seed=docidx)
#     chunkidx = docrng.permutation(len(chunk_lists))[0]
#     text_chunk = tokenizer.tokenizer.decode(chunk_lists[chunkidx])
#     text_chunks.append(text_chunk)
#     indices.append({"id": processed_ds["id"][doc_idx], "chunk": chunkidx})
#     if len(text_chunks) >= max_chunks:
#         break

# text_pairs = [
#     {
#         "option_a_text": text_chunks[i],
#         "option_b_text": text_chunks[i + 1],
#         "option_a_id": indices[i]["id"],
#         "option_b_id": indices[i + 1]["id"],
#         "option_a_chunk": indices[i]["chunk"],
#         "option_b_chunk": indices[i + 1]["chunk"],
#     }
#     for i in range(0, max_chunks - 1, 2)
# ]

# # %%
# template_folders = ["ours_v2", "FLN_student-facing", "FLN_teacher-facing"]

# template_files = chain.from_iterable(
#     (TEMPLATES_DIR / tfolder).glob("*.txt") for tfolder in template_folders
# )

# template_text = {}
# for fl in template_files:
#     with open(fl, "r") as f:
#         template_text[f"{fl.parent.name}/{fl.stem}"] = f.read()

# rng = np.random.default_rng(seed=936635)

# for tp in text_pairs:
#     use_template = rng.choice(list(template_text))
#     tp["judgement_dimension"] = use_template
#     tp["judgement_text"] = template_text[use_template]

# %%
example_data_df = pd.DataFrame(text_pairs)
example_data_df.to_csv(
    DATA_DIR / "full_pairwise_data_markdown.csv", encoding="utf_8_sig"
)
