# %%
import json

from datasets import Dataset
import numpy as np
import pandas as pd

from qurating.constants import DATA_DIR, RESULTS_DIR

n_samples = 20000
seed = 72353534

NUM_EXAMPLES = 20000
TOKENS_MAX = 512
MODEL = "gpt-5-mini-2025-08-07-minimal"

annotations_dir = DATA_DIR / "annotations"

results_base = f"fwe-fortified_sampled-{n_samples}_seed-{seed}"
template_base = "ours_v2"

results_file = (
    RESULTS_DIR
    / f"tokens_max_{TOKENS_MAX}"
    / results_base
    / template_base
    / f"combined_{MODEL}_nexamples-{NUM_EXAMPLES}.parquet"
)

annotations_file = (
    annotations_dir
    / f"tokens_max_{TOKENS_MAX}"
    / results_base
    / template_base
    / f"combined_{MODEL}_nexamples-{NUM_EXAMPLES}.json"
)

# %%
dataset = Dataset.from_parquet(str(results_file), keep_in_memory=True)

with open(annotations_file) as f:
    annotations = json.load(f)


# %%
def logit_pairs_to_probs(logitsa, logitsb):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    logit_diffs = logitsa - logitsb
    probs = sigmoid(logit_diffs)
    out = np.zeros((probs.shape[0], 2))
    out[:, 0] = probs
    out[:, 1] = 1 - probs
    return out


labels = [
    "factual_accuracy",
    "pedagogical_structure",
    "lesson_engagement",
    "education_level",
    "education_level_primary",
    "education_level_secondary",
]

annotations_df = pd.DataFrame(annotations)
n_annotations = len(annotations)

annot_arrs = {}
annot_probs_arrs = {}
probs_arrs = {}
for label in labels:
    annot_arr = np.zeros((n_annotations // 2, 2))
    annot_arr[annotations_df["original_pair_id"], annotations_df["text_position"]] = (
        annotations_df[f"{label}_average"]
    )
    annot_arrs[label] = annot_arr
    annot_probs_arrs[label] = logit_pairs_to_probs(*annot_arr.T)

    probs = np.array(dataset[f"{labels[0]}_average"])
    probs_arr = probs[
        annotations_df["original_pair_id"],
        annotations_df["text_position"],
        1 - annotations_df["text_position"],
    ].reshape((-1, 2))
    probs_arrs[label] = probs_arr

for label in labels:
    print(
        ((annot_probs_arrs[label][:, 0] - probs_arrs[label][:, 0]) ** 2).mean() ** 0.5
    )
