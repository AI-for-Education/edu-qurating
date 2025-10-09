# %%
from datasets import Dataset
import numpy as np

from qurating.constants import RESULTS_DIR

n_samples = 20000
seed = 72353534

NUM_EXAMPLES = 500
TOKENS_MAX = 512
JUDGEMENTS_MODEL = "gpt-4.1-mini"

results_base = f"fwe-fortified_sampled-{n_samples}_seed-{seed}"
template_base = "ours_v2"

results_file = (
    RESULTS_DIR
    / f"tokens_max_{TOKENS_MAX}"
    / results_base
    / template_base
    / f"combined_{JUDGEMENTS_MODEL}_nexamples-{NUM_EXAMPLES}.parquet"
)

compare_file = (
    results_file.parent / f"{results_file.stem}_use-logprobs{results_file.suffix}"
)

# %%
dataset = Dataset.from_parquet(str(results_file), keep_in_memory=True)

dataset_compare = Dataset.from_parquet(str(compare_file), keep_in_memory=True)

# %%
compare_cols = [col for col in dataset.column_names if col.endswith("_average")]

conf = 0.5
for col in compare_cols:
    x = np.array(dataset[col])[:, 0, 1]
    filt = np.abs((x - 0.5)) >= (conf / 2)
    x_c = np.array(dataset_compare[col])[:, 0, 1]
    print(np.abs(x-x_c)[filt].mean())