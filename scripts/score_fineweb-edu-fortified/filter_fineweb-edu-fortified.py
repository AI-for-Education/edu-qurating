# %%
from pathlib import Path
import os
import copy
import subprocess

import pyarrow as pa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from dotenv import load_dotenv
from datasets import get_dataset_config_names

from qurating.scoring_projects.fwe_fortified.filtering import (
    load_score_subset,
    filter_subset,
)

load_dotenv(override=True)

# %%
configs = get_dataset_config_names("airtrain-ai/fineweb-edu-fortified")

model_name = "AI-for-Education/qurater_gemma-3-4b-pt_ds-ours_v2-200000"
model_string = [
    substr for substr in Path(model_name).parts if substr.startswith("qurater_")
]
assert len(model_string) == 1
model_string = model_string[0]

# %%
AZURE_CLIENT_KWARGS = {
    "account_url": "https://quratingscoressa.blob.core.windows.net",
    "credential": os.getenv("QURATING_SCORES_AZURE_STORAGE_KEY"),
}


# ds = load_batch(configs[0], 0)

# %%
### load scores
pool = pa.default_memory_pool()
print(f"Allocated: {pool.bytes_allocated()}, Available: {pool.max_memory()}")
n_jobs = 20
with Parallel(n_jobs=n_jobs, verbose=60) as p:
    df_list = p(
        delayed(load_score_subset)(subset, model_string, AZURE_CLIENT_KWARGS)
        for subset in configs
    )
print(f"Allocated: {pool.bytes_allocated()}, Available: {pool.max_memory()}")
subprocess.call(["rm", "-fr", "~/.cache/huggingface"], shell=True)

scores_df = pd.concat(df_list, axis=0, ignore_index=True)
scores_arr = np.array(scores_df.iloc[:, 1:])


# %%
# get percentiles
pct_edges = [50, 75, 90, 95]
pct = np.percentile(scores_arr, pct_edges, axis=0)

print(pct)

# %%
filters = {col: pct[0, i].item() for i, col in enumerate(scores_df.columns[4:])}
filter_greater_than = False

del df_list
del scores_df
del scores_arr

n_jobs = 19
complete = False
while not complete:
    try:
        with Parallel(n_jobs=n_jobs, verbose=60) as p:
            out = p(
                delayed(filter_subset)(
                    subset,
                    copy.deepcopy(filters),
                    filter_greater_than,
                    model_string,
                    AZURE_CLIENT_KWARGS,
                )
                for subset in configs
            )
        complete = True
    except Exception:
        raise
        complete = False
