# %%
from pathlib import Path
import os
import time
from tempfile import NamedTemporaryFile

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv
from cloudpathlib import CloudPath, AzureBlobClient
from datasets import Dataset, get_dataset_config_names

from qurating.scoring_projects.fwe_fortified.utils import get_partition_subsets

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
client = AzureBlobClient(**AZURE_CLIENT_KWARGS)


def load_subset(subset):
    df_list = []
    batchi = 0
    while True:
        df = load_batch(subset, batchi)
        if df is None:
            break
        df_list.append(df)
        batchi += 1
    return pd.concat(df_list, axis=0, ignore_index=True)

def load_batch(subset, batchi):
    full_path = f"quratingscores/{model_string}/{subset}/{subset}_{batchi :04d}.parquet"
    cloud_path = CloudPath(f"az://{full_path}", client=client)
    ## return null if it doesn't exist
    if not cloud_path.exists():
        return
    ## otherwise return id and average score cols
    with NamedTemporaryFile(mode="+wb") as f:
        f.write(cloud_path.read_bytes())
        f.seek(0)
        ds = Dataset.from_parquet(f.name, keep_in_memory=True)
    score_cols = ["id", *[col for col in ds.column_names if col.endswith("_average")]]

    return ds.select_columns(score_cols).to_pandas()


ds = load_batch(configs[0], 0)

# %%
