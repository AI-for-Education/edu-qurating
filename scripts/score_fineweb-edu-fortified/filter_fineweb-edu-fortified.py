# %%
from pathlib import Path
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory
from functools import reduce
import copy

from tqdm import tqdm
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from dotenv import load_dotenv
from cloudpathlib import CloudPath, AzureBlobClient
from datasets import Dataset, get_dataset_config_names, load_dataset

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


def filter_subset(subset, filters):
    scores = load_subset(subset)
    fwe_ds = load_dataset(
        "airtrain-ai/fineweb-edu-fortified",
        name=subset,
        split="train",
        streaming=True,
        token=True,
    )
    filtered_rows_list = []
    for (_, score_row), fwe_row in tqdm(
        zip(scores.iterrows(), fwe_ds), total=fwe_ds.info.splits["train"].num_examples
    ):
        filt_list = [score_row[col] > val for col, val in filters.items()]
        filt = reduce(lambda a, b: a and b, filt_list)
        assert score_row["id"] == fwe_row["id"]
        if filt:
            filtered_rows_list.append(
                {
                    **fwe_row,
                    **{
                        key: val
                        for key, val in score_row.items()
                        if key.endswith("_average")
                    },
                }
            )
    ds = Dataset.from_list(filtered_rows_list)
    ## save dataset
    client = AzureBlobClient(**AZURE_CLIENT_KWARGS)
    with TemporaryDirectory() as tdir:
        savedir = Path(tdir) / "dataset"
        ds.save_to_disk(str(savedir), max_shard_size="50MB")
        fl_list = sorted(savedir.glob("*"))
        for fl in tqdm(fl_list):
            full_path = (
                f"quratingfiltered/{model_string}/{subset}/{fl.relative_to(savedir)}"
            )
            cloud_path = CloudPath(f"az://{full_path}", client=client)
            cloud_path.upload_from(fl)
    client.close()


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
    client = AzureBlobClient(**AZURE_CLIENT_KWARGS)
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
### load scores
n_jobs = 30
p = Parallel(n_jobs=n_jobs, verbose=60)

df_list = p(delayed(load_subset)(subset) for subset in configs)

scores_df = pd.concat(df_list, axis=0, ignore_index=True)
scores_arr = np.array(scores_df.iloc[:, 1:])

# %%
# get percentiles
pct_edges = [50, 75, 90, 95]
pct = np.percentile(scores_arr, pct_edges, axis=0)

print(pct)

# %%
filters = {col: pct[0, i].item() for i, col in enumerate(scores_df.columns[4:])}

del scores_arr
del scores_df

p = Parallel(n_jobs=n_jobs, verbose=60)
out = p(delayed(filter_subset(subset, copy.deepcopy(filters))) for subset in configs)
