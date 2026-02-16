# %%
from pathlib import Path
import os
from tempfile import NamedTemporaryFile
from functools import reduce
import copy
import gc
from io import BytesIO

import pyarrow as pa
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
    pool = pa.default_memory_pool()

    fwe_ds = load_dataset(
        "airtrain-ai/fineweb-edu-fortified",
        name=subset,
        split="train",
        streaming=True,
        token=True,
    )
    fwe_iter = iter(fwe_ds)
    iteri = 0
    itertot = fwe_ds.info.splits["train"].num_examples
    for batchi, batch_scores in enumerate(gen_batches(subset)):
        full_path = (
            f"quratingfiltered/{model_string}/{subset}/{subset}_{batchi :04d}.parquet"
        )
        client = AzureBlobClient(**AZURE_CLIENT_KWARGS)
        cloud_path = CloudPath(f"az://{full_path}", client=client)
        if cloud_path.exists():
            print(f"Skipping batch: {subset} - {batchi}")
            continue
        filtered_rows_list = []
        for _, score_row in batch_scores.iterrows():
            prog = 100 * (iteri / itertot)
            if prog * 100 % 10 == 0:
                print(f"{subset}: {prog :.03f}%")
            fwe_row = next(fwe_iter)
            filt_list = [score_row[col] > val for col, val in filters.items()]
            filt = reduce(lambda a, b: a and b, filt_list)
            if score_row["id"] != fwe_row["id"]:
                errstr = f"ID mismatch: {subset} - row {iteri}"
                print(errstr)
                raise ValueError(errstr)
            if filt:
                filtered_rows_list.append(
                    {
                        **{
                            key: val
                            for key, val in fwe_row.items()
                            if key not in ["embeddings"]
                        },
                        **{
                            key: val
                            for key, val in score_row.items()
                            if key.endswith("_average")
                        },
                    }
                )
            iteri += 1
        out_ds = Dataset.from_list(filtered_rows_list)
        print(f"Uploading output dataset to: {full_path}")
        with BytesIO() as bts:
            out_ds.to_parquet(bts)
            bts.seek(0)
            cloud_path.write_bytes(bts.read())
        out_ds.cleanup_cache_files()
        print(
            f"Allocated: {pool.bytes_allocated() * 1e-6}, Available: {pool.max_memory() * 1e-6}"
        )
        gc.collect()
        print(
            f"Allocated: {pool.bytes_allocated() * 1e-6}, Available: {pool.max_memory() * 1e-6}"
        )
    # ####################################
    # ds = Dataset.from_list(filtered_rows_list)
    # del filtered_rows_list
    # ## save dataset
    # with TemporaryDirectory() as tdir:
    #     savedir = Path(tdir) / "dataset"
    #     ds.save_to_disk(str(savedir), max_shard_size="50MB")
    #     fl_list = sorted(savedir.glob("*"))
    #     for fl in tqdm(fl_list):
    #         full_path = (
    #             f"quratingfiltered/{model_string}/{subset}/{fl.relative_to(savedir)}"
    #         )
    #         cloud_path = CloudPath(f"az://{full_path}", client=client)
    #         cloud_path.upload_from(fl)
    # ds.cleanup_cache_files()
    # del ds
    # pool.release_unused()


def load_subset(subset):
    df_list = list(gen_batches(subset))
    return pd.concat(df_list, axis=0, ignore_index=True)


def gen_batches(subset):
    batchi = 0
    while True:
        df = load_batch(subset, batchi)
        if df is None:
            break
        batchi += 1
        yield df


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
        print(f"Len batch: {subset} - {batchi}: {len(ds)}")
    score_cols = ["id", *[col for col in ds.column_names if col.endswith("_average")]]

    ds_df = ds.select_columns(score_cols).to_pandas()
    ds.cleanup_cache_files()

    return ds_df


# ds = load_batch(configs[0], 0)

# %%
### load scores
pool = pa.default_memory_pool()
print(f"Allocated: {pool.bytes_allocated()}, Available: {pool.max_memory()}")
n_jobs = 30
with Parallel(n_jobs=n_jobs, verbose=60) as p:
    df_list = p(delayed(load_subset)(subset) for subset in configs)
print(f"Allocated: {pool.bytes_allocated()}, Available: {pool.max_memory()}")

scores_df = pd.concat(df_list, axis=0, ignore_index=True)
scores_arr = np.array(scores_df.iloc[:, 1:])


# %%
# get percentiles
pct_edges = [50, 75, 90, 95]
pct = np.percentile(scores_arr, pct_edges, axis=0)

print(pct)

# %%
filters = {col: pct[0, i].item() for i, col in enumerate(scores_df.columns[4:])}

del df_list
del scores_df
del scores_arr

n_jobs = 30
complete = False
while not complete:
    try:
        with Parallel(n_jobs=n_jobs, verbose=60) as p:
            out = p(
                delayed(filter_subset)(subset, copy.deepcopy(filters))
                for subset in configs
            )
        complete = True
    except Exception:
        complete = False
