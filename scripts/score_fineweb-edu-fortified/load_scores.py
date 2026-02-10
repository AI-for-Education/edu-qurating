# %%
from pathlib import Path
import os
import time
from tempfile import NamedTemporaryFile

from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv
from cloudpathlib import CloudPath, AzureBlobClient
from datasets import Dataset

from qurating.scoring_projects.fwe_fortified.utils import (
    init_dataset,
    get_partition_subsets,
)

load_dotenv(override=True)

# %%
n_partitions = 32
fw, subset_counts, fw_nshards = init_dataset(wait=5)

model_name = "AI-for-Education/qurater_gemma-3-4b-pt_ds-ours_v2-200000"
model_string = [
    substr for substr in Path(model_name).parts if substr.startswith("qurater_")
]
assert len(model_string) == 1
model_string = model_string[0]


# %%
def scores_generator():
    AZURE_CLIENT_KWARGS = {
        "account_url": "https://quratingscoressa.blob.core.windows.net",
        "credential": os.getenv("QURATING_SCORES_AZURE_STORAGE_KEY"),
    }
    client = AzureBlobClient(**AZURE_CLIENT_KWARGS)

    for start_partition in range(n_partitions):
        subsets = get_partition_subsets(
            start_partition, start_partition + 1, n_partitions, subset_counts
        )
        for subset in subsets:
            batchi = 0
            full_path = (
                f"quratingscores/{model_string}/{subset}/{subset}_{batchi :04d}.parquet"
            )
            cloud_path = CloudPath(f"az://{full_path}", client=client)
            while cloud_path.exists():
                with NamedTemporaryFile(mode="+wb") as f:
                    f.write(cloud_path.read_bytes())
                    f.seek(0)
                    ds = Dataset.from_parquet(f.name, keep_in_memory=True)
                score_cols = [
                    col for col in ds.column_names if col.endswith("_average")
                ]
                for row in ds:
                    yield {key: val for key, val in row.items() if key in score_cols}
                batchi += 1
                full_path = f"quratingscores/{model_string}/{subset}/{subset}_{batchi :04d}.parquet"
                cloud_path = CloudPath(f"az://{full_path}", client=client)


# %%
# scores_rows = []
# for i, row in tqdm(enumerate(scores_generator()), total=sum(subset_counts.values())):
#     scores_rows.append(row)

# %%
AZURE_CLIENT_KWARGS = {
    "account_url": "https://quratingscoressa.blob.core.windows.net",
    "credential": os.getenv("QURATING_SCORES_AZURE_STORAGE_KEY"),
}
client = AzureBlobClient(**AZURE_CLIENT_KWARGS)


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

ds = load_batch(next(iter(subset_counts)), 0)

# %%
