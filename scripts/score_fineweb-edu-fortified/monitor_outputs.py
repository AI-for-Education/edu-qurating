# %%
from pathlib import Path
import os
import time

import numpy as np
from dotenv import load_dotenv
from cloudpathlib import CloudPath, AzureBlobClient

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
last_run = {}
last_update_time = {}

# %%
AZURE_CLIENT_KWARGS = {
    "account_url": "https://quratingscoressa.blob.core.windows.net",
    "credential": os.getenv("QURATING_SCORES_AZURE_STORAGE_KEY"),
}
client = AzureBlobClient(**AZURE_CLIENT_KWARGS)

# should flag if no update 60 mins after last update
should_have_updated_time = 3600
outer_batch_size = 100000
for start_partition in range(n_partitions):
    subsets = get_partition_subsets(
        start_partition, start_partition + 1, n_partitions, subset_counts
    )
    n_rows_partition = sum(subset_counts[subset] for subset in subsets)
    nbatches_partition = {}
    for subset in subsets:
        nrows = subset_counts[subset]
        nbatches_float = nrows / outer_batch_size
        nbatches = np.floor(nbatches_float).astype(int)
        if nbatches < nbatches_float:
            nbatches += 1
        nbatches_partition[subset] = nbatches

    nbatches_partition_tot = sum(nbatches_partition.values())
    cnt_batches_partition = 0
    for subset in subsets:
        # first check last batch of subset. If it's done then we can move to next subset
        batchi = nbatches_partition[subset]
        full_path = (
            f"quratingscores/{model_string}/{subset}/{subset}_{batchi :04d}.parquet"
        )
        cloud_path = CloudPath(f"az://{full_path}", client=client)
        if cloud_path.exists():
            cnt_batches_partition += nbatches_partition[subset]
            # move to next subset
            continue
        ### otherwise go through batches
        for batchi in range(nbatches_partition[subset]):
            full_path = (
                f"quratingscores/{model_string}/{subset}/{subset}_{batchi :04d}.parquet"
            )
            cloud_path = CloudPath(f"az://{full_path}", client=client)
            if cloud_path.exists():
                cnt_batches_partition += 1
            else:
                # we can short-cicuit here due to knowing the order that batches are processed
                break
    new_time = time.perf_counter()
    prct = (cnt_batches_partition / nbatches_partition_tot) * 100
    diff = prct - last_run.get(start_partition, 0)
    if diff > 0:
        increase = f" (+{diff :0.2f})"
        if prct == 100:
            extra_message = " - COMPLETE!"
        else:
            extra_message = ""
        last_update_time[start_partition] = new_time
    else:
        increase = ""
        if prct == 100:
            extra_message = " - COMPLETE!"
        elif (
            new_time - last_update_time.get(start_partition, new_time)
            > should_have_updated_time
        ):
            extra_message = " - THIS ONE IS SLOW"
        else:
            extra_message = ""

    print(
        f"Partition: {start_partition} - Progress: {prct :.02f}{increase}{extra_message}"
    )
    last_run[start_partition] = prct
