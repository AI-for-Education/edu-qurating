# %%
from pathlib import Path
import os

from joblib import Parallel, delayed
from dotenv import load_dotenv
from cloudpathlib import CloudPath, AzureBlobClient
from datasets import get_dataset_config_names

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


def gen_filtered_batches_info(subset):
    batchi = 0
    while True:
        info = load_filtered_batch_info(subset, batchi)
        if info is None:
            break
        batchi += 1
        yield info


def load_filtered_batch_info(subset, batchi):
    client = AzureBlobClient(**AZURE_CLIENT_KWARGS)
    full_path = (
        f"quratingfiltered/{model_string}/{subset}/{subset}_{batchi :04d}.parquet"
    )
    cloud_path = CloudPath(f"az://{full_path}", client=client)
    ## return null if it doesn't exist
    if not cloud_path.exists():
        return
    ## otherwise return info
    info = cloud_path.stat()

    return info

# %%
def count_subset(subset):
    sizes = [
        info.st_size * 1e-9 for info in gen_filtered_batches_info(subset)
    ]
    return subset, sizes

p = Parallel(n_jobs=30, verbose=60)

out = p(delayed(count_subset)(subset) for subset in configs)

nfiles = {subset: len(sizes) for subset, sizes in out}
subset_sizes = {subset: sum(sizes) for subset, sizes in out}