"""
Script to approximately randomly sample from large fineweb-edu-fortified dataset.
The dataset is actually split over 95 different datasets, each corresponding to
a different dump of the common crawl.

Each of these datsets is further split into ~20-50 shards.

Due to the size of the datasets, we load as a streaming dataset, which complicates
randomization. Randomization of streaming datasets relies on a buffer which is filled
in memory. This buffer is filled *in order*, such that a buffer of size n will be initially
filled with the first n rows of the dataset. Samples are drawn randomly from the buffer, and
the buffer is filled with the next rows in order after each sample is taken. Shards are also
randomised, but importantly, the buffer still fills with contiguous rows from the same shard.

The only way to get true randomisation is if buffer_size == n_rows.

The way that we try to approximate better randomness is as follows:

- split each dataset manually into shards
- get the number of rows in each shard of each dataset
- calculate the total number of rows over all datasets x shards
- calculate sampling probability from each dataset x shard as n_ds_shard / n_total
- convert this to n_samples_ds_shard by sampling from a multinomial distribution
- iterate over each dataset x shard and apply the streaming dataset shuffle method,
    with a buffer of no larger than 20000 rows
- take the first n_samples_ds_shard from each shuffled dataset x shard
- concatenate all of this together into a single new dataset in memory
- shuffle the rows (true shuffle) of this dataset to interleave rows between shards

The benefit of this approach over the built-in streaming shuffle is that the final shuffled dataset
is no longer necessarily contiguous within shards.
"""

# %%
import time
from functools import reduce
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import json

from datasets import load_dataset, get_dataset_config_names, Dataset, IterableDataset
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

from qurating.constants import DATASETS_DIR, RESULTS_DIR
from qurating.inference import ModelAnnotator, TokenizeAndChunk

load_dotenv(override=True)

N = 20000
OVERSAMPLE_FACTOR = 4.0

# %%
### configs are the different datasets (95, corresponding to CC dumps)

configs = get_dataset_config_names("airtrain-ai/fineweb-edu-fortified")

print(configs)
print(len(configs))

# get last n_configs for testing
### NOTE: in fact, changed this now so we just take all of them
n_configs = len(configs)
use_configs = configs[-1 : -n_configs - 1 : -1]
print(use_configs)

# %%
# load each dataset by config name as a streaming dataset into the dict fw
fw = {}
for config in use_configs:
    fw[config] = load_dataset(
        "airtrain-ai/fineweb-edu-fortified",
        name=config,
        split="train",
        streaming=True,
        token=True,
    )

# %%
### length of each dataset
fw_len = {config: fw_.info.splits["train"].num_examples for config, fw_ in fw.items()}
# number of shards in each dataset
fw_nshards = {config: fw_.num_shards for config, fw_ in fw.items()}

print(fw_len)
print(fw_nshards)

# %%
### manually split datasets into shards and get the number of rows in each shard
fw_sharded = {}
fw_sharded_len = {}
for config, fw_ in tqdm(fw.items()):
    fw_sharded[config] = []
    sharded_len = [int(np.ceil(fw_len[config] / fw_.num_shards))] * (fw_.num_shards - 1)
    sharded_len.append(fw_len[config] - sum(sharded_len))
    fw_sharded_len[config] = sharded_len
    for i in range(fw_.num_shards):
        fw_sharded[config].append(fw_.shard(fw_.num_shards, i))

# %%
# set up seeds
main_seed = 274634520
rng = np.random.default_rng(seed=main_seed)
shard_order_seed = rng.integers(low=0, high=2 ^ 32 - 1)
config_seeds = rng.integers(low=0, high=2 ^ 32 - 1, size=len(fw_sharded))
multinomial_seed = rng.integers(low=0, high=2 ^ 32 - 1)
final_seed = rng.integers(low=0, high=2 ^ 32 - 1)

# %%
### calculate probability of drawing from any single config / shard combination
fw_sharded_ratio = {}
tot = 0
for config, shard_lens in tqdm(fw_sharded_len.items()):
    tot += sum(shard_lens)
for config, shard_lens in tqdm(fw_sharded_len.items()):
    fw_sharded_ratio[config] = [shard_len / tot for shard_len in shard_lens]

### flatten config / shard probibility distribution
index_probability = np.hstack([shard_prob for shard_prob in fw_sharded_ratio.values()])

# %%
### sample the number from each config / shard from a multinomial with p = index_probability
rng = np.random.default_rng(seed=multinomial_seed)
n_samples = rng.multinomial(n=N, pvals=index_probability, size=1).ravel()

print(index_probability)
print(n_samples / N)
print(n_samples)
print(n_samples.sum())

# %%
#### shuffle shards separately
buffer_size = min(N * 2, 20000)
print(f"Buffer size: {buffer_size}")

shard_order_rng = np.random.default_rng(seed=shard_order_seed)

fw_sharded_shuffled = {}
for (config, fw_), config_seed in zip(tqdm(fw_sharded.items()), config_seeds):
    fw_sharded_shuffled[config] = []
    config_rng = np.random.default_rng(seed=config_seed)
    shard_seeds = config_rng.integers(low=0, high=2 ^ 32 - 1, size=len(fw_))
    for shard, shard_seed in zip(tqdm(fw_, leave=False), shard_seeds):
        fw_sharded_shuffled[config].append(
            shard.shuffle(seed=shard_seed, buffer_size=buffer_size)
        )


# %%
# load annotator model
print("Loading pairwise dataset...")

dataset_file = (
    RESULTS_DIR
    / "tokens_max_512"
    / "fwe-fortified_sampled-500000_seed-72353534"
    / "ours_v2"
    / "combined_gpt-4.1-mini_nexamples-200000_use-logprobs"
)
parquetf = Path(dataset_file).with_suffix(".parquet")
if parquetf.exists():
    ds = Dataset.from_parquet(str(dataset_file))
elif Path(dataset_file).is_dir():
    ds = Dataset.load_from_disk(dataset_file)
else:
    raise ValueError(f"{dataset_file} doesn't exist or is not a valid format")
labels = [
    "_".join(col.split("_")[:-1]) for col in ds.column_names if col.endswith("_average")
]
print(f"Labels: {labels}")

model = "AI-for-Education/qurater_gemma-3-4b-pt_ds-ours_v2-200000"

annotator_batch_size = 2000

annotator = ModelAnnotator(str(model), labels, annotator_batch_size)

tokenizer = TokenizeAndChunk(str(model), "text", 512)

# %%
ds_batched = iter(next(iter(fw_sharded_shuffled.values()))[0].batch(batch_size=100))
ds_batch = Dataset.from_dict(next(ds_batched))
processed_ds = ds_batch.map(tokenizer, batched=True, remove_columns=["text"])
column_names = list(next(iter(processed_ds)))
results = processed_ds.map(
    annotator,
    batched=True,
    with_indices=True,
    remove_columns=column_names,
)

res_list = [
    {**ds_, **res}
    for res, ds_ in zip(results, ds_batch)
    if res["education_level_primary_average"] > 0
]


# %%
lock = threading.Lock()
count_lock = threading.Lock()


### function to draw samples from each shard according to n_samples distribution
def take_ns_filtered(
    ds: IterableDataset,
    ns: int,
    shard_info: dict,
    cnt: list[int],
    tot: int,
    filters: dict | None = None,
    oversample_factor: float = 1.0,
):
    if filters is None:
        filters = {
            "education_level_primary_average": (0.0, math.inf)
        }
    use_ns = int(ns * oversample_factor)
    if ns > 0:
        try:
            res_list = []
            skip = 0
            remaining_ns = ns
            for batch_ds in ds.batch(batch_size=use_ns):
                print(
                    f"Taking {use_ns}\n"
                    f"Remaining ns %: {100 * remaining_ns / ns:.02f}\n"
                    f"shard_info:\n{json.dumps(shard_info, indent=2)}\n"
                    f"Completion %: {100 * cnt[0] / tot:.02f}"
                )
                current_ds = Dataset.from_dict(batch_ds)
                print(
                    "Taking complete\n"
                    f"Remaining ns %: {100 * remaining_ns / ns:.02f}\n"
                    f"shard_info:\n{json.dumps(shard_info, indent=2)}\n"
                    f"Completion %: {100 * cnt[0] / tot:.02f}"
                )
                with lock:
                    processed_ds = current_ds.map(
                        tokenizer, batched=True, remove_columns=["text"]
                    )
                    results = processed_ds.map(
                        annotator,
                        batched=True,
                        with_indices=True,
                        batch_size=annotator_batch_size*4,
                        remove_columns=processed_ds.column_names,
                    )
                res_list_new = [
                    {**res, **curr_ds_item}
                    for res, curr_ds_item in zip(results, current_ds)
                    if all(
                        (res[field] > thresh_low) & (res[field] <= thresh_high)
                        for field, (thresh_low, thresh_high) in filters.items()
                    )
                ]
                res_list.extend(res_list_new)
                res_list = res_list[:ns]
                skip += use_ns
                remaining_ns = ns - len(res_list)
                if remaining_ns == 0:
                    break
            with count_lock:
                cnt[0] += 1
            print(
                "Shard complete\n"
                f"shard_info:\n{json.dumps(shard_info, indent=2)}\n"
                f"Completion %: {100 * cnt[0] / tot:.02f}"
            )
            return res_list, shard_info

        except Exception:
            return None, shard_info


### flatten shards
flat_shards = reduce(
    lambda a, b: a + b, [ds_list for ds_list in fw_sharded_shuffled.values()]
)
flat_shard_info = reduce(
    lambda a, b: a + b,
    [
        [{"config": config, "shard": i} for i in range(len(ds_list))]
        for config, ds_list in fw_sharded_shuffled.items()
    ],
)

print(len(flat_shards))

## use joblib with threading to sample from the shards concurrently.
# As the main bottleneck is downloading the data to fill the buffer, threading
# a decent speed up. Didn't observe much additional improvement with multi-processing.
filters = {
    "education_level_primary_average": (5.0, math.inf),
    "pedagogical_structure_average": (5.0, math.inf)
}
assert all(filter_name.removesuffix("_average") in labels for filter_name in filters)

njobs = int(80 / (0.8 * (buffer_size / 10000)))

print("Running threaded sampling")
print(f"njobs: {njobs}")
cnt = [0]
st = time.perf_counter()
with ThreadPoolExecutor(max_workers=njobs) as executor:
    futures = [
        executor.submit(
            take_ns_filtered,
            ds=ds,
            ns=ns,
            shard_info=shard_info,
            cnt=cnt,
            tot=len(flat_shards),
            filters=filters,
            oversample_factor=OVERSAMPLE_FACTOR,
        )
        for ds, ns, shard_info in zip(flat_shards, n_samples, flat_shard_info)
    ]
    samples = [future.result() for future in as_completed(futures)]
print(time.perf_counter() - st)

fixed_samples = samples

# %%
nzsamples = [s for s in fixed_samples if s is not None]
dataset_list = shard_order_rng.permuted(
    [d for s in nzsamples if s[0] is not None for d in s[0]]
).tolist()

sampled_ds = Dataset.from_list(dataset_list)

## final shuffle of samples within original shards
sampled_ds_final_shuffled = sampled_ds.shuffle(seed=final_seed).flatten_indices(
    keep_in_memory=True
)

if len(sampled_ds_final_shuffled) > 10000:
    outfile = DATASETS_DIR / f"fwe-fortified_sampled-primary-5-pedagogical-5-{N}_seed-{main_seed}"
    sampled_ds_final_shuffled.save_to_disk(outfile, max_shard_size="200MB")
else:
    outfile = DATASETS_DIR / f"fwe-fortified_sampled-primary-5-pedagogical-5-{N}_seed-{main_seed}.parquet"
    sampled_ds_final_shuffled.to_parquet(outfile)
