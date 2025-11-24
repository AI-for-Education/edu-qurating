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

from datasets import load_dataset, get_dataset_config_names, Dataset
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from dotenv import load_dotenv

from qurating.constants import DATASETS_DIR, ROOT, RESULTS_DIR
from qurating.inference import ModelAnnotator, TokenizeAndChunk

load_dotenv(override=True)

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
n = 100
rng = np.random.default_rng(seed=multinomial_seed)
n_samples = rng.multinomial(n=n, pvals=index_probability, size=1).ravel()

print(index_probability)
print(n_samples / n)
print(n_samples)
print(n_samples.sum())

# %%
#### shuffle shards separately
buffer_size = min(n * 2, 20000)
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

model = (
    ROOT
    / "test_training_output"
    / "qurater_gemma-3-4b-pt_bsz512_lr5e-5_epochs2_warmup0.1_conf0.5_labeltemp1.0_ds-ours_v2-500000-200000-512-72353534-gpt-4.1-mini-logprobs"
    / "checkpoint-352"
)

batch_size = 20

annotator = ModelAnnotator(str(model), labels, batch_size)


# %%
### function to draw samples from each shard according to n_samples distribution
def take_ns(ds, ns, shard_info):
    if ns > 0:
        try:
            return list(ds.take(ns)), shard_info
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
njobs = int(40 / (0.8 * (buffer_size / 10000)))

st = time.perf_counter()
p = Parallel(n_jobs=njobs, verbose=80, backend="threading")
samples = p(
    delayed(take_ns)(ds, ns, shard_info)
    for ds, ns, shard_info in zip(flat_shards, n_samples, flat_shard_info)
)
print(time.perf_counter() - st)

retry_info = [s[1] for s in samples if s is not None and s[0] is None]

st = time.perf_counter()
p = Parallel(n_jobs=50, verbose=80, backend="threading")
retry_samples = p(
    delayed(take_ns)(ds, ns, shard_info)
    for ds, ns, shard_info in zip(flat_shards, n_samples, flat_shard_info)
    if shard_info in retry_info
)
print(time.perf_counter() - st)

fixed_samples = []
for s in samples:
    if s is not None:
        if s[0] is None:
            cand = [srep for srep in retry_samples if srep[1] == s[1]]
            assert len(cand) == 1
            cand = cand[0]
            if cand[0] is not None:
                s = cand
    fixed_samples.append(s)

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

sampled_ds_final_shuffled.to_parquet(
    DATASETS_DIR / f"fwe-fortified_sampled-{n}_seed-{main_seed}.parquet"
)
