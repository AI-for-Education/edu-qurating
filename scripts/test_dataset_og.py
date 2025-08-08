# %%
import time
from functools import reduce

from datasets import load_dataset, get_dataset_config_names
from transformers import AutoTokenizer
import argparse
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from qurating.prompting.openai_util import query_openai

# %%
configs = get_dataset_config_names("airtrain-ai/fineweb-edu-fortified")

print(configs)
print(len(configs))

# get last n_configs for testing
n_configs = 40
use_configs = configs[-1 : -n_configs - 1 : -1]
print(use_configs)

# %%
fw = {}
for config in use_configs:
    fw[config] = load_dataset(
        "airtrain-ai/fineweb-edu-fortified",
        name=config,
        split="train",
        streaming=True,
    )

# %%
### length of dataset
fw_len = {config: fw_.info.splits["train"].num_examples for config, fw_ in fw.items()}
fw_nshards = {config: fw_.num_shards for config, fw_ in fw.items()}

print(fw_len)
print(fw_nshards)

# %%
### manually split datasets into shards
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
n = 500
rng = np.random.default_rng(seed=2163454098)
n_samples = rng.multinomial(n=n, pvals=index_probability, size=1).ravel()

print(index_probability)
print(n_samples / n)
print(n_samples)
print(n_samples.sum())

# %%
#### shuffle shards separately
buffer_size = 1000
main_seed = 72353534
rng = np.random.default_rng(seed=main_seed)
config_seeds = rng.integers(low=0, high=2 ^ 32 - 1, size=len(fw_sharded))

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
### draw samples from each shard according to n_samples distribution

### flatten shards
flat_shards = reduce(
    lambda a, b: a + b, [ds_list for ds_list in fw_sharded_shuffled.values()]
)

p = Parallel(n)
samples = []
for n_s, ds in zip(tqdm(n_samples), flat_shards):
    if n_s > 0:
        samples.extend(list(ds.take(n_s)))


# %%
### get first n ids from each dataset
n = 50000
fw_first_ids = {}
for config, fw_ in tqdm(fw.items()):
    fw_first_ids[config] = [row["id"] for row in fw_.select_columns("id").take(5000)]

# %%
### shuffle datasets and see how many ids are taken from the first n
### shuffle on dataset after applying skip(0) to ensure that the shards aren't shuffled
### for the test
buffer_sizes = [1000, 10000, 100000]
for buffer_size in buffer_sizes:
    print(f"buffer size: {buffer_size}")
    shuff_first_ids = {}
    for config, fw_ in tqdm(fw.items()):
        shuff = fw_.skip(0).shuffle(seed=72353534, buffer_size=buffer_size)
        shuff_first_ids[config] = [
            row["id"] for row in shuff.select_columns("id").take(5000)
        ]

    intersect_first_ids = {}
    for config in fw:
        intersect_first_ids[config] = set(shuff_first_ids[config]).intersection(
            set(fw_first_ids[config])
        )
        print(len(intersect_first_ids[config]) / len(shuff_first_ids[config]))

# %%
### sample trials
rng = np.random.default_rng(seed=72353534)

sampidx = rng.permutation(fw_len)

for i in sampidx[:100]:
    print(i)
    samp = fw.skip(i)
    print(next(iter(samp))["dump"])

# %%
ds = fw.select_columns("text")
samp = ds.skip(86000)
print(next(iter(samp)))

# %%
### test built-in shuffle
batch_size = 1
rawi = 50000
maxi = np.ceil(rawi / batch_size)
shuff = fw.shuffle(seed=72353534, buffer_size=10000).batch(batch_size=batch_size)

st_global = time.perf_counter()
st = st_global
for i, samp in enumerate(shuff):
    if i > maxi:
        break
    print(i)
    print(samp["dump"])
    print(time.perf_counter() - st)
    st = time.perf_counter()

final_time = time.perf_counter() - st_global
print(final_time)
print(final_time / (maxi * batch_size))

# %%
from qurating.prompting.score_pairwise import Comparator
