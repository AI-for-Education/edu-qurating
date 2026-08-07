# %% 
import pickle

import numpy as np

from qurating.constants import CACHE_DIR

# %%
cachedir = CACHE_DIR / r"pairwise_education_level_gpt-4.1-mini_512_500_use-logprobs"

cachefiles = cachedir.glob("*.pkl")

out = {}
for file in cachefiles:
    print(file.stem)
    prefix, indices = file.stem.split("_")
    idx1, idx2 = [int(idx) for idx in indices.split("-")]
    print(idx1, idx2)
    with open(file, "rb") as f:
        obj = pickle.load(f)
    out[(idx1, idx2)] = obj
    tot_votes = np.array(obj["votes_a"][0]) + np.array(obj["votes_b"][0])
    print(tot_votes[0, 1])
    print(tot_votes[1, 0])
    print(obj["average"])