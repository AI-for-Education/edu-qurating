# %%
import re
from collections import defaultdict
import json

import numpy as np
import pandas as pd
from datasets import Dataset, load_from_disk

from qurating.constants import DATA_DIR

# %%
evals_dir = DATA_DIR / "education_evals"
train_dir = evals_dir / "train"
eval_dir = evals_dir / "eval"

# %%
dataset = Dataset.from_parquet(str(evals_dir / "flteach_grpo_dataset.parquet"))

# %%
seed = 957346
split_ds = dataset.train_test_split(test_size=0.1, seed=seed)

split_ds.save_to_disk(str(evals_dir / "flteach_grpo_dataset_train-test"))

# %%
ds = load_from_disk(str(evals_dir / "flteach_grpo_dataset_train-test"))