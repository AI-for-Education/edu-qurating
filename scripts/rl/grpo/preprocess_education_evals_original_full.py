# %%
import re
from collections import defaultdict
import json

import numpy as np
import pandas as pd
from datasets import Dataset

from qurating.constants import DATA_DIR

timestamp_req = r"^(.*?) (\d{8}_\d{6}) Raw.xlsx$"
timestamp_rec = re.compile(timestamp_req)

# %%
evals_dir = DATA_DIR / "education_evals_original_full"
train_dir = evals_dir / "train"
eval_dir = evals_dir / "eval"
sheet_names = ["T1", "T2", "T3", "T4", "T5", "T6"]

full_file = evals_dir / "2026-06-08 Tasks 1-6 Tranches 1-2 SHARED.xlsx"

all_sheets = pd.read_excel(full_file, sheet_name=sheet_names)

full_df = pd.concat(list(all_sheets.values()), axis=0, ignore_index=True)

isnumgrade = full_df["Grade"].apply(lambda x: isinstance(x, int))
full_df = full_df.loc[isnumgrade]

filters = [
    "Subject == 'Literacy'",
    "`Subtask Subject Restrictions` != 'Numeracy'",
    "Grade <= 3",
]
filt_df = full_df.query(" & ".join(filters)).reset_index(drop=True)

# %%
filt_df.to_csv(
    evals_dir / "education_evals_combined_literacy_grade0-3.csv",
    index=False,
    encoding="utf_8_sig",
)

# %%
usecols = [
    "Task Name",
    "Subtask Name",
    "Grade",
    "Subject",
    "Topic",
    "Learning Objective",
    "Rendered Prompt",
    "Good Response",
    "Okay Response",
    "Bad Response",
]

for row in json.loads(filt_df[usecols].to_json(orient="records")):
    print(json.dumps(row, indent=2))
    print(f"#" * 50)

# %%
test_df = filt_df.copy()
seed = 957346
rng = np.random.default_rng(seed=seed)

test_ds = Dataset.from_pandas(test_df[usecols].reset_index())
test_ds.save_to_disk(
    evals_dir / "education_evals_combined_literacy_grade0-3_test.parquet"
)
