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
evals_dir = DATA_DIR / "education_evals_original"
train_dir = evals_dir / "train"
eval_dir = evals_dir / "eval"

filelist = sorted(evals_dir.rglob("* Raw.xlsx"))

timestamps = defaultdict(list)
df_dict = defaultdict(list)

for fl in filelist:
    match = timestamp_rec.match(fl.name)
    assert match is not None
    task_name, date_time = match.groups()
    timestamps[task_name].append(date_time)
    df = pd.read_excel(fl)
    df["task_name"] = task_name
    df["date_time"] = date_time
    if fl.parent == train_dir:
        df["split"] = "train"
    else:
        df["split"] = "test"
    df_dict[task_name].append(df)

full_df = pd.concat(
    [df for dflist in df_dict.values() for df in dflist], axis=0, ignore_index=True
)
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
    "split",
]

for row in json.loads(filt_df[usecols].to_json(orient="records")):
    print(json.dumps(row, indent=2))
    print(f"#" * 50)

# %%
train_df = filt_df.query("split == 'train'")
seed = 957346
rng = np.random.default_rng(seed=seed)

nrows = len(train_df)
sampidx = rng.choice(nrows, size=nrows * 10, replace=True)

train_ds = Dataset.from_pandas(train_df[usecols].iloc[sampidx].reset_index())
train_ds.save_to_disk(evals_dir / "education_evals_combined_literacy_grade0-3_train.parquet")

test_df = filt_df.query("split == 'test'")
test_ds = Dataset.from_pandas(test_df[usecols].reset_index())
test_ds.save_to_disk(evals_dir / "education_evals_combined_literacy_grade0-3_test.parquet")