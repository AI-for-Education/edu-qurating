# %%
import re
from collections import defaultdict

import pandas as pd

from qurating.constants import DATA_DIR

timestamp_req = r"^(.*?) (\d{8}_\d{6}) Raw.xlsx$"
timestamp_rec = re.compile(timestamp_req)

# %%
evals_dir = DATA_DIR / "education_evals"

filelist = sorted(evals_dir.glob("* Raw.xlsx"))

timestamps = defaultdict(list)
df_dict = defaultdict(list)

for fl in filelist:
    match = timestamp_rec.match(fl.name)
    task_name, date_time = match.groups()
    timestamps[task_name].append(date_time)
    df = pd.read_excel(fl)
    df["task_name"] = task_name
    df["date_time"] = date_time
    df_dict[task_name].append(df)

full_df = pd.concat(
    [df for dflist in df_dict.values() for df in dflist], axis=0, ignore_index=True
)

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
