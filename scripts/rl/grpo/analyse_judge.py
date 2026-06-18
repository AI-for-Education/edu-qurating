# %%
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent

# %%
judge_csv = HERE / "judge_all_full_fln.csv"

judge_df = pd.read_csv(judge_csv)

with open(HERE / "grpo_configs.yaml") as f:
    cfg = yaml.safe_load(f)

# %%
unq_variant_labels = sorted(
    set([lab for ab in ["a", "b"] for lab in judge_df[f"variant_{ab}"].unique()])
)

cfg_map = {lab: cfg.get("/".join(lab.split("/")[:-1])) for lab in unq_variant_labels}
