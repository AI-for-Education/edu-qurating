# %%
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from qurating.constants import DATA_DIR, FIGURES_DIR

# %%
results_dir = DATA_DIR / "pretraining_eval_results" / "results"
run_names = [
    "base-run-30000",
    "qr-filtered-run-30000",
    "qr-filtered-corrected-run-30000",
    "qr-filtered-dclm-run-30000",
]
step_size = 1000

res_df_full_list = []
for run_name in run_names:
    results_files = results_dir.rglob("*.json")

    files = sorted(
        (rf for rf in results_files if run_name in rf.parent.name), key=lambda x: x.name
    )

    res_list_dict = defaultdict(list)
    for step_file in files:
        # step_file_dir = results_dir / f"{step}"
        # step_file = sorted(step_file_dir.glob("*.json"))
        # assert len(step_file) >= 1
        # step_file = step_file[-1]
        with open(step_file, "r") as f:
            step_res = json.load(f)
        for eval, res in step_res["results"].items():
            if (
                eval.startswith(("custom|mmlu_cf:",)) and "_average" not in eval
            ) or eval == "all":
                continue
            res_list_dict[eval].append(res)

    res_df_list = []
    for eval, res_list in res_list_dict.items():
        df = pd.DataFrame(
            res_list,
            index=list(range(step_size, (len(files) + 1) * step_size, step_size)),
        )
        if "acc_norm" in df.columns:
            df["acc"] = df["acc_norm"]
            df["acc_stderr"] = df["acc_norm_stderr"]
        df.columns = pd.MultiIndex.from_product([[eval], df.columns])
        res_df_list.append(df)
    res_df = (
        pd.concat(res_df_list, axis=1).reset_index().rename(columns={"index": "step"})
    )
    res_df["run_name"] = run_name
    res_df_full_list.append(res_df)

res_df_full = pd.concat(res_df_full_list, axis=0, ignore_index=True)

# %%
figdir = FIGURES_DIR / "pretraining_eval_comparison_results"
figdir.mkdir(exist_ok=True, parents=True)

eval_names = [
    en for en in res_df_full.columns.levels[0] if en not in ("step", "run_name")
]
for eval_name in eval_names:
    plot_df = res_df_full[["step", eval_name, "run_name"]].droplevel(0, axis=1)
    plot_df.columns = ["step", *plot_df.columns[1:-1], "run_name"]

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.lineplot(plot_df, x="step", y="acc", hue="run_name", ax=ax)
    ax.grid(visible=True, which="both")
    fig.suptitle(eval_name)
    fig.savefig(
        figdir / f"{eval_name.replace(':', '-').replace('|', '_')}.png", dpi=300
    )

# %%
agg_df = pd.concat(
    [
        res_df_full["run_name"],
        res_df_full["step"],
        res_df_full.loc[:, (slice(None), "acc")].mean(axis=1).rename("acc"),
    ],
    axis=1,
)

fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(agg_df, x="step", y="acc", hue="run_name", ax=ax)
ax.grid(visible=True, which="both")

fig.savefig(figdir / "average_all.png", dpi=300)