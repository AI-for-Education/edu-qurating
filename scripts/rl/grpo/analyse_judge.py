# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from qurating.constants import DATA_DIR, FIGURES_DIR

HERE = Path(__file__).resolve().parent
DATA_DIR_GRPO_EVALS = DATA_DIR / "grpo_evals"

# %%
judge_csv = DATA_DIR_GRPO_EVALS / "judge_all_full_fln.csv"
judge_df = pd.read_csv(judge_csv)

with open(HERE / "grpo_configs.yaml") as f:
    cfg = yaml.safe_load(f)

prompt_dimensions_csv = DATA_DIR_GRPO_EVALS / "prompt_dimensions.csv"
prompt_dimensions_df = pd.read_csv(prompt_dimensions_csv)

# %%
unq_variant_labels = sorted(
    {lab for ab in ["a", "b"] for lab in judge_df[f"variant_{ab}"].unique()}
)

cfg_map = {
    lab: cfg.get("/".join(lab.split("/")[:-1]), {}) for lab in unq_variant_labels
}

variable_maps = {}

variable_maps["base_model"] = {
    key: val.get("base_model") for key, val in cfg_map.items()
}
variable_maps["base_model"]["base_model"] = "unsloth/Qwen3-4B-Base"

variable_maps["reward"] = {key: val.get("reward", {}) for key, val in cfg_map.items()}

# NOTE: for reward_descriptor, the label refers to which things were included in the reward function:
# core-ed: core-educational model was included
# <fl-dimension> (e.g. writing-encoding): this dimension of fl-teacher model was included
# fl_teacher: all dimensions of fl-teacher were included together
# correctness_reward: correctness reward was included (LLM-as-a-judge vs good response as reference)
# length_target: length matching reward was included (vs good response reference)
variable_maps["reward_descriptor"] = {}
for key, val in cfg_map.items():
    qr_reward_list = val.get("reward", {}).get("qurating", [])
    qrr_descr = "-".join(sorted(qrr["name"] for qrr in qr_reward_list)).strip("-")
    efn_list = sorted(val.get("reward", {}).get("extra_functions", []))
    extra_descr = "-".join(efn_list).strip("-")
    variable_maps["reward_descriptor"][key] = f"{qrr_descr}-{extra_descr}".strip("-")

variable_maps["reward_descriptor_fl"] = {}
for key, val in cfg_map.items():
    qr_reward_list = val.get("reward", {}).get("qurating", [])
    qrr_descr = "-".join(
        sorted(qrr["name"] for qrr in qr_reward_list if qrr["name"] not in ["core_ed"])
    )
    variable_maps["reward_descriptor_fl"][key] = qrr_descr

variable_maps["checkpoint"] = {}
for key in cfg_map:
    ckp_str = key.split("/")[-1].split("-")[-1]
    try:
        variable_maps["checkpoint"][key] = int(ckp_str)
    except:  # noqa: E722
        variable_maps["checkpoint"][key] = None

# %%
## add variable columns for each variant according to variable maps
variant_list = ["a", "b"]

for variant in variant_list:
    for variable_name, variable_map in variable_maps.items():
        judge_df[f"variant_{variant}_{variable_name}"] = (
            pd.Series(variable_map).loc[judge_df[f"variant_{variant}"]].to_list()
        )

## merge in prompt dimensions info
judge_df = judge_df.merge(prompt_dimensions_df, on="prompt_index", how="left")


# %%
## win over base
def win_rate(df, comparator, comparison):
    res = (
        df.query(f"variant_a == '{comparator}'")[f"{comparison}_winner"] == "variant_b"
    ).mean()
    return res


####################################
print()
print("#" * 50)
print("Overall win rate vs BASE:")
print()

## overall - quality
print("Quality:")
print("*" * 50)
print(win_rate(judge_df, comparator="base_model", comparison="quality"))

## overall - follow
print()
print("Instruction following:")
print("*" * 50)
print(win_rate(judge_df, comparator="base_model", comparison="follow"))


#####################################
print()
print("#" * 50)
print("Per REWARD TYPE and CHECKPOINT win rate vs BASE:")
print()

## by reward descriptor and checkpoint - quality
print("Quality:")
print("*" * 50)
print(
    judge_df.groupby(
        ["variant_b", "variant_b_reward_descriptor", "variant_b_checkpoint"]
    )
    .apply(win_rate, comparator="base_model", comparison="quality")
    .sort_values(ascending=False)
)

## by reward descriptor and checkpoint - follow
print()
print("Instruction following:")
print("*" * 50)
print(
    judge_df.groupby(
        ["variant_b", "variant_b_reward_descriptor", "variant_b_checkpoint"]
    )
    .apply(win_rate, comparator="base_model", comparison="follow")
    .sort_values(ascending=False)
)

#####################################
print()
print("#" * 50)
print("Per REWARD TYPE and CHECKPOINT win rate vs GOOD:")
print()

## by reward descriptor and checkpoint - quality
print("Quality:")
print("*" * 50)
print(
    judge_df.groupby(
        ["variant_b", "variant_b_reward_descriptor", "variant_b_checkpoint"]
    )
    .apply(win_rate, comparator="Good Response", comparison="quality")
    .sort_values(ascending=False)
)

## by reward descriptor and checkpoint - follow
print()
print("Instruction following:")
print("*" * 50)
print(
    judge_df.groupby(
        ["variant_b", "variant_b_reward_descriptor", "variant_b_checkpoint"]
    )
    .apply(win_rate, comparator="Good Response", comparison="follow")
    .sort_values(ascending=False)
)

# %%
plot_variant_b = {
    "Answer-structure reward": "instruction_following/test12/checkpoint-1000",
    "Edu-Qurating reward\n(Core-Ed & FLT)": "test2/checkpoint-2000",
    "Combined reward\nEdu-Qurating (Core-Ed & FLT)\n and Answer-structure": "instruction_following/test9/checkpoint-2000",
    "Combined reward\nEdu-Qurating (Core-Ed)\nand Answer-structure": "instruction_following/test10/checkpoint-1000",
}

comparison_info = {
    "quality": "Pedagogical quality",
    "follow": "Instruction following",
}

plot_df_list = []
jquery_a = "variant_a == 'base_model'"
jquery_b = " | ".join(f"variant_b == '{val}'" for val in plot_variant_b.values())
jquery = f"{jquery_a} & ({jquery_b})"
for comparison, comparison_label in comparison_info.items():
    plot_df = judge_df.query(jquery)
    plot_df["Win-Rate (%)"] = (plot_df[f"{comparison}_winner"] == "variant_b") * 100
    plot_df["Comparison"] = comparison_label
    plot_df_list.append(plot_df)

plot_df = pd.concat(plot_df_list, axis=0, ignore_index=True)

fig, ax = plt.subplots(figsize=(8, 6))

with sns.color_palette("Set2"):
    sns.barplot(
        plot_df,
        y="variant_b",
        x="Win-Rate (%)",
        hue="Comparison",
        ax=ax,
        order=list(plot_variant_b.values()),
        errorbar="se",
        capsize=0.1,
        orient="h",
    )

ax.set_xlim(0, 100)

ax.set_yticklabels(list(plot_variant_b), rotation=0, ha="right", rotation_mode="anchor", )
ax.set_ylabel("Reward function", fontweight="bold")
ax.set_xlabel("Win-rate (%)", fontweight="bold")
# ax.tick_params("x", rotation=45)

ax.axvline(50, color="black", linestyle="--")
ax.annotate("Chance level", xy=[35, -0.29])

for container in ax.containers:
    ax.bar_label(container, fmt="{:.2f}%", padding=25)

fig.savefig(FIGURES_DIR / "grpo_main_comparison_v2.png", dpi=300, bbox_inches="tight")

# %%
plot_variant_b = {
    "Oral language / vocabulary": "test10/checkpoint-2000",
    "Phonological awareness": "test5/checkpoint-1000",
    "Systematic phonics": "test6/checkpoint-2000",
    "Reading fluency": "test7/checkpoint-1000",
    "Reading comprehension": "test8/checkpoint-1000",
    "Writing / encoding": "test9/checkpoint-1000",
    "Qurating combined": "test2/checkpoint-2000",
}

comparison_info = {
    "quality": "Pedagogical quality",
    "follow": "Instruction following",
}

plot_df_list = []
jquery_a = "variant_a == 'base_model'"
jquery_b = " | ".join(f"variant_b == '{val}'" for val in plot_variant_b.values())
jquery = f"{jquery_a} & ({jquery_b})"
for comparison, comparison_label in comparison_info.items():
    plot_df = judge_df.query(jquery)
    plot_df["Win-Rate (%)"] = (plot_df[f"{comparison}_winner"] == "variant_b") * 100
    plot_df["Comparison"] = comparison_label
    plot_df_list.append(plot_df)

plot_df = pd.concat(plot_df_list, axis=0, ignore_index=True)

fig, ax = plt.subplots(figsize=(10, 7))

with sns.color_palette("husl", 2):
    im = sns.barplot(
        plot_df,
        x="variant_b",
        y="Win-Rate (%)",
        hue="Comparison",
        ax=ax,
        order=list(plot_variant_b.values()),
        errorbar="se",
    )
ax.set_xticklabels(list(plot_variant_b), rotation=45, ha="right", rotation_mode="anchor")
ax.set_xlabel("Reward function")
# ax.tick_params("x", rotation=45)

ax.axhline(50, color="black", linestyle="--")
ax.annotate("Chance level", xy=[-0.48, 51])

fig.savefig(FIGURES_DIR / "grpo_FL_comparison.png", dpi=300, bbox_inches="tight")

# %%
mygb = judge_df.groupby(
    ["variant_a_reward_descriptor_fl", "variant_b_reward_descriptor_fl"]
)
myiter = iter(mygb)

# %%
comparison = "quality"
subdf = next(myiter)

tmp = mygb.apply(lambda x: (x["quality_winner"] == "variant_b").mean())

res_list = []
for (
    a_reward_fl,
    b_reward_fl,
    a_reward,
    b_reward,
    a_checkpoint,
    b_checkpoint,
), subdf in judge_df.groupby(
    [
        "variant_a_reward_descriptor_fl",
        "variant_b_reward_descriptor_fl",
        "variant_a_reward_descriptor",
        "variant_b_reward_descriptor",
        "variant_a_checkpoint",
        "variant_b_checkpoint",
    ]
):
    # if any(reward_descr_fl in ["", "fl_teacher"] for reward_descr_fl in (a_reward_fl, b_reward_fl)):
    #     continue
    if any(checkpoint != 2000 for checkpoint in (a_checkpoint, b_checkpoint)):
        continue
    if any("length_target" in reward_descr for reward_descr in (a_reward, b_reward)):  # type: ignore
        continue
    if any("core_ed" not in reward_descr for reward_descr in (a_reward, b_reward)):  # type: ignore
        continue
    if a_reward == b_reward:
        continue
    # print(a_checkpoint, b_checkpoint)
    for variant, reward_descr in zip(
        ("variant_a", "variant_b"), (a_reward_fl, b_reward_fl)
    ):
        if reward_descr in subdf.columns:
            filt = np.array(subdf[reward_descr])
            win_logical = np.array(subdf[f"{comparison}_winner"] == variant)
            win_n = float((win_logical & filt).sum())
            filt_n = float(filt.sum())
            res_list.append(
                {
                    "target_variant": variant,
                    "target_reward": (
                        a_reward_fl if variant == "variant_a" else b_reward_fl
                    ),
                    "comparator_reward": (
                        a_reward_fl if variant == "variant_b" else b_reward_fl
                    ),
                    "win_n": win_n,
                    "filt_n": filt_n,
                    "unfilt_n": len(subdf),
                }
            )
res_df = pd.DataFrame(res_list)

res_final = res_df.groupby(["target_reward", "comparator_reward"]).apply(
    lambda x: x["win_n"].sum() / x["filt_n"].sum()
)

res_final
# %%
for _, subdf in res_df.groupby(["target_reward", "comparator_reward"]):
    print(_)
    print(subdf)
