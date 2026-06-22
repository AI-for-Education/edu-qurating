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
    qrr_descr = "-".join(sorted(qrr["name"] for qrr in qr_reward_list))
    efn_list = sorted(val.get("reward", {}).get("extra_functions", []))
    extra_descr = "-".join(efn_list)
    variable_maps["reward_descriptor"][key] = "-".join([qrr_descr, extra_descr])

variable_maps["checkpoint"] = {}
for key in cfg_map:
    ckp_str = key.split("/")[-1].split("-")[-1]
    try:
        variable_maps["checkpoint"][key] = int(ckp_str)
    except:
        variable_maps["checkpoint"][key] = None

# %%
variant_list = ["a", "b"]

for variant in variant_list:
    for variable_name, variable_map in variable_maps.items():
        judge_df[f"variant_{variant}_{variable_name}"] = (
            pd.Series(variable_map).loc[judge_df[f"variant_{variant}"]].to_list()
        )


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
    judge_df.groupby(["variant_b_reward_descriptor", "variant_b_checkpoint"])
    .apply(win_rate, comparator="base_model", comparison="quality")
    .sort_values(ascending=False)
)

## by reward descriptor and checkpoint - follow
print()
print("Instruction following:")
print("*" * 50)
print(
    judge_df.groupby(["variant_b_reward_descriptor", "variant_b_checkpoint"])
    .apply(win_rate, comparator="base_model", comparison="follow")
    .sort_values(ascending=False)
)
