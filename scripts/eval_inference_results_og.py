# %%
import json
from itertools import product

from datasets import Dataset
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from qurating.constants import DATA_DIR, RESULTS_DIR, FIGURES_DIR

n_samples = 500000
seed = 72353534

NUM_EXAMPLES = 100000
OFFSET = 400000
TOKENS_MAX = 512
JUDGEMENTS_MODEL = "gpt-4.1-mini"
MODELS = [
    "qurater_Sheared-LLaMA-1.3b_bsz512_lr5e-5_epochs2_warmup0.1_conf0.5_labeltemp1.0_ds-ours_v2-500000-200000-512-72353534-gpt-4.1-mini-logprobs",
    "qurater_Qwen3-Reranker-0.6B-seq-cls_bsz512_lr5e-5_epochs2_warmup0.1_conf0.5_labeltemp1.0_ds-ours_v2-500000-200000-512-72353534-gpt-4.1-mini-logprobs",
    "qurater_Qwen3-Reranker-4B-seq-cls_bsz512_lr5e-5_epochs2_warmup0.1_conf0.5_labeltemp1.0_ds-ours_v2-500000-200000-512-72353534-gpt-4.1-mini-logprobs",
    "qurater_Qwen3-Reranker-8B-seq-cls_bsz512_lr5e-5_epochs2_warmup0.1_conf0.5_labeltemp1.0_ds-ours_v2-500000-200000-512-72353534-gpt-4.1-mini-logprobs",
    "qurater_gemma-3-4b-pt_bsz512_lr5e-5_epochs2_warmup0.1_conf0.5_labeltemp1.0_ds-ours_v2-500000-200000-512-72353534-gpt-4.1-mini-logprobs",
]

USE_LOGPROBS = True

use_logprobs_suffix = "_use-logprobs" if USE_LOGPROBS else ""
offset_suffix = f"_offset-{OFFSET}" if OFFSET != 0 else ""

annotations_dir = DATA_DIR / "annotations"

results_base = f"fwe-fortified_sampled-{n_samples}_seed-{seed}"
template_base = "ours_v2"

results_file = (
    RESULTS_DIR
    / f"tokens_max_{TOKENS_MAX}"
    / results_base
    / template_base
    / f"combined_{JUDGEMENTS_MODEL}_nexamples-{NUM_EXAMPLES}"
    f"{offset_suffix}{use_logprobs_suffix}"
)

# %%
pfile = results_file.with_suffix(".parquet")
if pfile.exists():
    dataset = Dataset.from_parquet(str(pfile), keep_in_memory=True)
else:
    dataset = Dataset.load_from_disk(results_file)


# %%
def logit_pairs_to_probs(logitsa, logitsb):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    logit_diffs = logitsa - logitsb
    probs = sigmoid(logit_diffs)
    out = np.zeros((probs.shape[0], 2))
    out[:, 0] = 1 - probs
    out[:, 1] = probs
    return out


labels = [
    "education_level",
    "education_level_primary",
    "education_level_secondary",
    "factual_accuracy",
    "lesson_engagement",
    "pedagogical_structure",
]

mae_dict = {}
acc_dict = {}

for model in MODELS:
    annotations_file = (
        annotations_dir
        / model
        / f"tokens_max_{TOKENS_MAX}"
        / results_base
        / template_base
        / f"combined_{JUDGEMENTS_MODEL}_nexamples-{NUM_EXAMPLES}"
        f"{offset_suffix}{use_logprobs_suffix}.json"
    )

    with open(annotations_file) as f:
        annotations = json.load(f)

    annotations_df = pd.DataFrame(annotations)
    n_annotations = len(annotations)

    annot_arrs = {}
    annot_probs_arrs = {}
    probs_arrs = {}
    for label in labels:
        annot_arr = np.zeros((n_annotations // 2, 2))
        annot_arr[
            annotations_df["original_pair_id"], annotations_df["text_position"]
        ] = annotations_df[f"{label}_average"]
        annot_arrs[label] = annot_arr
        annot_probs_arrs[label] = logit_pairs_to_probs(*annot_arr.T)

        probs = np.array(dataset[f"{label}_average"])
        probs_arr = probs[
            annotations_df["original_pair_id"],
            annotations_df["text_position"],
            1 - annotations_df["text_position"],
        ].reshape((-1, 2))
        probs_arrs[label] = probs_arr

    mae = np.zeros((len(labels), len(labels)))
    acc = np.zeros((len(labels), len(labels)))
    for (i, label1), (j, label2) in product(enumerate(labels), repeat=2):
        # print(label1, label2)
        mae[i, j] = (
            np.abs(annot_probs_arrs[label1][:, 0] - probs_arrs[label2][:, 0])
        ).mean()
        acc[i, j] = (
            annot_probs_arrs[label1][:, 0].round() == probs_arrs[label2][:, 0].round()
        ).mean()

    print("MAE")
    print(mae.round(3))
    print()
    print("Choice Accuracy")
    print(acc.round(3))

    mae_dict[model] = mae
    acc_dict[model] = acc

# %%
acc_df = (
    pd.DataFrame(
        {key.split("_")[1]: arr.diagonal() for key, arr in acc_dict.items()},
        index=labels,
    )
    .reset_index()
    .melt(id_vars="index", var_name="model", value_name="accuracy")
    .rename(columns={"index": "Education Dimension"})
)

ratio = [2.5, 1]
fig, (ax_top, ax_bottom) = plt.subplots(
    ncols=1,
    nrows=2,
    sharex=False,
    gridspec_kw={"hspace": 0.1},
    figsize=(10, 5),
    height_ratios=ratio,
)

sns.barplot(acc_df, x="model", y="accuracy", hue="Education Dimension", ax=ax_top)
sns.barplot(acc_df, x="model", y="accuracy", hue="Education Dimension", ax=ax_bottom)
ax_bottom.set_xticklabels(ax_bottom.get_xticklabels(), rotation=45)
for tl in ax_bottom.get_xticklabels():
    tl.set_horizontalalignment("right")
ax_top.set_xticks([])
ax_top.set_xlabel("")
ax_top.yaxis.set_label_coords(-0.05, 0.25)
ax_top.set_ylim(bottom=0.7, top=acc_df["accuracy"].max() * 1.02)
ax_top.grid()
ax_bottom.set_ylim(bottom=0, top=0.1)
ax_bottom.set_yticks([0, 0.05, 0.1])
ax_bottom.set_ylabel("")
ax_bottom.grid()

sns.despine(ax=ax_bottom)
sns.despine(ax=ax_top, bottom=True)

ax = ax_top
d = 0.015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color="k", clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal

ax2 = ax_bottom
d2 = d * (ratio[0] / ratio[1])
kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d2, 1 + d2), **kwargs)  # bottom-left diagonal

# remove one of the legend
ax_bottom.legend_.remove()
sns.move_legend(ax_top, "lower right")

fig.suptitle("Accuracy of pairwise predictions on 100k held out samples")

fig.savefig(FIGURES_DIR / "qurater_model_comparison.png", dpi=300, bbox_inches="tight")
