# %%
import multiprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from readability import Readability
from readability.exceptions import ReadabilityException
import nltk
from joblib import Parallel, delayed

from qurating.constants import (
    VALIDATION_DATA_RESULTS_DIR,
    VALIDATION_DATA_DATASETS_DIR,
    FIGURES_DIR,
)

nltk.download("punkt_tab")

ED_LEVEL_NUMERICAL = {
    "bottom_up_sample_english_markdown.parquet": {
        "Preschool": 0,
        "Lower primary": 2,
        "Upper primary": 4,
        "Lower secondary": 6,
        "Upper secondary": 8,
        "Tertiary": 10,
    },
    "cosmopedia-v2_sample_50000.parquet": {
        "young_children": 0,
        "children": 2,
        "middle_school_students": 5,
        "high_school_studnets": 8,
        "college_students": 11,
        "college_studnets": 11,
        "researchers": 13,
    },
}

ED_LEVEL_COLUMN = {
    "bottom_up_sample_english_markdown.parquet": "education_level_normalized",
    "cosmopedia-v2_sample_50000.parquet": "audience",
}

MODEL_TYPES = {
    "general_educational": "ours_v2",
    "FLN_student-facing": None,
    "FLN_teacher-facing": None,
}

# BASE_MODEL = "Qwen3-Reranker-4B-seq-cls"
BASE_MODEL = "gemma-3-4b-pt"


def map_education_level(level_string, dataset_name):
    level_multi = False
    if not isinstance(level_string, str):
        level_num = level_num_max = level_num_min = np.nan
    else:
        level_num = []
        for level_component in level_string.split(","):
            level_component = level_component.strip()
            level_num.append(
                ED_LEVEL_NUMERICAL[dataset_name].get(level_component, np.nan)
            )
        if len(level_num) == 1:
            level_num = level_num[0]
            level_num_max = level_num
            level_num_min = level_num
        elif len(level_num) == 0:
            level_num = level_num_max = level_num_min = np.nan
        else:
            level_num_max = float(np.nanmax(level_num))
            level_num_min = float(np.nanmin(level_num))
            level_multi = True
    return pd.Series(
        {
            "education_level_numerical": level_num,
            "education_level_numerical_min": level_num_min,
            "education_level_numerical_max": level_num_max,
            "education_level_numerical_multi": level_multi,
        }
    )


def calc_fk(text):
    r = Readability(text)
    try:
        fk = r.flesch_kincaid()
    except ReadabilityException:
        return float(np.nan)
    return fk.score


# %%
# dataset_name = "bottom_up_sample_english_markdown.parquet"
dataset_name = "cosmopedia-v2_sample_50000.parquet"

dataset_file = VALIDATION_DATA_DATASETS_DIR / dataset_name
results_files = VALIDATION_DATA_RESULTS_DIR.rglob(f"*{dataset_name}")

dataset_df = pd.read_parquet(dataset_file)

if "material_type_normalized" in dataset_df.columns:
    dataset_df["material_type_normalized"] = dataset_df[
        "material_type_normalized"
    ].replace("nan", pd.NA)

results_dfs = {}
for rf in results_files:
    long_name = rf.parent.name
    for mod_type, short_name in MODEL_TYPES.items():
        if short_name is None:
            short_name = mod_type
        if short_name in long_name and BASE_MODEL in long_name:
            if mod_type in results_dfs:
                raise ValueError(f"More than one model found for {mod_type}")
            results_dfs[mod_type] = pd.read_parquet(rf).set_index("index")
            break

# %%
"""
Rating distributions split by metadata education level
"""

for mod_type in MODEL_TYPES:

    result_df = results_dfs[mod_type]

    ed_level_meta = dataset_df[ED_LEVEL_COLUMN[dataset_name]].apply(
        map_education_level, dataset_name=dataset_name
    )
    not_multi_filt = ~np.array(ed_level_meta["education_level_numerical_multi"])
    not_nan_filt = ~np.array(ed_level_meta["education_level_numerical"].isna())
    x = np.array(ed_level_meta["education_level_numerical"])[
        not_multi_filt & not_nan_filt
    ].astype(float)

    level_vars = [col for col in result_df.columns if col.endswith("_average")]

    fig, axs = plt.subplots(nrows=1, ncols=len(level_vars), figsize=(18, 5))
    if not isinstance(axs, np.ndarray):
        axs = [axs]

    for i, model_var in enumerate(level_vars):
        ax: plt.Axes = axs[i]
        ed_level_model = result_df[model_var]
        y = np.array(ed_level_model)[not_multi_filt & not_nan_filt].astype(float)

        # sns.boxplot(x=x, y=y, hue=x, whis=[5, 95], width=0.6, ax=axs[i])
        sns.violinplot(x=y, y=x, hue=x, ax=ax, orient="h", legend=False)
        sns.stripplot(
            x=y, y=x, ax=ax, size=2, jitter=0.08, orient="h", color=[0, 0, 0, 0.1]
        )
        # sns.swarmplot(x=y, y=x, ax=axs[i], size=1, orient="h", color=[0, 0, 0, 1])
        ax.set_title(" ".join(model_var.split("_")[:-1]))
        ax.set_xlabel("Model Score")
        if i == 0:
            ax.set_ylabel("Education Level (Scrape metadata)")
            ax.set_yticklabels(ED_LEVEL_NUMERICAL[dataset_name])
        else:
            ax.set_yticks([], [])

    fig.savefig(
        FIGURES_DIR / f"bottom_up_dataset_comparison_education-level_{mod_type}.png",
        dpi=300,
        bbox_inches="tight",
    )

# %%
"""
Rating distributions split by metadata material type
"""

for mod_type in MODEL_TYPES:

    result_df = results_dfs[mod_type]

    mat_type_meta = dataset_df["material_type_normalized"]
    not_nan_filt = ~np.array(mat_type_meta.isna())
    x = np.array(mat_type_meta)[not_nan_filt]
    print(len(x))
    unqvals, unq_counts = np.unique(x, return_counts=True)
    unqvals_sample = unqvals[unq_counts >= 10]
    sample_filt = np.isin(x, unqvals_sample)
    x = x[sample_filt]
    print(len(x))

    level_vars = [col for col in result_df.columns if col.endswith("_average")]

    fig, axs = plt.subplots(nrows=1, ncols=len(level_vars), figsize=(18, 5))
    if not isinstance(axs, np.ndarray):
        axs = [axs]

    for i, model_var in enumerate(level_vars):
        ax: plt.Axes = axs[i]
        ed_level_model = result_df[model_var]
        y = np.array(ed_level_model)[not_nan_filt][sample_filt].astype(float)

        # sns.boxplot(x=x, y=y, hue=x, whis=[5, 95], width=0.6, ax=axs[i])
        sns.violinplot(x=y, y=x, hue=x, ax=ax, orient="h", legend=False)
        sns.stripplot(
            x=y, y=x, ax=ax, size=2, jitter=0.08, orient="h", color=[0, 0, 0, 0.1]
        )
        # sns.swarmplot(x=y, y=x, ax=axs[i], size=1, orient="h", color=[0, 0, 0, 1])
        ax.set_title(" ".join(model_var.split("_")[:-1]))
        ax.set_xlabel("Model Score")
        if i == 0:
            ax.set_ylabel("Material type (Scrape metadata)")
            # ax.set_yticklabels(np.unique(x))
        else:
            ax.set_yticks([], [])

    fig.savefig(
        FIGURES_DIR / f"bottom_up_dataset_comparison_material-type_{mod_type}.png",
        dpi=300,
        bbox_inches="tight",
    )

# %%
ncpus = multiprocessing.cpu_count()
print(ncpus)

n_jobs = min(6, max(ncpus - 2, 1))

p = Parallel(n_jobs=n_jobs, backend="loky", verbose=60, batch_size=16)
res = p(delayed(calc_fk)(ft) for ft in dataset_df["full_text"])

fk_scores = np.array(res)

# %%
fig, ax = plt.subplots()

y_fk = fk_scores[not_multi_filt & not_nan_filt]
sns.boxplot(x=y, y=y_fk, hue=y, whis=[5, 95], width=0.6, ax=ax)
ax.set_ybound([np.nanmin(y_fk), 100])

# %%
# lowest scoring tertiaries
tertiary_filt = mat_type_meta["education_level_numerical"] == 14

lowest_50_index = ed_level_model.loc[tertiary_filt].sort_values()[:50].index
lowest_50_df = dataset_df.loc[lowest_50_index]

for ft in lowest_50_df["full_text"]:
    print(f"\n{'#'*50}\n")
    print(ft)
    print(f"\n{'#'*50}\n")
