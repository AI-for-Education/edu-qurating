# %%
import os
import subprocess
import time
from itertools import combinations
from pathlib import Path

import datashader as ds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import seaborn as sns
import statsmodels.api as sm
from datasets import Dataset, get_dataset_config_names, load_dataset
from dotenv import load_dotenv
from joblib import Parallel, delayed
from scipy.stats import gaussian_kde, pearsonr, rankdata
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from tqdm import tqdm

from qurating.constants import DATA_DIR
from qurating.scoring_projects.fwe_fortified.filtering import load_score_subset

load_dotenv(override=True)

# %%
configs = get_dataset_config_names("airtrain-ai/fineweb-edu-fortified")

model_name = "AI-for-Education/qurater_gemma-3-4b-pt_ds-ours_v2-200000"
model_string = [
    substr for substr in Path(model_name).parts if substr.startswith("qurater_")
]
assert len(model_string) == 1
model_string = model_string[0]

# %%
AZURE_CLIENT_KWARGS = {
    "account_url": "https://quratingscoressa.blob.core.windows.net",
    "credential": os.getenv("QURATING_SCORES_AZURE_STORAGE_KEY"),
}


# ds = load_batch(configs[0], 0)

# %%
### load scores
pool = pa.default_memory_pool()
print(f"Allocated: {pool.bytes_allocated()}, Available: {pool.max_memory()}")
n_jobs = 20
parallel_backend = "loky"
with Parallel(n_jobs=n_jobs, verbose=60, backend=parallel_backend) as p:
    df_list = p(
        delayed(load_score_subset)(subset, model_string, AZURE_CLIENT_KWARGS)
        for subset in configs
    )
print(f"Allocated: {pool.bytes_allocated()}, Available: {pool.max_memory()}")
subprocess.call(["rm -fr ~/.cache/huggingface"], shell=True)

scores_df = pd.concat(df_list, axis=0, ignore_index=True)
scores_arr = np.array(scores_df.iloc[:, 1:])

labels = ["_".join(lab.split("_")[:-1]) for lab in scores_df.columns[1:]]
ids_arr = scores_df["id"].to_list()

del scores_df
del df_list

# %%
# get percentiles
pct_edges = [50, 75, 90, 95]
pct = np.percentile(scores_arr, pct_edges, axis=0)

print(pct)

del pct
# %%
## load fineweb scores
fwe_full = {}
for subset in configs:
    fwe_full[subset] = load_dataset(
        "airtrain-ai/fineweb-edu-fortified",
        name=subset,
        split="train",
        streaming=True,
        token=True,
    )


# %%
outdir = DATA_DIR / "fwe_cache"
outdir.mkdir(exist_ok=True, parents=True)


def load_fwe_score_subset(subset, streaming_ds, shardi):
    fname = outdir / f"{subset}-shard_{shardi:03d}"
    if fname.exists():
        return
    scores_list = []
    shard = streaming_ds.shard(streaming_ds.n_shards, shardi, contiguous=True)
    for row in shard:
        scores_list.append(
            {
                "subset": subset,
                "shard": shardi,
                **{key: val for key, val in row.items() if key in ["id", "score"]},
            }
        )
    shard_ds = Dataset.from_list(scores_list)
    shard_ds.save_to_disk(fname)


job_list_flat = [
    (subset, fwe_full[subset], batchi)
    for subset in configs
    for batchi in range(fwe_full[subset].n_shards)
]

print(len(job_list_flat))

n_jobs = 12
parallel_backend = "loky"
with Parallel(n_jobs=n_jobs, verbose=60, backend=parallel_backend) as p:
    p(
        delayed(load_fwe_score_subset)(subset, streaming_ds, shardi)
        for subset, streaming_ds, shardi in job_list_flat
    )


# %%
fwe_scores_arr = []
nexti = 0
for subset, _, shardi in tqdm(job_list_flat):
    fname = outdir / f"{subset}-shard_{shardi:03d}"
    dataset = Dataset.load_from_disk(fname)
    scores_shard = list(dataset["score"])
    ids_shard = list(dataset["id"])
    assert ids_shard == ids_arr[nexti : nexti + len(ids_shard)]
    nexti += len(ids_shard)
    fwe_scores_arr.extend(scores_shard)
    # for row in ds:
    #     assert row["id"] == ids_arr[nexti]
    #     nexti += 1
    #     fwe_scores_arr.append(row["score"])

# %%
fwe_scores_vec = np.array(fwe_scores_arr)

del fwe_scores_arr

# %%
r_vec = np.zeros(scores_arr.shape[1])
p_vec = np.zeros(scores_arr.shape[1])

for i in tqdm(range(scores_arr.shape[1])):
    res = pearsonr(fwe_scores_vec, scores_arr[:, i])
    r_vec[i] = res.statistic
    p_vec[i] = res.pvalue

# %%
print()
print("Correlation (pearson r) between FWE Score and Core Ed Edu-Qurater Dimensions")
print("-" * 50)
pd.Series(index=["_".join(c.split("_")[:-1]) for c in labels], data=r_vec)

# %%
seed = 402645542
scores_df = pd.DataFrame(
    np.hstack([fwe_scores_vec[:, None], scores_arr]), columns=["fwe_score", *labels]
)
samp_scores_df = scores_df.sample(n=int(1e6), replace=True, random_state=seed, axis=0)

# %%
### marginal distributions
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 5))

for i, lab in enumerate(labels):
    x = labels
    y = np.array(samp_scores_df.iloc[:, 1:]).astype(float)

    # sns.boxplot(x=x, y=y, hue=x, whis=[5, 95], width=0.6, ax=axs[i])
    sns.violinplot(x=y, y=x, hue=x, ax=ax, orient="h", legend=False)
    sns.stripplot(
        x=y, y=x, ax=ax, size=2, jitter=0.08, orient="h", color=[0, 0, 0, 0.1]
    )
    # sns.swarmplot(x=y, y=x, ax=axs[i], size=1, orient="h", color=[0, 0, 0, 1])
    # ax.set_title(" ".join(model_var.split("_")[:-1]))
    ax.set_xlabel("Model Score")
    if i == 0:
        ax.set_ylabel("Material type (Scrape metadata)")
        # ax.set_yticklabels(np.unique(x))
    else:
        ax.set_yticks([], [])

# %%
figs = []
axs = []

for labi in range(1, 7):
    lab = labels[labi-1]
    print(labi)

    values = np.array(samp_scores_df.iloc[:, [0, labi]]).T
    u = rankdata(values[0]) / (values.shape[1] + 1)
    v = rankdata(values[1]) / (values.shape[1] + 1)
    values = np.vstack([u, v])

    xmin = values[0].min()
    xmax = values[0].max()
    ymin = values[1].min()
    ymax = values[1].max()

    kernel = gaussian_kde(values)

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

    positions = np.vstack([X.ravel(), Y.ravel()])

    zvals = kernel(positions).T

    Z = np.reshape(zvals, X.shape)


    fig, ax = plt.subplots()
    im = ax.contourf(X, Y, Z, levels=10, cmap="viridis")
    fig.colorbar(im, label="Empirical copula density")
    # ax.imshow(np.rot90(Z), cmap=plt.cm.viridis, extent=[xmin, xmax, ymin, ymax])
    # ax.plot(values[0], values[1], 'k.', markersize=0.005)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_xlabel("u (rank of FWE_Score)")
    ax.set_ylabel(f"v (rank of {lab})")
    ax.set_title("Empirical Copula Density")
    # ax.axis("auto")

    figs.append(fig)
    axs.append(ax)

    fig.show()
    time.sleep(0.1)

# %%
cvs = ds.Canvas(plot_width=850, plot_height=500)

scores_df = pd.DataFrame(scores_arr, columns=labels)

violin_density = scores_df.hvplot.scatter(
    group_label="Dimension",
    datashade=True,
    dynspread=True,
    cmap="viridis",
    cnorm="eq_hist",  # Equalises histogram to highlight low-density areas
    height=500,
    width=700,
    title="Large-Scale Density Distribution via Datashader",
)


# %%
### CV r-squared (overall)
seed = 734635346
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
oof_predictions = np.zeros(len(fwe_scores_vec))

X = sm.add_constant(scores_arr)
for train_idx, val_idx in tqdm(kf.split(X), total=n_folds):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = fwe_scores_vec[train_idx], fwe_scores_vec[val_idx]

    model = sm.OLS(y_train, X_train).fit()
    oof_predictions[val_idx] = model.predict(X_val)

cv_r2 = r2_score(fwe_scores_vec, oof_predictions)

del X, model, X_train, X_val, y_train, y_val

# %%
### CV r-squared (leave-out-one-column)
ndims = scores_arr.shape[1]

cv_r2_leave_one = np.zeros(ndims)

for leavei in range(ndims):
    print(f"Leave out dim: {leavei}")
    use_cols_logical = np.ones(shape=ndims, dtype=bool)
    use_cols_logical[leavei] = False

    oof_predictions_leave_one = np.zeros(len(fwe_scores_vec))
    X = sm.add_constant(scores_arr[:, use_cols_logical])

    for train_idx, val_idx in tqdm(kf.split(X), total=n_folds):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = fwe_scores_vec[train_idx], fwe_scores_vec[val_idx]

        model = sm.OLS(y_train, X_train).fit()
        oof_predictions_leave_one[val_idx] = model.predict(X_val)

    del X, model, X_train, X_val, y_train, y_val

    cv_r2_leave_one[leavei] = r2_score(fwe_scores_vec, oof_predictions_leave_one)

# %%
print()
print(f"CV R^2 (Full model): {cv_r2 :.04f}")
print("-" * 50)
for i in range(ndims):
    print(f"CV R^2 (without {labels[i]}): {cv_r2_leave_one[i] :.04f}")

# %%
### CV r-squared (leave-in-one-column)
ndims = scores_arr.shape[1]

cv_r2_leave_one_in = np.zeros(ndims)

for leavei in range(ndims):
    print(f"Leave in dim: {leavei}")

    oof_predictions_leave_one_in = np.zeros(len(fwe_scores_vec))
    X = sm.add_constant(scores_arr[:, [leavei]])

    for train_idx, val_idx in tqdm(kf.split(X), total=n_folds):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = fwe_scores_vec[train_idx], fwe_scores_vec[val_idx]

        model = sm.OLS(y_train, X_train).fit()
        oof_predictions_leave_one_in[val_idx] = model.predict(X_val)

    del X, model, X_train, X_val, y_train, y_val

    cv_r2_leave_one_in[leavei] = r2_score(fwe_scores_vec, oof_predictions_leave_one_in)

# %%
print()
print(f"CV R^2 (Full model): {cv_r2 :.04f}")
print("-" * 50)
for i in range(ndims):
    print(f"CV R^2 (with only {labels[i]}): {cv_r2_leave_one_in[i] :.04f}")

# %%
### CV r-squared (leave-in-two-columns)
ndims = scores_arr.shape[1]

cv_r2_leave_two_in = np.zeros((ndims, ndims))

for leavei, leavej in combinations(range(ndims), 2):
    print(f"Leave in dim: {leavei}")

    oof_predictions_leave_two_in = np.zeros(len(fwe_scores_vec))
    X = sm.add_constant(scores_arr[:, [leavei, leavej]])

    for train_idx, val_idx in tqdm(kf.split(X), total=n_folds):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = fwe_scores_vec[train_idx], fwe_scores_vec[val_idx]

        model = sm.OLS(y_train, X_train).fit()
        oof_predictions_leave_two_in[val_idx] = model.predict(X_val)

    del X, model, X_train, X_val, y_train, y_val

    cv_r2_leave_two_in[leavei, leavej] = r2_score(
        fwe_scores_vec, oof_predictions_leave_two_in
    )

# %%
print()
print(f"CV R^2 (Full model): {cv_r2 :.04f}")
print("-" * 50)
for i, j in combinations(range(ndims), 2):
    print(
        f"CV R^2 (with only {labels[i]}, {labels[j]}): {cv_r2_leave_two_in[i, j] :.04f}"
    )
