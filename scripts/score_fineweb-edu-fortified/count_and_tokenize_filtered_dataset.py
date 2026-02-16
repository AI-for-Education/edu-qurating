# %%
from pathlib import Path
import os
from multiprocessing import cpu_count
from io import BytesIO

from joblib import Parallel, delayed
from dotenv import load_dotenv
from cloudpathlib import CloudPath, AzureBlobClient
from datasets import get_dataset_config_names

from qurating.inference import TokenizeAndChunk
from qurating.scoring_projects.fwe_fortified.filtering import (
    gen_filtered_batches_info,
    gen_filtered_batches,
)

load_dotenv(override=True)

COUNT_SIZE = False

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


# %%
def count_subset(subset, model_string, azure_client_kwargs):
    sizes = [
        info.st_size * 1e-9
        for info in gen_filtered_batches_info(subset, model_string, azure_client_kwargs)
    ]
    return subset, sizes


if COUNT_SIZE:

    p = Parallel(n_jobs=30, verbose=60)

    out = p(
        delayed(count_subset)(subset, model_string, AZURE_CLIENT_KWARGS)
        for subset in configs
    )

    nfiles = {subset: len(sizes) for subset, sizes in out}
    subset_sizes = {subset: sum(sizes) for subset, sizes in out}

    print(f"Num files: {sum(nfiles.values())}")
    print(f"Total size: {sum(subset_sizes.values()) :.03f} GB")

# %%
tokenizer_model = "HuggingFaceTB/SmolLM3-3B"
tokenizer = TokenizeAndChunk(tokenizer_model, "text", 512)


def tokenize_example(example):
    input_ids = tokenizer.tokenizer(
        example[tokenizer.text_field],
        truncation=False,
        padding=False,
        add_special_tokens=False,
    ).input_ids
    positions = list(range(len(input_ids)))
    return {"input_ids": input_ids, "positions": positions}


def tokenize_subset(subset, azure_client_kwargs):
    ntokens_subset = []
    for batchi, batch_ds in enumerate(
        gen_filtered_batches(
            subset,
            model_string=model_string,
            azure_client_kwargs=AZURE_CLIENT_KWARGS,
            to_pandas=False,
        )
    ):
        subset, batchi, ntokens = tokenize_batch(
            subset, batch_ds, batchi, azure_client_kwargs
        )
        ntokens_subset.append(ntokens)
        batch_ds.cleanup_cache_files()
    return subset, ntokens_subset


def tokenize_batch(subset, batch_ds, batchi, azure_client_kwargs):
    print(f"{subset} - {batchi}")
    full_path = (
        "quratingfilttoken"
        f"/tokenizer_{tokenizer_model.split('/')[-1]}"
        f"/{model_string}"
        f"/{subset}"
        f"/{subset}_{batchi :04d}.parquet"
    )
    client = AzureBlobClient(**azure_client_kwargs)
    cloud_path = CloudPath(f"az://{full_path}", client=client)
    if cloud_path.exists():
        print(f"Skipping batch: {subset} - {batchi}")
        return
    keep_cols = ["id"]
    processed_ds = batch_ds.map(
        tokenize_example,
        batched=True,
        remove_columns=[col for col in batch_ds.column_names if col not in keep_cols],
        batch_size=2000,
    )
    print(f"Uploading output dataset to: {full_path}")
    with BytesIO() as bts:
        processed_ds.to_parquet(bts)
        bts.seek(0)
        cloud_path.write_bytes(bts.read())

    ntokens = [len(iids) for iids in processed_ds["input_ids"]]
    
    processed_ds.cleanup_cache_files()

    return subset, batchi, ntokens


with Parallel(n_jobs=cpu_count() - 2, verbose=60) as p:
    out = p(
        delayed(tokenize_subset)(subset, azure_client_kwargs=AZURE_CLIENT_KWARGS)
        for subset in configs
    )

# %%
ntokens_tot = 0
for subset, ntokens_subset in out:
    for batchi, ntokens_batch in ntokens_subset:
        ntokens_batch_tot = sum(ntokens_batch) * 1e-6
        print(f"{subset} - {batchi}: {ntokens_batch :.03f}")
        ntokens_tot += ntokens_batch

print(f"Total - {ntokens_tot :.03f}")
