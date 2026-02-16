from tempfile import NamedTemporaryFile
from functools import reduce
import gc
from io import BytesIO

import pyarrow as pa
import pandas as pd
from cloudpathlib import CloudPath, AzureBlobClient
from datasets import Dataset, load_dataset


def filter_subset(subset, filters, model_string, azure_client_kwargs):
    pool = pa.default_memory_pool()

    fwe_ds = load_dataset(
        "airtrain-ai/fineweb-edu-fortified",
        name=subset,
        split="train",
        streaming=True,
        token=True,
    )
    fwe_iter = iter(fwe_ds)
    iteri = 0
    itertot = fwe_ds.info.splits["train"].num_examples
    for batchi, batch_scores in enumerate(
        gen_score_batches(subset, model_string, azure_client_kwargs)
    ):
        full_path = (
            f"quratingfiltered/{model_string}/{subset}/{subset}_{batchi :04d}.parquet"
        )
        client = AzureBlobClient(**azure_client_kwargs)
        cloud_path = CloudPath(f"az://{full_path}", client=client)
        if cloud_path.exists():
            print(f"Skipping batch: {subset} - {batchi}")
            continue
        filtered_rows_list = []
        for _, score_row in batch_scores.iterrows():
            prog = 100 * (iteri / itertot)
            if prog * 100 % 10 == 0:
                print(f"{subset}: {prog :.03f}%")
            fwe_row = next(fwe_iter)
            filt_list = [score_row[col] > val for col, val in filters.items()]
            filt = reduce(lambda a, b: a and b, filt_list)
            if score_row["id"] != fwe_row["id"]:
                errstr = f"ID mismatch: {subset} - row {iteri}"
                print(errstr)
                raise ValueError(errstr)
            if filt:
                filtered_rows_list.append(
                    {
                        **{
                            key: val
                            for key, val in fwe_row.items()
                            if key not in ["embeddings"]
                        },
                        **{
                            key: val
                            for key, val in score_row.items()
                            if key.endswith("_average")
                        },
                    }
                )
            iteri += 1
        out_ds = Dataset.from_list(filtered_rows_list)
        print(f"Uploading output dataset to: {full_path}")
        with BytesIO() as bts:
            out_ds.to_parquet(bts)
            bts.seek(0)
            cloud_path.write_bytes(bts.read())
        out_ds.cleanup_cache_files()
        print(
            f"Allocated: {pool.bytes_allocated() * 1e-6}, Available: {pool.max_memory() * 1e-6}"
        )
        gc.collect()
        print(
            f"Allocated: {pool.bytes_allocated() * 1e-6}, Available: {pool.max_memory() * 1e-6}"
        )
    # ####################################
    # ds = Dataset.from_list(filtered_rows_list)
    # del filtered_rows_list
    # ## save dataset
    # with TemporaryDirectory() as tdir:
    #     savedir = Path(tdir) / "dataset"
    #     ds.save_to_disk(str(savedir), max_shard_size="50MB")
    #     fl_list = sorted(savedir.glob("*"))
    #     for fl in tqdm(fl_list):
    #         full_path = (
    #             f"quratingfiltered/{model_string}/{subset}/{fl.relative_to(savedir)}"
    #         )
    #         cloud_path = CloudPath(f"az://{full_path}", client=client)
    #         cloud_path.upload_from(fl)
    # ds.cleanup_cache_files()
    # del ds
    # pool.release_unused()


def load_score_subset(subset, model_string, azure_client_kwargs):
    df_list = list(gen_score_batches(subset, model_string, azure_client_kwargs))
    return pd.concat(df_list, axis=0, ignore_index=True)


def gen_score_batches(subset, model_string, azure_client_kwargs):
    batchi = 0
    while True:
        df = load_score_batch(subset, batchi, model_string, azure_client_kwargs)
        if df is None:
            break
        batchi += 1
        yield df


def load_score_batch(subset, batchi, model_string, azure_client_kwargs):
    client = AzureBlobClient(**azure_client_kwargs)
    full_path = f"quratingscores/{model_string}/{subset}/{subset}_{batchi :04d}.parquet"
    cloud_path = CloudPath(f"az://{full_path}", client=client)
    ## return null if it doesn't exist
    if not cloud_path.exists():
        return
    ## otherwise return id and average score cols
    with NamedTemporaryFile(mode="+wb") as f:
        f.write(cloud_path.read_bytes())
        f.seek(0)
        ds = Dataset.from_parquet(f.name, keep_in_memory=True)
        print(f"Len batch: {subset} - {batchi}: {len(ds)}")
    score_cols = ["id", *[col for col in ds.column_names if col.endswith("_average")]]

    ds_df = ds.select_columns(score_cols).to_pandas()
    ds.cleanup_cache_files()

    return ds_df


def gen_filtered_batches(subset, model_string, azure_client_kwargs, to_pandas=True):
    batchi = 0
    while True:
        df = load_filtered_batch(subset, batchi, model_string, azure_client_kwargs, to_pandas=to_pandas)
        if df is None:
            break
        batchi += 1
        yield df


def load_filtered_batch(subset, batchi, model_string, azure_client_kwargs, to_pandas=True):
    client = AzureBlobClient(**azure_client_kwargs)
    full_path = (
        f"quratingfiltered/{model_string}/{subset}/{subset}_{batchi :04d}.parquet"
    )
    cloud_path = CloudPath(f"az://{full_path}", client=client)
    ## return null if it doesn't exist
    if not cloud_path.exists():
        return
    ## otherwise return id and average score cols
    with NamedTemporaryFile(mode="+wb") as f:
        f.write(cloud_path.read_bytes())
        f.seek(0)
        ds = Dataset.from_parquet(f.name, keep_in_memory=True)
        print(f"Len batch: {subset} - {batchi}: {len(ds)}")

    if to_pandas:
        ds_df = ds.to_pandas()
        ds.cleanup_cache_files()
        return ds_df
    else:
        return ds


def gen_filtered_batches_info(subset, model_string, azure_client_kwargs):
    batchi = 0
    while True:
        info = load_filtered_batch_info(
            subset, batchi, model_string, azure_client_kwargs
        )
        if info is None:
            break
        batchi += 1
        yield info


def load_filtered_batch_info(subset, batchi, model_string, azure_client_kwargs):
    client = AzureBlobClient(**azure_client_kwargs)
    full_path = (
        f"quratingfiltered/{model_string}/{subset}/{subset}_{batchi :04d}.parquet"
    )
    cloud_path = CloudPath(f"az://{full_path}", client=client)
    ## return null if it doesn't exist
    if not cloud_path.exists():
        return
    ## otherwise return info
    info = cloud_path.stat()

    return info
