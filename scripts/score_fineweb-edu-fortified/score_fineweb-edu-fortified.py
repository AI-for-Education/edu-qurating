# %%
from pathlib import Path
import os
from io import BytesIO
from tempfile import NamedTemporaryFile
import time

import typer
from datasets import Dataset, load_dataset, get_dataset_config_names
import numpy as np
from dotenv import load_dotenv
from cloudpathlib import CloudPath, AzureBlobClient

from qurating.constants import RESULTS_DIR
from qurating.inference import ModelAnnotator, TokenizeAndChunk


def init_dataset():
    ### configs are the different datasets (95, corresponding to CC dumps)
    configs = get_dataset_config_names("airtrain-ai/fineweb-edu-fortified")

    print(configs)
    print(len(configs))

    # get last n_configs for testing
    ### NOTE: in fact, changed this now so we just take all of them
    n_configs = len(configs)
    use_configs = configs[-1 : -n_configs - 1 : -1]
    print(use_configs)

    # load each dataset by config name as a streaming dataset into the dict fw
    fw = {}
    for config in use_configs:
        fw[config] = load_dataset(
            "airtrain-ai/fineweb-edu-fortified",
            name=config,
            split="train",
            streaming=True,
            token=True,
        )

    ### length of each dataset
    fw_len = {
        config: fw_.info.splits["train"].num_examples for config, fw_ in fw.items()
    }
    # number of shards in each dataset
    fw_nshards = {config: fw_.num_shards for config, fw_ in fw.items()}

    return fw, fw_len, fw_nshards


def get_partition_subsets(start_partition, end_partition, n_partitions, subset_counts):
    partitions = [[]]
    for subset, n_rows in sorted(
        subset_counts.items(), key=lambda x: x[1], reverse=True
    ):
        if len(partitions) < n_partitions:
            partitions.append([subset])
        else:
            min_part = np.argmin(
                [sum(subset_counts[subset] for subset in part) for part in partitions]
            )
            partitions[min_part].append(subset)
    return sum(partitions[start_partition:end_partition], [])


def init_annotator():
    dataset_file = (
        RESULTS_DIR
        / "tokens_max_512"
        / "fwe-fortified_sampled-500000_seed-72353534"
        / "ours_v2"
        / "combined_gpt-4.1-mini_nexamples-200000_use-logprobs"
    )

    parquetf = Path(dataset_file).with_suffix(".parquet")
    if parquetf.exists():
        ds = Dataset.from_parquet(str(dataset_file))
    elif Path(dataset_file).is_dir():
        ds = Dataset.load_from_disk(dataset_file)
    else:
        raise ValueError(f"{dataset_file} doesn't exist or is not a valid format")
    labels = [
        "_".join(col.split("_")[:-1])
        for col in ds.column_names
        if col.endswith("_average")
    ]
    print(f"Labels: {labels}")

    model = "AI-for-Education/qurater_gemma-3-4b-pt_ds-ours_v2-200000"

    annotator_batch_size = 2000

    annotator = ModelAnnotator(str(model), labels, annotator_batch_size)

    tokenizer = TokenizeAndChunk(str(model), "text", 512)

    return annotator, tokenizer, model


def main(start_partition: int, end_partition: int, n_partitions: int = 32):
    load_dotenv(override=True)
    AZURE_CLIENT_KWARGS = {
        "account_url": "https://quratingscoressa.blob.core.windows.net",
        "credential": os.getenv("AZURE_STORAGE_KEY"),
    }

    fw, subset_counts, fw_nshards = init_dataset()

    subsets = get_partition_subsets(
        start_partition, end_partition, n_partitions, subset_counts
    )

    annotator, tokenizer, model_name = init_annotator()
    model_string = [
        substr for substr in Path(model_name).parts if substr.startswith("qurater_")
    ]
    assert len(model_string) == 1
    model_string = model_string[0]

    client = AzureBlobClient(**AZURE_CLIENT_KWARGS)

    subset = subsets[0]
    # for subset in subsets:

    keep_cols = ["id"]
    ds: Dataset = fw[subset]
    batch_cnt = 0
    max_batch_cnt = 3
    outer_batch_size = 2000

    for batchi, batch_ds in enumerate(ds.batch(batch_size=outer_batch_size)):
        if batch_cnt >= max_batch_cnt:
            break
        batch_cnt += 1
        run_ds = Dataset.from_dict(batch_ds)
        processed_ds = run_ds.map(tokenizer, batched=True, remove_columns=["text"])
        results = processed_ds.map(
            annotator,
            batched=True,
            with_indices=True,
            batch_size=annotator.device_batch_size,
            remove_columns=[
                col for col in processed_ds.column_names if col not in keep_cols
            ],
        )
        ##########################
        # create output dataset
        print("Creating output dataset")
        st = time.perf_counter()
        out_cols = [
            "id",
            *[col for col in results.column_names if col.endswith("_average")],
        ]
        out_ds = results.select_columns(out_cols).add_column(
            "dump", [subset] * len(results)
        )
        print(f"Elapsed time: {time.perf_counter() - st :.04f}")
        # upload output dataset to blob
        full_path = (
            f"quratingscores/{model_string}/{subset}/{subset}_{batchi :04d}.parquet"
        )
        print(f"Uploading output dataset to: {full_path}")
        st = time.perf_counter()
        cloud_path = CloudPath(f"az://{full_path}", client=client)
        with BytesIO() as bts:
            out_ds.to_parquet(bts)
            bts.seek(0)
            cloud_path.write_bytes(bts.read())
        print(f"Elapsed time: {time.perf_counter() - st :.04f}")
        # check that blob is valid
        # i.e. it can be loaded as dataset and values are identical to output dataset
        print("Validating uploaded blob")
        st = time.perf_counter()
        with NamedTemporaryFile(mode="+wb") as f:
            f.write(cloud_path.read_bytes())
            f.seek(0)
            print(f.name)
            temp_ds = Dataset.from_parquet(f.name, keep_in_memory=True)
        assert np.all(temp_ds.to_pandas() == out_ds.to_pandas()).item()
        print(f"Elapsed time: {time.perf_counter() - st :.04f}")
        time.sleep(0.1)
        ###################


if __name__ == "__main__":
    typer.run(main)
