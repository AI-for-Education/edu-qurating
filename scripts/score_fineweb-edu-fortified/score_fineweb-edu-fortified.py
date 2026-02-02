# %%
from pathlib import Path
import os
from io import BytesIO
from tempfile import NamedTemporaryFile
import time

import torch
import typer
from datasets import Dataset, load_dataset
import numpy as np
from dotenv import load_dotenv
from cloudpathlib import CloudPath, AzureBlobClient

from qurating.constants import RESULTS_DIR
from qurating.inference import ModelAnnotator, TokenizeAndChunk


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


def main(subsets: list[str]):
    load_dotenv(override=True)
    AZURE_CLIENT_KWARGS = {
        "account_url": "https://quratingscoressa.blob.core.windows.net",
        "credential": os.getenv("AZURE_STORAGE_KEY"),
    }

    cuda_avail = torch.cuda.is_available()
    print(cuda_avail)
    print(torch.cuda.get_device_name(0))
    print("__CUDNN VERSION:", torch.backends.cudnn.version())
    print("__Number CUDA Devices:", torch.cuda.device_count())
    print("__CUDA Device Name:", torch.cuda.get_device_name(0))
    print(
        "__CUDA Device Total Memory [GB]:",
        torch.cuda.get_device_properties(0).total_memory / 1e9,
    )
    print("Memory Usage:")
    print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
    print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")

    if not cuda_avail:
        raise ValueError("CUDA not available")

    annotator, tokenizer, model_name = init_annotator()
    model_string = [
        substr for substr in Path(model_name).parts if substr.startswith("qurater_")
    ]
    assert len(model_string) == 1
    model_string = model_string[0]

    client = AzureBlobClient(**AZURE_CLIENT_KWARGS)

    for subset in subsets:

        keep_cols = ["id"]
        ds = load_dataset(
            "airtrain-ai/fineweb-edu-fortified",
            name=subset,
            split="train",
            streaming=True,
            token=True,
        )
        outer_batch_size = 100000

        for batchi, batch_ds in enumerate(ds.batch(batch_size=outer_batch_size)):
            ############################
            full_path = (
                f"quratingscores/{model_string}/{subset}/{subset}_{batchi :04d}.parquet"
            )
            cloud_path = CloudPath(f"az://{full_path}", client=client)
            if cloud_path.exists():
                continue
            ############################
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
            # upload output dataset to blob with validation
            valid = False
            retry_cnt = 0
            while not valid:
                retry_cnt += 1
                if retry_cnt > 3:
                    break
                print(f"Uploading output dataset to: {full_path}")
                st = time.perf_counter()
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
                valid = np.all(temp_ds.to_pandas() == out_ds.to_pandas()).item()
            assert valid
            print(f"Elapsed time: {time.perf_counter() - st :.04f}")
            time.sleep(0.1)
            ###################


if __name__ == "__main__":
    typer.run(main)
