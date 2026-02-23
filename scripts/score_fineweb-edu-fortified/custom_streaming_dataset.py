import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlparse

from cloudpathlib import AzureBlobClient, CloudPath
from datasets import Dataset, IterableDataset, get_dataset_config_names, load_dataset
from dotenv import load_dotenv

load_dotenv(override=True)


def load_dataset_possibly_azure(path: str, *args, **kwargs):
    if path.startswith("az://"):
        sa, container, model_string = parse_azure_url(path)
        azure_client_kwargs = {
            "account_url": f"https://{sa}.blob.core.windows.net",
            "credential": os.getenv("QURATING_SCORES_AZURE_STORAGE_KEY"),
        }
        shards = data_generator_sharder(
            container, model_string, azure_client_kwargs=azure_client_kwargs
        )
        features = data_generator_features(
            container, model_string, azure_client_kwargs=azure_client_kwargs
        )
        ds = IterableDataset.from_generator(
            data_generator,
            features=features,
            gen_kwargs={
                "shards": shards,
                "model_string": model_string,
                "container": container,
                "azure_client_kwargs": azure_client_kwargs,
            },
        )
        return ds
    else:
        return load_dataset(path, *args, **kwargs)


def parse_azure_url(url):
    purl = urlparse(url)
    sa = purl.netloc
    file_path = purl.path
    container, model_string, *_ = file_path.lstrip("/").split("/")

    return sa, container, model_string


def count_filtered_batches(subset, container, model_string, azure_client_kwargs):
    client = AzureBlobClient(**azure_client_kwargs)
    full_path = f"{container}/{model_string}/{subset}"
    cloud_path = CloudPath(f"az://{full_path}", client=client)
    fllist = list(cloud_path.glob(f"*{subset}_*.parquet"))
    return len(fllist)


def load_filtered_batch(
    subset,
    batchi,
    container,
    model_string,
    azure_client_kwargs,
    to_pandas=True,
):
    client = AzureBlobClient(**azure_client_kwargs)
    full_path = f"{container}/{model_string}/{subset}/{subset}_{batchi :04d}.parquet"
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


def data_generator_sharder(container, model_string, azure_client_kwargs):
    configs = get_dataset_config_names("airtrain-ai/fineweb-edu-fortified")
    shards = []
    for subset in configs:
        print(subset)
        n_batches = count_filtered_batches(
            subset, container, model_string, azure_client_kwargs
        )
        for batchi in range(n_batches):
            shards.append((subset, batchi))
    return shards


def data_generator_features(container, model_string, azure_client_kwargs):
    configs = get_dataset_config_names("airtrain-ai/fineweb-edu-fortified")
    batch = load_filtered_batch(
        configs[0], 0, container, model_string, azure_client_kwargs, to_pandas=False
    )
    features = batch.features
    return features


def data_generator(shards, container, model_string, azure_client_kwargs):
    for subset, batchi in shards:
        batch_ds = load_filtered_batch(
            subset,
            batchi,
            container,
            model_string,
            azure_client_kwargs,
            to_pandas=False,
        )
        for row in batch_ds:
            yield row
