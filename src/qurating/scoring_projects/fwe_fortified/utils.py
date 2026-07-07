import time

import numpy as np
from datasets import load_dataset, get_dataset_config_names


def init_dataset(wait=0.0):
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
        if wait > 0:
            time.sleep(float(wait))

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
