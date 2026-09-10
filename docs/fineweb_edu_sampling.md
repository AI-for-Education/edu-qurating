# Fineweb-Edu Sampling

We approximately randomly sample from large fineweb-edu-fortified dataset.
The dataset is actually split over 95 different datasets, each corresponding to
a different dump of the common crawl.

Each of these datasets is further split into ~20-50 shards.

Due to the size of the datasets, we load as a streaming dataset, which complicates
randomization. Randomization of streaming datasets relies on a buffer which is filled
in memory. This buffer is filled *in order*, such that a buffer of size n will be initially
filled with the first n rows of the dataset. Samples are drawn randomly from the buffer, and the buffer is filled with the next rows in order after each sample is taken. Shards are also randomised, but importantly, the buffer still fills with contiguous rows from the same shard.

The only way to get true randomisation is if buffer_size == n_rows.

The way that we try to approximate better randomness is as follows:
- split each dataset manually into shards
- get the number of rows in each shard of each dataset
- calculate the total number of rows over all datasets x shards
- calculate sampling probability from each dataset x shard as n_ds_shard / n_total
- convert this to n_samples_ds_shard by sampling from a multinomial distribution
- iterate over each dataset x shard and apply the streaming dataset shuffle method,
    with a buffer of no larger than 20000 rows
- take the first n_samples_ds_shard from each shuffled dataset x shard
- concatenate all of this together into a single new dataset in memory
- shuffle the rows (true shuffle) of this dataset to interleave rows between shards

The benefit of this approach over the built-in streaming shuffle is that the final shuffled dataset is no longer necessarily contiguous within shards.