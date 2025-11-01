# %%
from argparse import ArgumentParser

from dotenv import load_dotenv
from fdllm import register_models
from datasets import concatenate_datasets, Dataset
# import nest_asyncio

from qurating.prompting.score_pairwise import Comparator
from qurating.constants import DATASETS_DIR, TEMPLATES_DIR, RESULTS_DIR, ROOT

# nest_asyncio.apply()

load_dotenv(override=True)

register_models(ROOT / "custom_models.yaml")

n_samples = 500000
seed = 72353534

dataset_base = f"fwe-fortified_sampled-{n_samples}_seed-{seed}"

# %%
parquetf = DATASETS_DIR / f"{dataset_base}.parquet"
if parquetf.exists():
    dataset = Dataset.from_parquet(str(parquetf), keep_in_memory=True)
else:
    dataset = Dataset.load_from_disk(
        str(DATASETS_DIR / f"{dataset_base}"),
        keep_in_memory=True,
    )

# %%
NUM_EXAMPLES = 200000
OFFSET = 0
TOKENS_MAX = 512
MODEL = "gemini-2.5-flash-vertex"
# MODEL = "gpt-4.1-mini"
# MODEL = "gpt-5-mini-2025-08-07-minimal"
# MODEL = "gemini-2.5-flash-preview-05-20"
# MODEL = "claude-sonnet-4-5-20250929"
# MODEL = "claude-3-5-haiku-20241022"

max_concurrency = 100
use_logprobs = True
use_logprobs_suffix = "_use-logprobs" if use_logprobs else ""

offset_suffix = f"_offset-{OFFSET}" if OFFSET != 0 else ""

use_templates_dir = TEMPLATES_DIR / "ours_v2"

parser = ArgumentParser()
Comparator.add_args(parser)

template_files = use_templates_dir.glob("pairwise_*.txt")

results_base = dataset_base
results_dict = {}
for template_file in template_files:
    if template_file.parent != TEMPLATES_DIR:
        template_parent = template_file.parent.name
    else:
        template_parent = "default"
    template_base = f"{template_parent}/{template_file.stem}"
    print(template_base)

    out_dir = RESULTS_DIR / f"tokens_max_{TOKENS_MAX}" / results_base / template_base
    result_path = (
        out_dir
        / f"{MODEL}_nexamples-{NUM_EXAMPLES}{offset_suffix}{use_logprobs_suffix}.parquet"
    )
    if NUM_EXAMPLES > 20000:
        result_path = result_path.parent / result_path.stem
    if not result_path.exists():
    # if True:
        arg_strs = [
            f"--template_file {template_file}",
            f"--model {MODEL}",
            f"--tokens_max {TOKENS_MAX}",
            f"--num_examples {NUM_EXAMPLES}",
            f"--max-concurrency {max_concurrency}",
            f"--offset {OFFSET}",
        ]
        if use_logprobs:
            arg_strs.append("--logprobs")

        args = parser.parse_args([arg for argstr in arg_strs for arg in argstr.split()])

        comp = Comparator(args)
        print(comp.args.tokens_max)

        output = comp.apply(dataset)

        out_dir.mkdir(exist_ok=True, parents=True)
        if NUM_EXAMPLES > 20000:
            output.save_to_disk(result_path, max_shard_size="200MB")
        else:
            output.to_parquet(result_path)
    if NUM_EXAMPLES > 20000:
        results_ds = Dataset.load_from_disk(result_path)
    else:
        results_ds = Dataset.from_parquet(str(result_path))
    results_dict[template_file.stem] = results_ds

# %%
shared_columns = ["texts", "indices", "examples"]
datasets = []
for template_name, dataset in results_dict.items():
    prefix = "_".join(template_name.split("_")[1:])
    for column in dataset.column_names:
        if column not in shared_columns:
            dataset = dataset.rename_column(column, prefix + "_" + column)
    datasets.append(dataset)

for shared_column in shared_columns:
    assert all(ds[shared_column] == datasets[0][shared_column] for ds in datasets[1:])
    datasets = [datasets[0]] + [ds.remove_columns(shared_column) for ds in datasets[1:]]

dataset: Dataset = concatenate_datasets(datasets, axis=1)
if len(dataset) > 10000:
    outfile = (
        RESULTS_DIR
        / f"tokens_max_{TOKENS_MAX}"
        / results_base
        / template_parent
        / f"combined_{MODEL}_nexamples-{NUM_EXAMPLES}{offset_suffix}{use_logprobs_suffix}"
    )
    dataset.save_to_disk(outfile, max_shard_size="200MB")
else:
    outfile = (
        RESULTS_DIR
        / f"tokens_max_{TOKENS_MAX}"
        / results_base
        / template_parent
        / f"combined_{MODEL}_nexamples-{NUM_EXAMPLES}{offset_suffix}{use_logprobs_suffix}.parquet"
    )
    dataset.to_parquet(outfile)
