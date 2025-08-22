# %%
import sys
from argparse import ArgumentParser

from fdllm import register_models
from datasets import load_dataset, get_dataset_config_names, Dataset
import nest_asyncio

from qurating.prompting.score_pairwise import Comparator
from qurating.constants import DATASETS_DIR, TEMPLATES_DIR, RESULTS_DIR, ROOT

nest_asyncio.apply()

register_models(ROOT / "custom_models.yaml")

n_samples = 10000
seed = 72353534

dataset_base = f"fwe-fortified_sampled-{n_samples}_seed-{seed}"

# %%
dataset = Dataset.from_parquet(
    str(DATASETS_DIR / f"{dataset_base}.parquet"),
    keep_in_memory=True,
)

# %%
NUM_EXAMPLES = 2000
TOKENS_MAX = 16000
MODEL = "gpt-4.1-mini"

parser = ArgumentParser()
Comparator.add_args(parser)

template_files = TEMPLATES_DIR.glob("pairwise_*.txt")

results_base = dataset_base
for template_file in template_files:
    template_base = template_file.stem
    print(template_base)
    
    out_dir = RESULTS_DIR / results_base / template_base
    result_filename = f"{MODEL}_nexamples-{NUM_EXAMPLES}.parquet"

    arg_strs = [
        f"--template_file {template_file}",
        f"--model {MODEL}",
        f"--tokens_max {TOKENS_MAX}",
        f"--num_examples {NUM_EXAMPLES}",
    ]

    args = parser.parse_args([arg for argstr in arg_strs for arg in argstr.split()])

    comp = Comparator(args)

    output = comp.apply(dataset)

    out_dir.mkdir(exist_ok=True, parents=True)
    output.to_parquet(out_dir / result_filename)
