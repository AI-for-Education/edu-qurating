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

n_samples = 20000
seed = 72353534

dataset_base = f"fwe-fortified_sampled-{n_samples}_seed-{seed}"

# %%
dataset = Dataset.from_parquet(
    str(DATASETS_DIR / f"{dataset_base}.parquet"),
    keep_in_memory=True,
)

# %%
NUM_EXAMPLES = 500
TOKENS_MAX = 16000
MODEL = "gpt-4.1-mini"

use_templates_dir = TEMPLATES_DIR / "ours"

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
    
    out_dir = RESULTS_DIR / results_base / template_base
    result_filename = f"{MODEL}_nexamples-{NUM_EXAMPLES}.parquet"

    results_ds = Dataset.from_parquet(str(out_dir / result_filename))
    results_dict[template_file.stem] = results_ds

# %%
test_ds = next(iter(results_dict.values()))