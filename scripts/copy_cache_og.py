# %%
from pathlib import Path
from shutil import copyfile

from qurating.constants import TEMPLATES_DIR, CACHE_DIR

n_samples = 500000
seed = 72353534

dataset_base = f"fwe-fortified_sampled-{n_samples}_seed-{seed}"

# %%
from_num_examples = 400000
to_num_examples = 500000
TOKENS_MAX = 512
MODEL = "gpt-4.1-mini"

max_concurrency = 20
use_logprobs = True
use_logprobs_suffix = "_use-logprobs" if use_logprobs else ""

use_templates_dir = TEMPLATES_DIR / "ours_v2"

template_files = use_templates_dir.glob("pairwise_*.txt")

results_base = dataset_base
results_dict = {}
for template_file in template_files:
    if template_file.parent != TEMPLATES_DIR:
        template_parent = template_file.parent.name
    else:
        template_parent = "default"
    template_base = f"{template_parent}/{template_file.stem}"

    from_cache_dir = (
        Path(CACHE_DIR)
        / f"{template_file.stem}_{MODEL}_{TOKENS_MAX}_{from_num_examples}{use_logprobs_suffix}"
    )
    to_cache_dir = (
        Path(CACHE_DIR)
        / f"{template_file.stem}_{MODEL}_{TOKENS_MAX}_{to_num_examples}{use_logprobs_suffix}"
    )        
    if from_cache_dir.exists():
        print(from_cache_dir)
        print(to_cache_dir)
        to_cache_dir.mkdir(exist_ok=True, parents=True)
        for fl in from_cache_dir.glob("*"):
            copyfile(fl, to_cache_dir / fl.name)
    else:
        print(f"Doesn't exist: {from_cache_dir}")