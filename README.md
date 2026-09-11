# Edu-QuRating

Code repository for Edu-QuRating ([paper](https://arxiv.org/abs/2609.09425)).

This initailly started out as a private fork of [QuRating](https://github.com/princeton-nlp/QuRating). The source repo is kept in the folder `upstream` (with history).

Edu-QuRating models on [HuggingFace](https://huggingface.co/collections/AI-for-Education/edu-qurating-models)

## Installation

We use [uv](https://docs.astral.sh/uv/) for package management.

There are 3 installation paths:

- `default`: Installs without support for GPU. Precludes the running of QuRating model training or GRPO experiments.
    ```bash
    cd <REPO_ROOT>
    uv sync 
    ```
- `linux-gpu`: Required to run QuRating model training. Only installable on linux systems with compatible GPU (anything that is compatible with `flash-attn>=2.8.3`).
    ```bash
    cd <REPO_ROOT>
    uv sync --group dev --group linux-gpu
    ```
- `rl`: Required to run GRPO experiments. Should only be used in conjunction with `linux-gpu`, so inherits requirements of that group. Additionally installs `unsloth`, `vllm`, `trl`.
    ```bash
    cd <REPO_ROOT>
    uv sync --group dev --group linux-gpu --group rl
    ```

## Relevant scripts

#### [`scripts/sampling/sample_dataset.py`](scripts/sampling/sample_dataset.py)
- Script to approximately randomly sample from large fineweb-edu-fortified dataset, described in [Fineweb-Edu sampling](docs/fineweb_edu_sampling.md). This was used to generate the sampled datasets used for **core-ed** model in [Pairwise comparisons](docs/pairwise_comparisons.md)
#### [`scripts/sampling/sample_dataset_with_scoring.py`](scripts/sampling/sample_dataset_with_scoring.py)
- Script to approximately randomly sample from large fineweb-edu-fortified dataset while scoring with qurating model at the same time (and filtering by score). This was used to generate sampled datasets that were pre-filtered for core-ed quality and age-relevance to use for [Pairwise comparisons](docs/pairwise_comparisons.md) datasets for the **foundational literacy** models.
#### [`scripts/pairwise_comparisons/run_pairwise_comparisons.py`](scripts/pairwise_comparisons/run_pairwise_comparisons.py)
- Script to run the [Pairwise comparisons](docs/pairwise_comparisons.md) procedure. Inputs to this are the outputs from [Fineweb-Edu sampling](docs/fineweb_edu_sampling.md). Parameters that can be set (as script constants) are:
	- The following 3 all just control which input dataset to load:
		- `n_samples`
		- `seed`
		- `dataset_base`
			- name template for the input dataset, like: `{prefix}-{n_samples}_seed-{seed}`
	- `NUM_EXAMPLES`
		- number of examples to run pairwise comparison on from input dataset. This can only meaningfully go as high as `n_samples`
	- `OFFSET`
		- offset to start examples from. Taken together, the rows of input dataset that are used are: `OFFSET : OFFSET + NUM_EXAMPLES`
	- `TOKENS_MAX`
		- the maximum size of the token window to chunk by
	- `MODEL`
		- the model to use for pairwise comparisons
	- `max_concurrency`
		- the maximum number of threads to use for comparisons
	- `use_logprobs`
		- whether to use the logprobs method (if `True`) or to use the generation sampling method (if `False`). Using logprobs reduces the number of LLM calls by 20x
	- `use_templates_dir`
		- The templates to use, from these options:
			- Core Educational ("ours_v2")
			- Foundational Literacy - Teacher-Facing ("FLN_teacher-facing")
			- Foundational Literacy - Student-Facing ("FLN_student-facing")
- Outputs are saved to `qurating.constants.RESULTS_DIR`. Filename template is like: `tokens_max_{TOKENS_MAX}/{dataset_base}/{use_templates_dir}/combined_{MODEL}_nexamples-{NUM_EXAMPLES}{offset_suffix}{use_logprobs_suffix}`. Depending on the size of the dataset, this will either be a single parquet file (can be loaded with `Dataset.from_parquet`) or a folder of arrow files (can be loaded with `Dataset.load_from_disk`).
#### [`scripts/run_train.sh`](scripts/run_train.sh)
- Shell script to launch [Qurating model training](docs/qurating_model_training.md) via torchrun and handle training params and input / output filenames, etc.
- Final models are based on 12 different parameter variants, chosen out of:
	- `model`:
		- `tomaarsen/Qwen3-Reranker-0.6B-seq-cls`
		- `tomaarsen/Qwen3-Reranker-4B-seq-cls`
		- `tomaarsen/Qwen3-Reranker-8B-seq-cls`
		- `google/gemma-3-4b-pt`
	- `dataset_prefix`:
		- `fwe-fortified_sampled` (core ed)
		- `fwe-fortified_sampled-pedagogical-5` (FL teacher-facing)
		- `fwe-fortified_sampled-primary-5-pedagogical-5` (FL student-facing)
	- `dataset_templates_base`:
		- `ours_v2` (core ed)
		- `FLN_teacher-facing` (FL teacher-facing)
		- `FLN_student-facing` (FL student-facing)
#### [`src/qurating/run_train.py`](src/qurating/run_train.py)
- python script that is launched by the above shell script
#### [`scripts/score_fineweb-edu-fortified/`](scripts/score_fineweb-edu-fortified/)
##### [`score_fineweb-edu-fortified.py`](scripts/score_fineweb-edu-fortified/score_fineweb-edu-fortified.py)
- Worker script that runs on a compute node during [Fineweb-Edu scoring](docs/fineweb_edu_scoring.md)
##### [`score_orchestrator.py`](scripts/score_fineweb-edu-fortified/score_orchestrator.py)
- Orchestrator script that runs on controller node during [Fineweb-Edu scoring](docs/fineweb_edu_scoring.md). This script is responsible for splitting the jobs into sets of roughly equal sizes, creating worker nodes, and sending jobs to them
##### [`monitor_outputs.py`](scripts/score_fineweb-edu-fortified/monitor_outputs.py)
- Script to run from the controller node that monitors the status of each job via the output files that have been created so far
##### [`poll_do_api.py`](scripts/score_fineweb-edu-fortified/poll_do_api.py)
- test script to play around with digital ocean API for droplet creation. This has since been turned into more of a useful script in `scripts/digital_ocean/create_droplet.py`
##### [`filter_fineweb-edu-fortified.py`](scripts/score_fineweb-edu-fortified/filter_fineweb-edu-fortified.py)
- Script to take the scored dataset and produced a filtered version based on some score thresholds. In this case we used **top 50th percentile on all 3 of: factual_accuracy, pedagogical_quality, lesson_engagement**