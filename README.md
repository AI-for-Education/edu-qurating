
# Fab-QuRating

Using the [QuRating](https://arxiv.org/abs/2402.09739) method to filter LLM datasets for educational quality. 

- [Upstream repo](https://github.com/princeton-nlp/QuRating) is in `upstream/` (with history)
- New `uv` venv with `dvc` for data files

## Plan

- Use the qurating approach to score [fineweb-edu-fortified](https://huggingface.co/datasets/airtrain-ai/fineweb-edu-fortified) (a de-duplicated version of fineweb-edu) on 4 [rubrics](scripts/rubrics.md) for different aspects of educational quality.

### Step 1

- Obtain pairwise judgements from a "good" model on a sample of data
  - Test and iterate the pairwise judgement rubrics
  - Scale up, to have around 250k pairwise examples

#### Notes

- Upstream tokenizes input texts to sample exactly 512 tokens for each comparison. But how important is this (not sure which model will finetune or which tokenizer to use). Text lengths are sampled randomly anyway so it is just about setting a max, but maybe a character based "close enough" sampling would still give valid pairwise judgements, and doesn't matter if a little bit is cut off of the end when training the regression model?

### Step 2

- Training the regression model. 
  - Use a more modern architecture?
  - What compute / storage infrastructure?
  - Might need to move from uv to pixi to support torch binaries


### Using this project

This project uses [uv](https://github.com/astral-sh/uv) to manage a reproducible Python environment. To get started you can run `uv sync` and point VScode to the `.venv` environment.

This project uses [dvc](https://dvc.org/) for data version control. You will need to add the azure access key to `.dvc/config.local` or with this command:

- `uv run dvc remote modify --local azure account_key XXX`

You can then run `uv run dvc pull` to obtain the data files. 
When entire folders are tracked `uv run dvc data status --granular` is a useful way to see all changes.