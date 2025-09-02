# %%
import sys
from argparse import ArgumentParser
from typing import Any, Dict
from collections import namedtuple

from fdllm import register_models
from datasets import concatenate_datasets, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import nest_asyncio
import torch
import matplotlib.pyplot as plt

from qurating.constants import DATASETS_DIR, TEMPLATES_DIR, RESULTS_DIR, ROOT

nest_asyncio.apply()

register_models(ROOT / "custom_models.yaml")


class DataCollator:
    def __init__(self, args, training_args, tokenizer):
        self.args = args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.pad_token_id = self.tokenizer.pad_token_id

        self.max_length = self.args.max_length

    @torch.no_grad()
    def __call__(self, features: Any) -> Dict[str, Any]:
        batch = self.tokenizer(
            sum([item[self.args.text_field] for item in features], []),
            add_special_tokens=False,
            truncation=True,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
        )

        bsz = batch.input_ids.size(0)
        num_labels = len(self.args.label_field)
        labels = -100 * torch.ones(bsz, bsz, num_labels, dtype=torch.float32)

        counter = 0
        for item in features:
            k = len(item[self.args.text_field])
            for i, label in enumerate(self.args.label_field):
                labels[counter : counter + k, counter : counter + k, i] = torch.tensor(
                    item[label], dtype=torch.float32
                )
            counter += k

        for i in range(labels.size(-1)):
            if (
                self.args.single_label_ablation >= 0
                and i != self.args.single_label_ablation
            ):
                labels[:, :, i] = -100

        labels[range(bsz), range(bsz)] = -100

        return dict(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=labels,
        )


# %%
tokenizer = AutoTokenizer.from_pretrained(
    "princeton-nlp/Sheared-LLaMA-1.3b",
    # cache_dir=args.cache_dir,
    use_fast=True,
    # revision=args.model_revision,
    # use_auth_token=True if args.use_auth_token else None,
    legacy=False,
)
tokenizer.pad_token_id = 0

# %%
n_samples = 20000
seed = 72353534

model_name = "gpt-4.1-mini"
num_examples = 500

dataset_base = f"fwe-fortified_sampled-{n_samples}_seed-{seed}"

dataset_file = (
    RESULTS_DIR
    / dataset_base
    / "ours"
    / f"combined_{model_name}_nexamples-{num_examples}.parquet"
)

dataset = Dataset.from_parquet(str(dataset_file))
label_names = [col for col in dataset.column_names if col.endswith("_average")]
# %%
argstup = namedtuple(
    "args", ["max_length", "text_field", "label_field", "single_label_ablation"]
)
args = argstup(
    max_length=512,
    text_field="texts",
    label_field=label_names,
    single_label_ablation=-1,
)
dc = DataCollator(args, (), tokenizer=tokenizer)

# %%
config = AutoConfig.from_pretrained("princeton-nlp/Sheared-LLaMA-1.3b")
# config = AutoConfig.from_pretrained("princeton-nlp/QuRater-1.3B")

# config.id2label = {str(i): f"LABEL_{i}" for i in range(len(label_names))}
# config.label2id = {f"LABEL_{i}": str(i) for i in range(len(label_names))}
config.num_labels = 6
model = AutoModelForSequenceClassification.from_pretrained(
    "princeton-nlp/Sheared-LLaMA-1.3b", config=config
)
# model.score = torch.nn.modules.linear.Linear(in_features=2048, out_features=6, bias=False)

# %%
collected = dc(dataset.take(4))

plt.imshow(collected["labels"][:, :, 0], vmin=0, vmax=1)
plt.colorbar()

labels = collected.pop("labels")
outputs = model(**collected, use_cache=False)

# %%
logit_diffs = outputs.logits.unsqueeze(0) - outputs.logits.unsqueeze(1)
