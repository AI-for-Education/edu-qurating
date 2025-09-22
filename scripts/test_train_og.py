"""
Script to test and help to understand the process of going from pairwise judgements
to rating scores for each judgement criterion.

Contains excerpts of code from upstream training package
"""

# %%
from typing import Any, Dict
from collections import namedtuple

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
import matplotlib.pyplot as plt

from qurating.constants import RESULTS_DIR


"""
DataCollator is the class that is responsible for taking batches from input dataset and returning input
to the model. Taken from training.train_preference_model.

After instantiation, it is passed as the "data_collator" parameter to transformers.trainer.Trainer (which
is subclassed by training.train_preference_model.PreferenceTrainer.

Returns a dict of:
    "input_ids": [n_rows*2, token_window] tensor of token indices for each pair of texts in the input batch.
        Each row of the input batch consists of a pair of texts, but these are separated here, such that
        each row of "input_ids" is a distint text.
    "attention_mask": haven't investigated the details of this one but I think it is probably uniform ones
        and same dimensions as "input_ids". Need to check.
    "labels": [nrows*2, nrows*2, nlabels] tensor of pairwise probabilities of selecting text i over text j on 
        criterion k.
        Each row of the input batch contains the [2 x 2 x nlabels] choice probabilities associated with the pair 
        of texts corresponding to that row. This [2 x 2 x nlabels] tensor is inserted at the slice [i*2:i*2+1, i*2:i*2+1, :]
        for row i of the input (in other words, the 2 x 2 square centred on ith the diagional element). All elements of
        the output tensor outside of this area around the diagonal are treated as missing data for the loss calculation.
"""
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
## load the tokenizer for Sheared-LLaMa-1.3b

tokenizer = AutoTokenizer.from_pretrained(
    "princeton-nlp/Sheared-LLaMA-1.3b",
    use_fast=True,
    legacy=False,
)
tokenizer.pad_token_id = 0

# %%
## load the preferences dataset

n_samples = 20000
seed = 72353534

model_name = "gpt-4.1-mini"
num_examples = 500

dataset_base = f"fwe-fortified_sampled-{n_samples}_seed-{seed}"

dataset_file = (
    RESULTS_DIR
    / f"tokens_max_512"
    / dataset_base
    / "ours_v2"
    / f"combined_{model_name}_nexamples-{num_examples}.parquet"
)

dataset = Dataset.from_parquet(str(dataset_file))

# get the label names from the dataset (i.e. the criterion names)
label_names = [col for col in dataset.column_names if col.endswith("_average")]

# %%
# instantiate DataCollator

# this namedtuple is just a hack to pass in an object that functions like args from ArgumentParser
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
# load config for Sheared-LLaMA-1.3b
config = AutoConfig.from_pretrained("princeton-nlp/Sheared-LLaMA-1.3b")
# adjust the number of output labels to match
config.num_labels = len(label_names)

# instantiate model from updated config
model = AutoModelForSequenceClassification.from_pretrained(
    "princeton-nlp/Sheared-LLaMA-1.3b", config=config
)

# %%
# pass first 4 rows of dataset to DataCollator
collected = dc(dataset.take(4))

# take a quick look at the output
plt.imshow(collected["labels"][:, :, 0], vmin=0, vmax=1)
plt.colorbar()

# %%
# pop labels from collected outputs (these are not inputs to the model)
labels = collected.pop("labels")

"""
They are used in loss calculation in training.train_preference_model.PreferenceTrainer.compute_loss
(lines 235 -239 are the key lines that convert the logits output of the model to target preference 
probabilities for comparison with labels):
    ```
       labels = inputs.pop("labels")
       outputs = model(**inputs, use_cache=False)
       logit_diffs = outputs.logits.unsqueeze(0) - outputs.logits.unsqueeze(1)

       probs = logit_diffs.float().sigmoid()
    ```
"""

# run inference on the inputs
outputs = model(**collected, use_cache=False)
# convert the logits (ntexts x nlabels) to pairwise preference probablities (ntexts x ntexts x nlabels)
logit_diffs = outputs.logits.unsqueeze(0) - outputs.logits.unsqueeze(1)
