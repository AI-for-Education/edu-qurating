# %%
from pathlib import Path
from argparse import ArgumentParser

from dotenv import load_dotenv
import tiktoken
import numpy as np

from fdllm import get_caller, LLMMessage
from fdllm.sysutils import register_models
from datasets import Dataset

from qurating.constants import DATASETS_DIR, TEMPLATES_DIR
from qurating.prompting.score_pairwise import Comparator

load_dotenv(override=True)

register_models(Path.home() / ".fdllm/custom_models.yaml")

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
template_file = TEMPLATES_DIR / (r"ours_v2\pairwise_education_level_primary.txt")
MODEL = "gpt-4.1-mini"
TOKENS_MAX = 512
NUM_EXAMPLES = 10

arg_strs = [
    f"--template_file {template_file}",
    f"--model {MODEL}",
    f"--tokens_max {TOKENS_MAX}",
    f"--num_examples {NUM_EXAMPLES}",
]

parser = ArgumentParser()
Comparator.add_args(parser)

args = parser.parse_args([arg for argstr in arg_strs for arg in argstr.split()])
comparator = Comparator(args)

# %%
caller = get_caller(MODEL)

# %%
enc = tiktoken.encoding_for_model(caller.Model.Api_Model_Name)
labels = ["A", "B"]
label_tokens = [enc.encode(label) for label in labels]
logit_bias = {
    str(token): 100 for token in set.union(*(set(tokens) for tokens in label_tokens))
}
max_tokens = max(len(tokens) for tokens in label_tokens)

# %%
idx = 6
prompt = comparator.args.template.format(
    text_a=dataset[idx]["text"],
    text_b=dataset[idx+1]["text"],
    label_a=comparator.args.labels[0],
    label_b=comparator.args.labels[1],
)
msg = LLMMessage(Role="user", Message=prompt)

out = caller.call(
    msg,
    max_tokens=max_tokens,
    logit_bias=logit_bias,
    logprobs=True,
    top_logprobs=20,
)

lp = {lpt.token: lpt.logprob for lpt in out.LogProbs.content[0].top_logprobs}
print(lp)
lp = {lab: lp.get(lab, -100) for lab in labels}
print(lp)


def logit_pairs_to_probs(logitsa, logitsb):
    def sigmoid(x):
        odds = np.exp(x)
        return odds / (odds + 1)

    logit_diffs = logitsa - logitsb
    probs = sigmoid(logit_diffs)
    out = np.zeros((probs.shape[0], 2))
    out[:, 0] = probs
    out[:, 1] = 1 - probs
    return out


print(logit_pairs_to_probs(*np.array(list(lp.values()))[:, None]).round(3))
