# %%
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import ModelCard

from qurating.scoring_projects.fwe_fortified.model_manager import QuratingModelManager

load_dotenv(override=True)

HERE = Path(__file__).resolve().parent

UPLOAD = False

# %%
modman = QuratingModelManager()

if UPLOAD:
    modman.upload_all()

# %%
base_model = "gemma-3-4b"

for model_type in modman.model_types(base_model):

    modman._download_s3(base_model, model_type)

    model_name = f"edu-qurating-{model_type.replace('_', '-')}-{base_model}"
    model_card_file = HERE / "model_cards" / f"{model_name}.md"

    model = AutoModelForSequenceClassification.from_pretrained(
        modman.model_folder(base_model, model_type, local_path=True)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        modman.model_folder(base_model, model_type, local_path=True)
    )

    repo = f"AI-for-Education/{model_name}"

    model.push_to_hub(repo, max_shard_size="5GB", private=True)
    tokenizer.push_to_hub(repo)

    model_card = ModelCard.load(model_card_file)
    model_card.push_to_hub(repo)


# %%
##### Test code for model card

import json

from datasets import load_dataset, Dataset
from qurating.inference import ModelAnnotator, TokenizeAndChunk

model_name = "AI-for-Education/edu-qurating-core-ed-gemma-3-4b"
# model_name = "AI-for-Education/qurater_gemma-3-4b-pt_ds-ours_v2-200000"

device_batch_size = 500
text_field = "text"

# labels = [
#     "education_level",
#     "education_level_primary",
#     "education_level_secondary",
#     "factual_accuracy",
#     "lesson_engagement",
#     "pedagogical_structure",
# ]

labels = None

annotator = ModelAnnotator(
    model_name, labels=labels, device_batch_size=device_batch_size
)
tokenizer = TokenizeAndChunk(model_name, text_field=text_field)

### Test with the the cosmopedia_v2 dataset.
### Load as streaming dataset because we only want to take a few rows
ds = load_dataset(
    "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True
)

# get the first 10 rows
ds_first10 = Dataset.from_list(ds.take(10).to_list())

#################
## The scoring process is split into 2 stages:
## 1. tokenize and chunk
## 2. annotate

# stage 1
tokenized = ds_first10.map(tokenizer, batched=True)

# stage 2
keep_columns = [text_field, "audience"]
remove_columns = [col for col in tokenized.column_names if col not in keep_columns]
scored = tokenized.map(
    annotator, batched=True, with_indices=True, remove_columns=remove_columns
)

# take a look at the items with scores
print(json.dumps(scored.to_list(), indent=2))


# %%
## test code 2 for model card

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# model_name = "AI-for-Education/qurating-core-ed-gemma-3-4b"
model_name = "AI-for-Education/qurater_gemma-3-4b-pt_ds-ours_v2-200000"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

text = "A plant uses sunlight, water, and carbon dioxide to make sugar through photosynthesis."

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=2048,
)
inputs = {
    key: value.to(next(model.parameters()).device) for key, value in inputs.items()
}

with torch.no_grad():
    outputs = model(**inputs)
    scores = outputs.logits[0].float().cpu()

labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

for label, score in zip(labels, scores):
    print(f"{label}: {score:.4f}")
