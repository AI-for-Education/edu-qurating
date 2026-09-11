---
base_model: google/gemma-3-4b-pt
license: mit
library_name: transformers
pipeline_tag: text-classification
tags:
- edu-qurating
- qurating
- education
- text-scoring
- sequence-classification
- reward-model
- gemma
---

# Edu-QuRating Foundational Literacy Student-Facing Scorer (Gemma 3 4B)

https://arxiv.org/abs/2609.09425

This model is an Edu-QuRating sequence-classification scorer for learner-facing foundational-literacy text. It takes a text passage as input and outputs one scalar logit for each student-facing literacy criterion. It is intended for ranking, filtering, corpus curation, and reward-modeling experiments, not for text generation.

## Scores

| Score | Meaning |
| --- | --- |
| `1_oral_language_vocabulary` | Match to a beginner reader's oral language and vocabulary. Higher scores indicate familiar, high-frequency, everyday words and culturally familiar contexts; lower scores indicate more unfamiliar, abstract, academic, or multi-syllabic vocabulary for a young child. |
| `2_phonological_awareness` | Support for noticing the sounds of spoken language. Higher scores indicate stronger textual cues such as rhyme, rhythm, alliteration, or language play that draws attention to word sounds. |
| `3_systematic_phonics` | Decodability based on systematic phonics. Higher scores indicate a higher density of short, simple, phonically regular words and consistent letter-sound patterns; lower scores indicate many irregular or tricky words for beginner readers. |
| `4_reading_fluency` | Support for building reading fluency. Higher scores indicate repetitive syntax, predictable sentence stems, and common repeated sight words that help beginner readers practise accurate and fluent reading. |
| `5_reading_comprehension` | Scaffolding for beginner reading comprehension. Higher scores indicate very simple sentence structure, clear sequencing or story cohesion, and cues that guide interpretation, such as questions or picture-based prompts. |
| `6_writing_expression` | Integration of reading with oral or written expression. Higher scores indicate simple prompts for a child to respond, interact, retell, draw, write, or otherwise express something based on the text. |
| `7_engagement_relevance` | Likelihood of engaging, motivating, and affirming a young learner. Higher scores indicate affirming language, positive emotion, inclusive representation, relatable tone, and child-centric support. |

The scores are rubric-specific model outputs. They are intended to support ranking, filtering, or reward construction.

## Training Procedure

This model was trained as a pairwise-preference distillation model. The training data starts from a 500k-row FineWeb-Edu-Fortified sample and uses the student-facing foundational-literacy prompt set to construct 200k pairwise comparisons. The pairwise judgement excerpts are capped at 512 tokens.

Pairwise labels were generated with GPT-4.1-mini using token log-probabilities. For each criterion-specific prompt, the judge is asked to choose between two excerpts. The log-probabilities of the two answer labels are converted into soft preference probabilities. To reduce order effects, each pair is judged in both text orders, and the two directional probabilities are averaged into one calibrated soft preference label.

For a pair of texts `(x_i, x_j)`, the scorer outputs a scalar score `s_c(x)` for each criterion `c`, and the model preference probability is computed as:

```text
P_model(x_j preferred over x_i for criterion c) = sigmoid(s_c(x_j) - s_c(x_i))
```

The loss is binary cross-entropy between this model preference probability and the GPT-4.1-mini soft preference label. Missing labels and self-comparisons are masked out. Low-confidence labels are filtered with `confidence_threshold = 0.5`, excluding labels close to a tie from the loss.

| Data and labels | Value | Data and labels | Value |
| --- | ---: | --- | ---: |
| Example pairs | 100,000 | Source sample size | 500,000 FineWeb-Edu-Fortified rows |
| Judge model | `gpt-4.1-mini` | Pairwise judgement mode | token log-probabilities |
| Judgement excerpt length | 512 tokens | Model max input length | 2,048 tokens |

| Optimization | Value | Optimization | Value |
| --- | ---: | --- | ---: |
| Total batch size | 512 | Per-device batch size | 8 |
| Learning rate | `5e-5` | Epochs | 2 |
| Warmup ratio | 0.1 | Weight decay | 0.1 |
| Max gradient norm | 1.0 | Label temperature | 1.0 |
| Confidence threshold | 0.5 | Train/validation split | 90% / 10% |

## Intended Use

This model is intended for scoring student-facing literacy materials and generated learner-facing educational responses. In the broader Edu-QuRating project, the student-facing foundational-literacy scorer can be used as a criterion-specific reward component for fine-tuning educational response models.

This model family is separate from the core educational corpus-filtering scorer. It should be used when the target scoring question is about beginner-reader literacy support rather than broad web-corpus educational quality.

## How to Use

### Batch inference pipeline with edu-qurating package

Install package from github: https://github.com/AI-for-Education/edu-qurating.  
Requires `flash-attn>=2.8.3` to run the inference pipline below.

```python
import json

from datasets import load_dataset, Dataset
from qurating.inference import ModelAnnotator, TokenizeAndChunk


model_name = "AI-for-Education/edu-qurating-fl-student-gemma-3-4b"

device_batch_size = 500
text_field = "text"

annotator = ModelAnnotator(
    model_name, labels=None, device_batch_size=device_batch_size
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
```

### Call the model directly with pytorch (no auto chunk handling)

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "AI-for-Education/edu-qurating-fl-student-gemma-3-4b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

text = "I can see a cat. The cat can run. Can you draw the cat?"

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=2048,
)
inputs = {key: value.to(next(model.parameters()).device) for key, value in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    scores = outputs.logits[0].float().cpu()

labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

for label, score in zip(labels, scores):
    print(f"{label}: {score:.4f}")
```

The returned values are criterion-specific scalar logits. They are most useful for ranking, filtering, or comparing texts under the same criterion, rather than as calibrated absolute ratings.

## Limitations

The scores are model-derived quality signals, not ground-truth educational or literacy labels. They should be interpreted as ranking, filtering, or reward signals rather than as definitive assessments. Scores from different Edu-QuRating families are rubric-specific and should not be treated as directly interchangeable.
