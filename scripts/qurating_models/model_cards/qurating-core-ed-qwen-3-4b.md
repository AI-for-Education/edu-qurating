---
license: other
library_name: transformers
pipeline_tag: text-classification
tags:
- edu-qurating
- qurating
- education
- text-scoring
- sequence-classification
- reward-model
- qwen
---

# Edu-QuRating Core Educational Scorer (Qwen3 4B)

Model repository: `AI-for-Education/qurating-core-ed-qwen-3-4b`

Backbone: Qwen3-Reranker-4B

This model is an Edu-QuRating sequence-classification scorer for educational text. It takes a text passage as input and outputs one scalar logit for each criterion in the core educational-quality rubric. It is intended for ranking, filtering, corpus curation, and reward-modeling experiments, not for text generation.

## Scores

The model outputs six criterion-specific scores. Each score follows the corresponding pairwise judgement prompt: a higher score means the text is closer to the side that the judge would prefer for that criterion. Scores should be interpreted criterion by criterion, not as one general educational-quality label.

| Score | Meaning |
| --- | --- |
| `education_level` | Relative level of assumed prior knowledge. Higher scores indicate text that assumes more background knowledge, uses more abstract ideas, introduces specialized terminology with less introductory scaffolding, or relies more on symbolic representations such as equations or diagrams. This is a level signal, not a quality score. |
| `education_level_primary` | Appropriateness for a primary or elementary-school educational context, roughly ages 5-11. Higher scores indicate text that is more suitable for younger or earlier-stage learners. |
| `education_level_secondary` | Appropriateness for a secondary or high-school educational context, roughly ages 12-18. Higher scores indicate text that is more suitable for older or more advanced school-level learners. |
| `factual_accuracy` | Factual reliability of the excerpt. Higher scores indicate more correct, precise, and internally consistent statements, with fewer errors, contradictions, conflicts with well-established knowledge, or unsupported unverifiable claims. |
| `lesson_engagement` | Likelihood that the text captures and sustains learner interest. Higher scores indicate stronger use of relatable examples, attention hooks, interaction prompts, learner-facing tone, or vivid imagery that supports understanding. |
| `pedagogical_structure` | Clarity of pedagogical structure for the intended learners. Higher scores indicate clearer sequencing and signposting, accessible vocabulary, concrete-to-abstract progression, aligned examples or worked steps, consistent terminology or notation, and less irrelevant detail. |

## Training Procedure

This model was trained as a pairwise-preference distillation model, not as an absolute-rating regressor. The training data starts from a 500k-row FineWeb-Edu-Fortified sample and uses the `ours_v2` core educational prompt set to construct 200k pairwise comparisons. The pairwise judgement excerpts are capped at 512 tokens.

Pairwise labels were generated with GPT-4.1-mini using token log-probabilities. For each criterion-specific prompt, the judge is asked to choose between two excerpts. The log-probabilities of the two answer labels are converted into soft preference probabilities. To reduce order effects, each pair is judged in both text orders, and the two directional probabilities are averaged into one calibrated soft preference label.

For a pair of texts `(x_i, x_j)`, the scorer outputs a scalar score `s_c(x)` for each criterion `c`, and the model preference probability is computed as:

```text
P_model(x_j preferred over x_i for criterion c) = sigmoid(s_c(x_j) - s_c(x_i))
```

The loss is binary cross-entropy between this model preference probability and the GPT-4.1-mini soft preference label. Missing labels and self-comparisons are masked out. Low-confidence labels are filtered with `confidence_threshold = 0.5`, excluding labels close to a tie from the loss.

| Data and labels | Value | Data and labels | Value |
| --- | ---: | --- | ---: |
| Pairwise examples | 200,000 | Source sample size | 500,000 FineWeb-Edu-Fortified rows |
| Source sample seed | 72,353,534 | Pairwise prompt set | `ours_v2` |
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

This model is intended for educational-quality scoring and corpus curation. In the Edu-QuRating pipeline, the core educational scorer is applied to FineWeb-Edu-Fortified documents to produce document-level score columns. These scores can then be used for multi-dimensional filtering, for example selecting documents that score highly on factual accuracy, lesson engagement, and pedagogical structure.

The model can also be used as a reward component for generated educational responses, although the core corpus-filtering use case is document scoring.

For long documents, the project scoring pipeline tokenizes the full document, splits it into 512-token chunks, scores each chunk, and averages chunk scores back to document-level scores using token-count weights.

## How to Use

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "AI-for-Education/qurating-core-ed-qwen-3-4b"

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

The scores are model-derived quality signals, not ground-truth educational labels. They should be interpreted as ranking, filtering, or reward signals rather than as definitive assessments of educational value. Performance may vary across domains, age levels, languages, and formats that differ from the training and validation data.
