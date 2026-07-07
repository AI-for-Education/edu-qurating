---
license: other
library_name: transformers
pipeline_tag: text-classification
tags:
- edu-qurating
- qurating
- education
- foundational-literacy
- text-scoring
- sequence-classification
- reward-model
- gemma
---

# Edu-QuRating Foundational Literacy Teacher-Facing Scorer (Gemma 3 4B)

Model repository: `AI-for-Education/qurating-fl-teacher-gemma-3-4b`

Backbone: Gemma-3-4B-PT

This model is an Edu-QuRating sequence-classification scorer for teacher-facing foundational-literacy text. It takes a text passage as input and outputs one scalar logit for each teacher-facing literacy criterion. It is intended for ranking, filtering, corpus curation, and reward-modeling experiments, not for text generation.

## Scores

| Score | Meaning |
| --- | --- |
| `1_oral_language_vocabulary` | Guidance for teaching oral language or vocabulary. Higher scores indicate explicit instruction for word meanings, word parts, usage, sentence use, or discussion that builds language comprehension. |
| `2_phonological_awareness` | Guidance for teaching phonological or phonemic awareness in spoken language. Higher scores indicate oral activities such as rhyming, blending, segmenting, identifying sounds, or manipulating syllables, onsets, and rimes. |
| `3_systematic_phonics` | Guidance for systematic and explicit phonics instruction. Higher scores indicate clear sequencing, explicit modelling of letter-sound correspondences, decoding, blending, sounding out, CVC words, digraphs, or other actionable phonics steps. |
| `4_reading_fluency` | Guidance for building reading fluency. Higher scores indicate guided oral reading practice, echo reading, choral reading, rereading, tracking words, modelling expression, pacing, or monitoring accuracy. |
| `5_reading_comprehension` | Guidance for teaching reading comprehension strategies. Higher scores indicate explicit modelling or guided practice for predicting, summarising, inferring, retelling, main idea, guided questioning, or background knowledge. |
| `6_writing_encoding` | Guidance for integrating writing and spelling to support reading. Higher scores indicate encoding, dictation, segmenting to spell, handwriting, sentence writing, or links between spelling patterns and phonics lessons. |
| `7_pedagogical_quality` | Overall pedagogical structure and responsiveness. Higher scores indicate scaffolding, gradual release of responsibility, formative checking, corrective feedback, multilingual alignment, and clear lesson sequencing. |

The scores are rubric-specific model outputs. They are intended to support ranking, filtering, or reward construction.

## Training Procedure

This model was trained as a pairwise-preference distillation model. The training data starts from a 500k-row FineWeb-Edu-Fortified sample and uses the teacher-facing foundational-literacy prompt set to construct 200k pairwise comparisons. The pairwise judgement excerpts are capped at 512 tokens.

Pairwise labels were generated with GPT-4.1-mini using token log-probabilities. For each criterion-specific prompt, the judge is asked to choose between two excerpts. The log-probabilities of the two answer labels are converted into soft preference probabilities. To reduce order effects, each pair is judged in both text orders, and the two directional probabilities are averaged into one calibrated soft preference label.

For a pair of texts `(x_i, x_j)`, the scorer outputs a scalar score `s_c(x)` for each criterion `c`, and the model preference probability is computed as:

```text
P_model(x_j preferred over x_i for criterion c) = sigmoid(s_c(x_j) - s_c(x_i))
```

The loss is binary cross-entropy between this model preference probability and the GPT-4.1-mini soft preference label. Missing labels and self-comparisons are masked out. Low-confidence labels are filtered with `confidence_threshold = 0.5`, excluding labels close to a tie from the loss.

| Data and labels | Value | Data and labels | Value |
| --- | ---: | --- | ---: |
| Pairwise examples | 200,000 | Source sample size | 500,000 FineWeb-Edu-Fortified rows |
| Source sample seed | 274,634,520 | Pairwise prompt set | `FLN_teacher-facing` |
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

This model is intended for scoring teacher-facing literacy materials, instructional guidance, and generated teacher-support responses. In the broader Edu-QuRating project, the teacher-facing foundational-literacy scorer can be used as a criterion-specific reward component for fine-tuning educational response models.

This model family is separate from the core educational corpus-filtering scorer. It should be used when the target scoring question is about literacy instruction and teacher guidance rather than broad web-corpus educational quality.

## How to Use

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "AI-for-Education/qurating-fl-teacher-gemma-3-4b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

text = "First, model how to blend /c/ /a/ /t/. Then ask pupils to sound out the word with you and write it in their books."

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
