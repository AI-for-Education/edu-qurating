# Edu-QuRating Model Cards

This folder keeps the source-of-truth Hugging Face model cards for the Edu-QuRating and legacy QuRater scorer models.

These files are intended to be reviewed here first, then copied as `README.md` into the corresponding Hugging Face model repository when the model files are ready.

## Current Model Cards

| Model card | Intended Hugging Face repo | Rubric | Backbone |
| --- | --- | --- | --- |
| `model_cards/qurating-core-ed-gemma-3-4b.md` | `AI-for-Education/qurating-core-ed-gemma-3-4b` | Core educational quality | Gemma-3-4B-PT |
| `model_cards/qurating-core-ed-qwen-3-4b.md` | `AI-for-Education/qurating-core-ed-qwen-3-4b` | Core educational quality | Qwen3-Reranker-4B |
| `model_cards/qurating-fl-student-gemma-3-4b.md` | `AI-for-Education/qurating-fl-student-gemma-3-4b` | Foundational literacy, student-facing | Gemma-3-4B-PT |
| `model_cards/qurating-fl-student-qwen-3-4b.md` | `AI-for-Education/qurating-fl-student-qwen-3-4b` | Foundational literacy, student-facing | Qwen3-Reranker-4B |
| `model_cards/qurating-fl-teacher-gemma-3-4b.md` | `AI-for-Education/qurating-fl-teacher-gemma-3-4b` | Foundational literacy, teacher-facing | Gemma-3-4B-PT |
| `model_cards/qurating-fl-teacher-qwen-3-4b.md` | `AI-for-Education/qurating-fl-teacher-qwen-3-4b` | Foundational literacy, teacher-facing | Qwen3-Reranker-4B |
| `model_cards/qurater_gemma-3-4b-pt_ds-ours_v2-200000.md` | `AI-for-Education/qurater_gemma-3-4b-pt_ds-ours_v2-200000` | Core educational quality | Gemma-3-4B-PT |
| `model_cards/qurater_Qwen3-Reranker-4B-seq-cls_ds-ours_v2-200000.md` | `AI-for-Education/qurater_Qwen3-Reranker-4B-seq-cls_ds-ours_v2-200000` | Core educational quality | Qwen3-Reranker-4B |

## Review Checklist Before Upload

- Confirm the final Hugging Face repository names in `repo_mapping.yaml`.
- Confirm each model card title, repo id, rubric, and backbone.
- Confirm each loading example uses `AutoModelForSequenceClassification`.
- Confirm each model card describes the model as a scoring/ranking model, not a generative model.
- Confirm the score labels match the uploaded model config.
- Confirm Gemma model `model.safetensors.index.json` metadata reports the correct parameter count before uploading weights.
- Confirm whether the two `qurater_*` repositories should remain as legacy repos or be replaced by the corresponding `edu-qurating-core-ed-*` repositories.
- Copy the selected model card to the target model repository as `README.md`.

## Notes

The Hugging Face repository privacy setting should be controlled by the repository itself. These model cards therefore do not include `private: true` in their YAML metadata.

The known Gemma parameter-count issue should be fixed in the model artifacts before final upload. The README alone cannot correct Hugging Face's displayed parameter count if the uploaded `model.safetensors.index.json` metadata is wrong.
