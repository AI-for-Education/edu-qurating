""" """

# %%
from datasets import load_dataset
from dotenv import load_dotenv

from qurating.inference import ModelAnnotator, TokenizeAndChunk

load_dotenv(override=True)

LABELS = [
    "education_level",
    "education_level_primary",
    "education_level_secondary",
    "factual_accuracy",
    "lesson_engagement",
    "pedagogical_structure",
]

ANNOTATOR_BATCH_SIZE = 500

# %%
### load texts to score
# you don't have to use load_dataset, you just need to end up with
# a huggingface Dataset or IterableDataset
texts_dataset_name = "YOUR_DATASET_NAME"
texts_dataset = load_dataset(texts_dataset_name)

# set this to the column name of the text in your dataset
text_column = "texts"

# %%
# load annotator model
model = "AI-for-Education/qurater_gemma-3-4b-pt_ds-ours_v2-200000"
annotator_batch_size = ANNOTATOR_BATCH_SIZE
labels = LABELS

annotator = ModelAnnotator(str(model), labels, annotator_batch_size)

tokenizer = TokenizeAndChunk(str(model), text_column, 512)

# %%
# prepare texts datset for annotation
processed_ds = texts_dataset.map(
    tokenizer,
    batched=True,
    remove_columns=[text_column],
    batch_size=4000,
    load_from_cache_file=False,
)

# score texts
results = processed_ds.map(
    annotator,
    batched=True,
    with_indices=True,
    batch_size=annotator_batch_size,
    remove_columns=[col for col in processed_ds.column_names if col != "record_id"],
    load_from_cache_file=False,
)
