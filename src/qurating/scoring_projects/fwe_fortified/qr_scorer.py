from pathlib import Path

from datasets import Dataset

from ...inference import ModelAnnotator, TokenizeAndChunk
from .model_manager import QuratingModelManager


class QuratingScorer:
    def __init__(
        self,
        base_model: str = "gemma-3-4b",
        model_types: str | list[str] | None = None,
        annotator_batch_size: int = 500,
        text_field: str = "text",
    ):
        self._text_field = text_field
        self.base_model = base_model

        self.annotator_batch_size = annotator_batch_size

        self.modman = QuratingModelManager()
        self.modman.download_all(base_models=base_model)

        if model_types is None:
            model_types = list(self.modman.model_types(base_model))
        elif isinstance(model_types, str):
            model_types = [model_types]
        if not all(
            model_type in self.modman.model_types(base_model)
            for model_type in model_types
        ):
            raise ValueError(
                f"model_types contains unsupported values. Must be in {list(self.modman.model_types(base_model))}"
            )
        self.annotators = {
            model_type: ModelAnnotator(
                self.modman.model_folder(base_model, model_type, local_path=True),
                None,
                self.annotator_batch_size,
            )
            for model_type in self.modman.model_types(base_model)
        }

        self.tokenizers = {
            model_type: TokenizeAndChunk(
                self.modman.model_folder(base_model, model_type, local_path=True),
                text_field,
                512,
            )
            for model_type in self.modman.model_types(base_model)
        }

    def score(self, dataset: Dataset, model_types: str | list[str] | None = None):
        if model_types is None:
            model_types = list(self.modman.model_types(self.base_model))
        results = {
            model_type: self._score_model(dataset, model_type)
            for model_type in model_types
        }

        return results

    def _score_model(self, dataset: Dataset, model_type: str):
        annotator, tokenizer = self.annotators[model_type], self.tokenizers[model_type]
        processed_ds = dataset.map(
            tokenizer,
            batched=True,
            remove_columns=[self.text_field],
            batch_size=4000,
            load_from_cache_file=True,
        )
        results = processed_ds.map(
            annotator,
            batched=True,
            with_indices=True,
            batch_size=self.annotator_batch_size,
            remove_columns=[col for col in processed_ds.column_names],
            load_from_cache_file=True,
        )
        return results

    @property
    def text_field(self):
        return self._text_field

    @property
    def labels(self):
        return self.modman.labels

    @text_field.setter
    def text_field(self, value: str):
        for tokenizer in self.tokenizers.values():
            tokenizer.text_field = value
        self._text_field = value
