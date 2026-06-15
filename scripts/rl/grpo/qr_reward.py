from pathlib import Path
import json
import re
import copy

import numpy as np
from datasets import Dataset
import requests


class QuratingReward:
    def __init__(
        self,
        annotator_batch_size: int = 500,
        text_field: str = "text",
        endpoint: str | None = "http://localhost:8000/score",
    ):
        self._text_field = text_field
        self.endpoint = endpoint
        if self.endpoint is None:
            raise NotImplementedError

    def reward_fun_generator(
        self,
        score_spec,
        name: str,
        score_cap: tuple[float, float] = (-np.inf, 12.0),
        grouper: re.Pattern | None = None,
        group_idx: int | None = None,
        verbose: bool = False,
    ):
        if grouper is not None and group_idx is None:
            raise ValueError("Must provide group_idx if grouper is provided")

        def reward_fun(prompts, completions, **kwargs):
            if verbose:
                responses = [completion[0]["content"] for completion in completions]
                q = prompts[0][-1]["content"]
                print("-" * 20, f"Question:\n{q}", f"\nResponse:\n{responses[0]}")
            if grouper is not None and group_idx is not None:
                use_completions = []
                for completion in completions:
                    # we don't apply the reward if the formatting isn't correct
                    # (empty string will lead to zero reward)
                    use_completions.append([{"content": ""}])
                    match = grouper.match(completion[0]["content"])
                    if match is not None:
                        groups = match.groups()
                        if len(groups) > group_idx:
                            use_completions[-1][0]["content"] = groups[group_idx]
            else:
                use_completions = completions

            return self.reward(prompts, use_completions, score_spec, score_cap, **kwargs)

        reward_fun.__name__ = name
        return reward_fun

    def reward(
        self,
        prompts: list[list[dict[str, str]]],
        completions: list[list[dict[str, str]]],
        score_spec: dict[str, dict[str | tuple, float | int]],
        score_cap: tuple[float, float],
        **kwargs,
    ):
        responses = [completion[0]["content"] for completion in completions]
        ds = Dataset.from_list([{"text": resp} for resp in responses])
        scores = self.score(ds, model_types=list(score_spec))
        score_holders = []
        for model_type, weight_dict in score_spec.items():
            for label, weight in weight_dict.items():
                if isinstance(label, str):
                    # normal behaviour
                    # - we take the score from the same dimension for all rows
                    label_score_vec = np.array(scores[model_type][f"{label}_average"])
                elif isinstance(label, tuple):
                    # extended behavior
                    # - we match individual rows to their correswponding scoring dimension
                    #   based on the row_matcher specification
                    # we initialise with all nans so that the score vectors are all the same shape
                    # but we use nanmean to ignore non-matching row-score_dimension combinations
                    label_score_vec = np.full(len(responses), fill_value=np.nan)
                    label, row_matcher = label
                    for column, value in row_matcher:
                        if column not in kwargs:
                            raise ValueError(
                                f"matcher column {column} is not in dataset"
                            )
                        filt = [
                            i for i, row in enumerate(kwargs[column]) if row == value
                        ]
                        label_score_vec[filt] = np.array(
                            scores[model_type][f"{label}_average"]
                        )[filt]
                else:
                    raise ValueError("label must be a string or a tuple")
                capped_score_vec = weight * np.maximum(
                    np.minimum(label_score_vec, score_cap[1]), score_cap[0]
                )
                score_holders.append(capped_score_vec)
        scores_arr = np.nanmean(score_holders, axis=0)
        return scores_arr.tolist()

    def score(self, dataset: Dataset, model_types: str | list[str] | None = None):
        if model_types is None:
            model_types = list(self.models)
        results = {
            model_type: self._score_model(dataset, model_type)
            for model_type in model_types
        }

        return results

    def _score_model(self, dataset: Dataset, model_type: str):
        if self.endpoint is None:
            raise NotImplementedError
            annotator, tokenizer = (
                self.annotators[model_type],
                self.tokenizers[model_type],
            )
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
        else:
            ds_dict = dataset.to_dict()
            assert isinstance(ds_dict, dict)
            body = {
                "texts": ds_dict[self.text_field],
                "model_type": model_type,
            }
            # handle empty strings (will raise error when trying to score)
            # first identify them, then remove them
            # their score will be reinserted as zero at the end
            not_empty_idx = [i for i, text in enumerate(body["texts"]) if len(text) > 0]
            if len(not_empty_idx) < len(dataset):
                body["texts"] = [
                    text for i, text in enumerate(body["texts"]) if i in not_empty_idx
                ]
            response = requests.post(self.endpoint, json=body)
            assert response.status_code == 200
            results = json.loads(response.content.decode("utf-8"))["scores"]
            # insert zero scores for empty strings (if there are any)
            if len(not_empty_idx) < len(dataset):
                filled_results = {}
                for key, scores in results.items():
                    # only take average scores
                    # (chunk scores are ignorted later and cause numpy error here due to being lists of lists)
                    if key.endswith("_average"):
                        empty_holder = np.zeros(len(dataset))
                        ## if all are empty then don't try to set, just keep all zeros
                        if not_empty_idx:
                            empty_holder[not_empty_idx] = scores
                        filled_results[key] = empty_holder.tolist()
                results = filled_results
        return results

    @property
    def text_field(self):
        return self._text_field

    @text_field.setter
    def text_field(self, value: str):
        for tokenizer in self.tokenizers.values():
            tokenizer.text_field = value
        self._text_field = value

    @staticmethod
    def _load_dataset(dataset_file):
        parquetf = Path(dataset_file).with_suffix(".parquet")
        if parquetf.exists():
            ds = Dataset.from_parquet(str(dataset_file))
        elif Path(dataset_file).is_dir():
            ds = Dataset.load_from_disk(dataset_file)
        else:
            raise ValueError(f"{dataset_file} doesn't exist or is not a valid format")
        if not isinstance(ds, Dataset):
            raise ValueError(f"dataset file must return a {type(Dataset)} object")

        return ds

    def _load_labels(self, dataset_file):
        ds = self._load_dataset(dataset_file)
        labels = [
            "_".join(col.split("_")[:-1])
            for col in ds.column_names
            if col.endswith("_average")
        ]
        return labels


#######################################################################################
#######################################################################################


class CorrectnessReward:
    def __init__(self, endpoint: str = "localhost:7654"):
        pass
