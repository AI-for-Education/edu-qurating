from pathlib import Path
import json
import re
import copy

import numpy as np
from datasets import Dataset
import requests

from qurating.constants import RESULTS_DIR, ROOT
from qurating.inference import ModelAnnotator, TokenizeAndChunk


class QuratingReward:
    def __init__(
        self,
        annotator_batch_size: int = 500,
        text_field: str = "text",
        endpoint: str | None = "http://localhost:8000/score",
    ):
        self._text_field = text_field
        self.endpoint = endpoint
        self.dataset_files = {
            "core_ed": (
                RESULTS_DIR
                / "tokens_max_512"
                / "fwe-fortified_sampled-500000_seed-72353534"
                / "ours_v2"
                / "combined_gpt-4.1-mini_nexamples-200000_use-logprobs"
            ),
            "fl_student": (
                RESULTS_DIR
                / "tokens_max_512"
                / "fwe-fortified_sampled-primary-5-pedagogical-5-500000_seed-274634520"
                / "FLN_student-facing"
                / "combined_gpt-4.1-mini_nexamples-200000_use-logprobs"
            ),
            "fl_teacher": (
                RESULTS_DIR
                / "tokens_max_512"
                / "fwe-fortified_sampled-pedagogical-5-500000_seed-274634520"
                / "FLN_teacher-facing"
                / "combined_gpt-4.1-mini_nexamples-200000_use-logprobs"
            ),
        }

        self.labels = {
            model_type: self._load_labels(dataset_file)
            for model_type, dataset_file in self.dataset_files.items()
        }

        if self.endpoint is None:
            self.models = {
                "core_ed": "AI-for-Education/qurater_gemma-3-4b-pt_ds-ours_v2-200000",
                "fl_student": str(
                    ROOT
                    / "checkpoints-preferences"
                    / "qurater_gemma-3-4b-pt_bsz512_lr5e-5_epochs2_warmup0.1_conf0.5_labeltemp1.0_ds-fwe-fortified_sampled-primary-5-pedagogical-5-FLN_student-facing-500000-200000-512-274634520-gpt-4.1-mini-logprobs"
                    / "checkpoint-340"
                ),
                "fl_teacher": str(
                    ROOT
                    / "checkpoints-preferences"
                    / "qurater_gemma-3-4b-pt_bsz512_lr5e-5_epochs2_warmup0.1_conf0.5_labeltemp1.0_ds-fwe-fortified_sampled-pedagogical-5-FLN_teacher-facing-500000-200000-512-274634520-gpt-4.1-mini-logprobs"
                    / "checkpoint-312"
                ),
            }
            assert set(self.models) == set(self.dataset_files)

            self.annotator_batch_size = annotator_batch_size

            self.annotators = {
                model_type: ModelAnnotator(
                    str(model), self.labels[model_type], self.annotator_batch_size
                )
                for model_type, model in self.models.items()
            }

            self.tokenizers = {
                model_type: TokenizeAndChunk(str(model), text_field, 512)
                for model_type, model in self.models.items()
            }

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

            return self.reward(prompts, use_completions, score_spec, score_cap)

        reward_fun.__name__ = name
        return reward_fun

    def reward(
        self,
        prompts: list[list[dict[str, str]]],
        completions: list[list[dict[str, str]]],
        score_spec: dict[str, dict[str | tuple, float | int]],
        score_cap: tuple[float, float],
    ):
        responses = [completion[0]["content"] for completion in completions]
        ds = Dataset.from_list([{"text": resp} for resp in responses])
        scores = self.score(ds, model_types=list(score_spec))
        score_holders = []
        for model_type, weight_dict in score_spec.items():
            for label, weight in weight_dict.items():
                if isinstance(label, str):
                    label_score_vec = np.array(scores[model_type][f"{label}_average"])
                elif isinstance(label, tuple):
                    label, row_matcher = label
                else:
                    raise ValueError("label must be a string or a tuple")
                capped_score_vec = weight * np.maximum(
                    np.minimum(label_score_vec, score_cap[1]), score_cap[0]
                )
                score_holders.append(capped_score_vec)
        scores_arr = np.mean(score_holders, axis=0)
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
