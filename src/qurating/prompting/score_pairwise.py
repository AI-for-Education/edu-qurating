import argparse
import asyncio
from pathlib import Path
import pickle

from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer
import numpy as np
from fdllm import get_caller

from .llm_util import (
    query_model,
    aquery_model,
    query_model_logprobs,
    aquery_model_logprobs,
)
from ..constants import LOG_DIR, CACHE_DIR


class Comparator:
    @staticmethod
    def add_args(parser):
        parser.add_argument("-k", "--num_compare_all", type=int, default=2)
        parser.add_argument("-n", "--num_examples", type=int, default=100)
        parser.add_argument("--offset", type=int, default=0)
        parser.add_argument("--num_examples_proportion", type=float, default=None)
        parser.add_argument("--num_examples_proportion_start", type=float, default=0.0)

        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument(
            "--model",
            type=str,
            default="gpt-3.5-turbo",
            # choices=["gpt-3.5-turbo", "gpt-4", "claude-2"],
        )

        parser.add_argument("-g", "--generations", type=int, default=20)

        parser.add_argument("--template", type=str, default="")
        parser.add_argument(
            "--template_file",
            type=str,
            default="annotate/templates/pairwise_default.txt",
        )
        parser.add_argument("--labels", type=str, nargs=2, default=["A", "B"])

        parser.add_argument("--text_field", type=str, default="text")
        parser.add_argument("--token_field", type=str, default="input_ids")
        parser.add_argument("--tokens_min", type=int, default=256)
        parser.add_argument("--tokens_max", type=int, default=512)
        parser.add_argument("--probability_tokens_max", type=float, default=0.5)
        parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf")

        parser.add_argument(
            "--system_prompt", type=str, default="You are a helpful assistant."
        )

        parser.add_argument("--flat_output_format", action="store_true")
        parser.add_argument("--cache_dir", type=str, default=str(CACHE_DIR))
        parser.add_argument("--max-concurrency", type=int, default=20)
        parser.add_argument("--logprobs", action="store_true")

    def __init__(self, args):
        self.args = args
        if self.args.template_file:
            with open(self.args.template_file) as f:
                self.args.template = f.read()

        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

        self.offset = 0
        self.num_examples = 0
        self.semaphore = asyncio.Semaphore(args.max_concurrency)

        cache_logprobs_suffix = "_use-logprobs" if args.logprobs else ""
        self.cache_dir = (
            Path(args.cache_dir)
            / f"{Path(args.template_file).stem}_{args.model}_{args.tokens_max}_{args.num_examples}{cache_logprobs_suffix}"
        )
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def __getstate__(self):
        return self.args

    def __setstate__(self, state):
        self.__init__(state)

    def extract_excerpt(self, text, index, num_tokens, token_ids=None):
        if token_ids is None:
            # heuristic for faster tokenization
            max_character_length = self.args.tokens_max * 40
            if len(text) > max_character_length:
                np.random.seed(self.args.seed + index + self.offset + 1)
                start_pos = np.random.randint(0, len(text) - max_character_length + 1)
                text = text[start_pos : start_pos + max_character_length]

            token_ids = self.tokenizer(
                text, truncation=False, padding=False, add_special_tokens=False
            ).input_ids

        if len(token_ids) <= self.args.tokens_max:
            return text

        np.random.seed(self.args.seed + index + self.offset)
        start_pos = np.random.randint(0, len(token_ids) - self.args.tokens_max + 1)
        token_ids = token_ids[start_pos : start_pos + num_tokens]
        return self.tokenizer.decode(token_ids)

    def parse_generations(self, generations):
        for generation in generations:
            if generation is None:
                yield None
            elif generation == self.args.labels[0]:
                yield 0
            elif generation == self.args.labels[1]:
                yield 1

    def sample_num_tokens(self, indices):
        np.random.seed(self.args.seed + sum(indices) + self.offset)
        # use length tokens_max with probability probability_tokens_max, otherwise sample uniformly from [tokens_min, tokens_max]
        if (
            self.args.probability_tokens_max > 0
            and np.random.rand() < self.args.probability_tokens_max
        ):
            return self.args.tokens_max
        else:
            return np.random.randint(self.args.tokens_min, self.args.tokens_max + 1)

    async def __acall__(self, examples, indices):
        cache_hit, pre_call_output = self._pre_call(examples, indices)
        if cache_hit:
            cached_data = pre_call_output
            if not self.args.flat_output_format:
                return {
                    **cached_data,
                    "examples": [examples],
                }
            else:
                return cached_data
        else:
            texts, caller, num_tokens, n, votes_a, votes_b, predictions = (
                pre_call_output
            )
        probs = np.full((2, n, n), fill_value=-100.0)
        async with self.semaphore:
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue

                    prompt = self.args.template.format(
                        text_a=texts[i],
                        text_b=texts[j],
                        label_a=self.args.labels[0],
                        label_b=self.args.labels[1],
                    )

                    if self.args.logprobs:
                        out_probs = await aquery_model_logprobs(
                            prompt,
                            caller,
                            system_prompt=self.args.system_prompt,
                            log_file_path=str(LOG_DIR / "api_cost.jsonl"),
                            retries=3,
                        )
                        probs[:, i, j] = np.array(out_probs)
                    else:
                        ######### otherwise estimate probablity by counting over multiple runs
                        generations = await aquery_model(
                            prompt,
                            caller,
                            system_prompt=self.args.system_prompt,
                            generations=self.args.generations,
                            log_file_path=str(LOG_DIR / "api_cost.jsonl"),
                            retries=0,
                        )

                        for vote in self.parse_generations(generations):
                            if vote == 0:
                                votes_a[i, j] += 1
                            elif vote == 1:
                                votes_b[i, j] += 1

        return self._post_call(
            examples, indices, texts, n, votes_a, votes_b, predictions, probs
        )

    def __call__(self, examples, indices):
        cache_hit, pre_call_output = self._pre_call(examples, indices)
        if cache_hit:
            cached_data = pre_call_output
            if not self.args.flat_output_format:
                return {
                    **cached_data,
                    "examples": [examples],
                }
            else:
                return cached_data
        else:
            texts, caller, num_tokens, n, votes_a, votes_b, predictions = (
                pre_call_output
            )

        probs = np.full((2, n, n), fill_value=-100.0)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                prompt = self.args.template.format(
                    text_a=texts[i],
                    text_b=texts[j],
                    label_a=self.args.labels[0],
                    label_b=self.args.labels[1],
                )
                if self.args.logprobs:
                    out_probs = query_model_logprobs(
                        prompt,
                        caller,
                        system_prompt=self.args.system_prompt,
                        log_file_path=str(LOG_DIR / "api_cost.jsonl"),
                        retries=3,
                    )
                    probs[:, i, j] = np.array(out_probs)
                else:
                    generations = query_model(
                        prompt,
                        caller,
                        system_prompt=self.args.system_prompt,
                        generations=self.args.generations,
                    )

                    for vote in self.parse_generations(generations):
                        if vote == 0:
                            votes_a[i, j] += 1
                        elif vote == 1:
                            votes_b[i, j] += 1
        return self._post_call(
            examples, indices, texts, n, votes_a, votes_b, predictions, probs
        )

    def _pre_call(self, examples, indices):
        cache_file = (
            self.cache_dir / f"cache_{'-'.join(str(idx) for idx in indices)}.pkl"
        )
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
            probs = cached_data["average"]
            n_nodata = int((np.array(probs) < 0).sum())
            if n_nodata == len(probs[0]):
                return True, cached_data

        num_tokens = self.sample_num_tokens(indices)

        if self.args.token_field in examples:
            texts = [
                self.extract_excerpt(text, index, num_tokens, token_ids)
                for text, token_ids, index in zip(
                    examples[self.args.text_field],
                    examples[self.args.token_field],
                    indices,
                )
            ]
        else:
            texts = [
                self.extract_excerpt(text, index, num_tokens)
                for text, index in zip(examples[self.args.text_field], indices)
            ]

        n = len(texts)
        votes_a = np.zeros((n, n), dtype=np.int32)
        votes_b = np.zeros((n, n), dtype=np.int32)
        predictions = np.full((n, n), -100, dtype=np.float32)
        caller = get_caller(self.args.model)
        # caller = self.args.model

        return False, (texts, caller, num_tokens, n, votes_a, votes_b, predictions)

    def _post_call(
        self, examples, indices, texts, n, votes_a, votes_b, predictions, probs
    ):
        if self.args.logprobs:
            calibrated_predictions = np.full((n, n), fill_value=-100.0)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        calibrated_predictions[i, j] = (
                            probs[1, i, j] + probs[0, j, i]
                        ) / 2

        else:
            np.divide(
                votes_b,
                votes_a + votes_b,
                out=predictions,
                where=votes_a + votes_b > self.args.generations // 2,
            )
            calibrated_predictions = np.where(
                (predictions != -100) & (predictions.T != -100),
                (predictions + (1 - predictions.T)) / 2,
                -100,
            )

        if not self.args.flat_output_format:
            out = {
                "examples": [examples],
                "indices": [indices],
                "texts": [texts],
                "votes_a": [votes_a.tolist()],
                "votes_b": [votes_b.tolist()],
                "average": [calibrated_predictions.tolist()],
            }
        else:
            indices_a, indices_b = np.where(np.triu(np.ones((n, n)), k=1))
            out = {
                "index_a": indices_a,
                "index_b": indices_b,
                "texts_a": [texts[i] for i in indices_a],
                "texts_b": [texts[j] for j in indices_b],
                "comparisons_forward": predictions[indices_a, indices_b].tolist(),
                "comparisons_backward": (
                    1 - predictions[indices_b, indices_a]
                ).tolist(),
                "comparisons_avg": calibrated_predictions[
                    indices_a, indices_b
                ].tolist(),
            }
        cache_file = (
            self.cache_dir / f"cache_{'-'.join(str(idx) for idx in indices)}.pkl"
        )
        cache_out = {key: val for key, val in out.items() if key not in ["examples"]}
        with open(cache_file, "wb") as f:
            pickle.dump(cache_out, f)
        return out

    def apply(self, dataset):
        if self.args.num_examples_proportion is not None:
            offset = int(len(dataset) * self.args.num_examples_proportion_start)
            num_examples = int(len(dataset) * self.args.num_examples_proportion)
        else:
            offset = self.args.offset
            num_examples = self.args.num_examples
        self.offset = (offset // self.args.num_compare_all) * self.args.num_compare_all
        self.num_examples = (
            num_examples // self.args.num_compare_all
        ) * self.args.num_compare_all

        print("Example offset", self.offset)
        print(
            "Total number of pairwise comparisons:",
            self.num_examples * (self.args.num_compare_all - 1),
        )

        return dataset.select(range(self.offset, self.offset + self.num_examples)).map(
            self.__acall__,
            with_indices=True,
            batched=True,
            batch_size=self.args.num_compare_all,
            remove_columns=dataset.column_names,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)

    parser.add_argument("--json", action="store_true")

    Comparator.add_args(parser)

    args = parser.parse_args()
    print(args)

    if args.json:
        dataset = load_dataset(
            "json",
            data_files=[args.input],
            split="train",
            download_mode="force_redownload",
        )
    else:
        dataset = load_from_disk(args.input)

    dataset = Comparator(args).apply(dataset)

    print(f"Saving to {args.output}")
    dataset.save_to_disk(args.output)
