import torch
import numpy as np
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoConfig
from qurating.modeling.model_factory import create_model


class TokenizeAndChunk:
    """Tokenizes and chunks text data for model input."""

    def __init__(
        self, tokenizer_name: str, text_field: str = "text", tokens: int = 512
    ):
        self.tokens = tokens
        self.tokenizer_name = tokenizer_name
        self.text_field = text_field

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.tokenizer.pad_token_id = 0

    def tokenize_and_chunk(self, source_tokens: List[List[int]]) -> tuple:
        """Split token sequences into chunks of specified length."""
        chunks_token_ids = []
        chunks_token_counts = []

        for seq in source_tokens:
            chunks = torch.tensor(seq, dtype=torch.long).split(self.tokens)
            chunks_token_ids.append([chunk.tolist() for chunk in chunks])
            chunks_token_counts.append([len(x) for x in chunks])

        return chunks_token_ids, chunks_token_counts

    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Process a batch of examples."""
        # Tokenize texts
        source_tokens = self.tokenizer(
            example[self.text_field],
            truncation=False,
            padding=False,
            add_special_tokens=False,
        ).input_ids

        chunks_token_ids, chunks_token_counts = self.tokenize_and_chunk(source_tokens)

        return {
            "chunks_token_ids": chunks_token_ids,
            "chunks_token_counts": chunks_token_counts,
        }


class ModelAnnotator:
    def __init__(self, model_name: str, labels: list[str] | None, device_batch_size: int):
        self.model_name = model_name
        self.device_batch_size = device_batch_size
        config = AutoConfig.from_pretrained(model_name)
        if labels is None:
            if config.label2id is not None:
                labels = list(config.label2id)
            else:
                raise ValueError("labels can't be None if config has no labels")
        self.labels = labels
        config.num_labels = len(labels)

        self.model = create_model(
            model_name,
            num_labels=len(labels),
            reinit_score_layer=False,
            config=config,
            torch_dtype=torch.bfloat16,
        )
        self.model.config.pad_token_id = 0
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device {self.device}")
        self.model.to(self.device)

        self.num_labels = len(labels)
        assert self.num_labels == self.model.config.num_labels, (
            f"Number of labels ({self.num_labels}) does not match model config ({self.model.config.num_labels})"
        )

    def __getstate__(self):
        return {
            "model_name": self.model_name,
            "labels": self.labels,
            "device_batch_size": self.device_batch_size,
        }

    def __setstate__(self, state):
        self.__init__(**state)

    @torch.inference_mode()
    def score_chunks(
        self, chunks_token_ids: List[torch.Tensor], chunks_token_counts: torch.Tensor
    ) -> torch.Tensor:
        """Score text chunks with the model."""
        sorted_indices = torch.argsort(chunks_token_counts)
        scores = torch.zeros(
            len(chunks_token_ids), self.num_labels, dtype=torch.float32
        )

        for batch_indices in sorted_indices.split(self.device_batch_size):
            max_len = chunks_token_counts[batch_indices].max()

            input_ids = torch.zeros((len(batch_indices), max_len), dtype=torch.long)
            attention_mask = torch.zeros(
                (len(batch_indices), max_len), dtype=torch.long
            )

            for i, j in enumerate(batch_indices):
                seq = chunks_token_ids[j]
                input_ids[i, : len(seq)] = seq
                attention_mask[i, : len(seq)] = 1

            outputs = self.model(
                input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                use_cache=False,
            )
            scores[batch_indices] = outputs.logits.float().cpu()

        return scores

    def __call__(self, example, indices):
        num_seqs = len(indices)

        source_ids = [
            i
            for i, counts in enumerate(example["chunks_token_counts"])
            for _ in range(len(counts))
        ]
        chunks_token_ids = [
            torch.tensor(chunk, dtype=torch.long)
            for chunks in example["chunks_token_ids"]
            for chunk in chunks
        ]
        flattened_chunks_token_counts = torch.tensor(
            [chunk for chunks in example["chunks_token_counts"] for chunk in chunks],
            dtype=torch.long,
        )

        flattened_scores = self.score_chunks(
            chunks_token_ids, flattened_chunks_token_counts
        )

        chunk_token_counts = example["chunks_token_counts"]
        chunk_scores = [[[] for _ in range(num_seqs)] for _ in range(self.num_labels)]

        for source_id, score in zip(source_ids, flattened_scores):
            for label in range(self.num_labels):
                chunk_scores[label][source_id].append(score[label].item())

        output = {
            "index": indices,
            "chunk_lengths": chunk_token_counts,
            "length": [sum(counts) for counts in chunk_token_counts],
        }

        for i, label in enumerate(self.labels):
            output[f"{label}_chunks"] = chunk_scores[i]
            output[f"{label}_average"] = [
                np.average(scores, weights=token_counts).item()
                for scores, token_counts in zip(chunk_scores[i], chunk_token_counts)
            ]

        return output
