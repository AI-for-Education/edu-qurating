#!/usr/bin/env python3
"""
Inference script for trained QuRater model on pairwise dataset.
Extracts individual texts from pairwise data and runs predictions.
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset
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
    def __init__(self, model_name, labels, device_batch_size):
        self.model_name = model_name
        self.labels = labels
        self.device_batch_size = device_batch_size
        config = AutoConfig.from_pretrained(model_name)
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


def extract_texts_from_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    """Extract individual texts from pairwise dataset."""
    all_texts = []
    metadata = []

    for idx, row in df.iterrows():
        texts = row["texts"]

        for text_idx, text in enumerate(texts):
            all_texts.append(
                {
                    "text": text,
                    "original_pair_id": idx,
                    "text_position": text_idx,  # 'a' or 'b' equivalent (0 or 1)
                    "pair_indices": row["indices"] if "indices" in row else None,
                }
            )

    return pd.DataFrame(all_texts)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on pairwise dataset texts"
    )
    parser.add_argument("input", type=str, help="Input parquet file with pairwise data")
    parser.add_argument("output", type=str, help="Output JSON file")
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "-t", "--tokens", type=int, default=512, help="Chunk size in tokens"
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=16, help="Device batch size"
    )
    parser.add_argument(
        "--subset", type=int, default=None, help="Use only first N pairs (for testing)"
    )

    args = parser.parse_args()

    print("Starting inference on pairwise dataset...")
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    # Load pairwise dataset
    print("Loading pairwise dataset...")
    parquetf = Path(args.input).with_suffix(".parquet")
    if parquetf.exists():
        ds = Dataset.from_parquet(str(args.input))
    elif Path(args.input).is_dir():
        ds = Dataset.load_from_disk(args.input)
    else:
        raise ValueError(f"{args.input} doesn't exist or is not a valid format")
    labels = [
        "_".join(col.split("_")[:-1])
        for col in ds.column_names
        if col.endswith("_average")
    ]
    print(f"Labels: {labels}")

    df = ds.to_pandas()
    print(f"Loaded {len(df)} pairs ({len(df) * 2} total texts)")

    # Use subset if specified
    if args.subset and args.subset > 0:
        df = df.head(args.subset)
        print(f"Using subset of {len(df)} pairs ({len(df) * 2} texts)")

    # Extract individual texts
    print("Extracting individual texts from pairs...")
    texts_df = extract_texts_from_pairwise(df)
    print(f"Extracted {len(texts_df)} individual texts")

    # Convert to dataset format
    dataset = Dataset.from_pandas(texts_df)

    # Get tokenizer name from model directory
    tokenizer_name = args.model
    if Path(args.model).is_dir():
        config_path = Path(args.model) / "config.json"
        if config_path.exists():
            tokenizer_name = args.model
        else:
            tokenizer_name = "princeton-nlp/Sheared-LLaMA-1.3b"

    # Tokenize and chunk
    print("Tokenizing and chunking...")
    tokenizer = TokenizeAndChunk(tokenizer_name, "text", args.tokens)
    processed_dataset = dataset.map(tokenizer, batched=True, remove_columns=["text"])

    print("Running inference...")
    # Initialize annotator
    annotator = ModelAnnotator(args.model, labels, args.batch_size)

    # Run predictions
    results = processed_dataset.map(
        annotator,
        batched=True,
        with_indices=True,
        remove_columns=processed_dataset.column_names,
    )

    # Add back metadata
    results_df = results.to_pandas()

    # Merge with original metadata
    final_df = pd.concat(
        [texts_df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1
    )

    # Save results
    print(f"Saving results to {args.output}")
    Path(args.output).parent.mkdir(exist_ok=True, parents=True)
    final_df.to_json(args.output, orient="records", indent=2)

    # Print summary statistics
    print("\nSummary Statistics:")
    for label in labels:
        if f"{label}_average" in final_df.columns:
            scores = final_df[f"{label}_average"]
            print(
                f"{label}: mean={scores.mean():.3f}, std={scores.std():.3f}, min={scores.min():.3f}, max={scores.max():.3f}"
            )

    print("Inference complete!")


if __name__ == "__main__":
    main()
