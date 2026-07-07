#!/usr/bin/env python3
"""
Inference script for trained QuRater model on pairwise dataset.
Extracts individual texts from pairwise data and runs predictions.
"""

import argparse
import pandas as pd
from pathlib import Path
from datasets import Dataset

from qurating.inference import TokenizeAndChunk, ModelAnnotator


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
