"""
Script to test and help to understand the process of going from pairwise judgements
to rating scores for each judgement criterion.

Contains excerpts of code from upstream training package.
Now adapted to actually train a model for a single epoch using the upstream methodology.
"""

# %%
from typing import Optional, List
from dataclasses import dataclass
import sys
import os
import logging
import argparse
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    set_seed,
)
import torch
from dotenv import load_dotenv

from qurating.constants import RESULTS_DIR

from qurating.training import (
    PreferenceTrainer,
    TrainingArguments,
    DataCollator,
    LabelFilter,
    ConfidenceFilter,
)
from qurating.modeling import create_model

load_dotenv(override=True)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def validate_dataset_folder(folder_path):
    folder_path = Path(folder_path)
    """Validate that the dataset folder exists and is readable"""
    if not folder_path.exists():
        logger.info(f"Dataset folder not found: {folder_path}")
        logger.info(
            "Please ensure the dataset has been generated using test_dataset_og.py"
        )
        return False
    try:
        test_ds = Dataset.load_from_disk(folder_path)
        if len(test_ds) == 0:
            logger.error("Dataset file is empty")
            return False
        logger.info(f"Dataset validation successful: {len(test_ds)} examples")
        return True
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return False

def validate_dataset_file(file_path):
    """Validate that the dataset file exists and is readable"""
    if not os.path.exists(file_path):
        logger.info(f"Dataset file not found: {file_path}")
        logger.info(
            "Please ensure the dataset has been generated using test_dataset_og.py"
        )
        return False
    try:
        test_ds = Dataset.from_parquet(str(file_path))
        if len(test_ds) == 0:
            logger.error("Dataset file is empty")
            return False
        logger.info(f"Dataset validation successful: {len(test_ds)} examples")
        return True
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return False


@dataclass
class ScriptArguments:
    """Arguments for the training script."""

    model_name_or_path: str = "princeton-nlp/Sheared-LLaMA-1.3b"
    config_name: Optional[str] = None
    tokenizer_name: Optional[str] = None
    max_length: int = 512
    text_field: str = "texts"
    label_field: List[str] = None
    single_label_ablation: int = -1
    cache_dir: str = ".cache"
    use_fast_tokenizer: bool = False
    eval_split_size: float = 0.1

    def __post_init__(self):
        if self.config_name is None:
            self.config_name = self.model_name_or_path
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name_or_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a preference model for educational content evaluation."
    )
    parser.add_argument("--report_to", type=str, default=None)
    parser.add_argument(
        "--model-name",
        type=str,
        default="princeton-nlp/Sheared-LLaMA-1.3b",
        help="Model name or path for the base model",
    )
    parser.add_argument(
        "--dataset-samples",
        type=int,
        default=20000,
        help="Number of samples in the source dataset",
    )
    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=72353534,
        help="Seed used for dataset sampling",
    )
    parser.add_argument(
        "--dataset-templates",
        type=str,
        default="ours_v2",
        help="Name of the templates folder used for pairwise dataset",
    )
    parser.add_argument(
        "--dataset-max-tokens",
        type=int,
        default=512,
        help="Max tokens used for pairwise dataset",
    )
    parser.add_argument(
        "--judgement-model",
        type=str,
        default="gpt-5-mini-2025-08-07-minimal",
        help="Model used for generating judgments",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=20000,
        help="Number of examples to use for training",
    )
    parser.add_argument("--logprobs", type=str, default="generations")
    parser.add_argument(
        "--save-steps", type=int, default=200, help="Number of training epochs"
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size-per-device",
        type=int,
        default=16,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--batch-size-total",
        type=int,
        default=512,
        help="Total training batch size across all devices with gradient accumulation",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Total number of GPUs used for training",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Total number of nodes used for training",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-5, help="Learning rate for training"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_training_output",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--max-length", type=int, default=2048, help="Maximum sequence length"
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.1,
        help="Fraction of data to use for evaluation",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Confidence threshold for filtering training examples",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training")

    return parser.parse_args()


def load_dataset_and_setup_args(args):
    """Load dataset based on command line arguments."""
    # Dataset parameters from args
    n_samples = args.dataset_samples
    seed = args.dataset_seed
    model_name = args.judgement_model
    num_examples = args.num_examples
    use_logprobs_suffix = "_use-logprobs" if args.logprobs == "logprobs" else ""

    dataset_base = f"fwe-fortified_sampled-{n_samples}_seed-{seed}"
    dataset_file = (
        RESULTS_DIR
        / f"tokens_max_{args.dataset_max_tokens}"
        / dataset_base
        / args.dataset_templates
        / f"combined_{model_name}_nexamples-{num_examples}{use_logprobs_suffix}.parquet"
    )

    # Validate dataset file exists and is readable
    logger.info(f"Looking for dataset at: {dataset_file}")
    ds_is_file = True
    if not validate_dataset_file(dataset_file):
        if not validate_dataset_folder(dataset_file.parent / dataset_file.stem):
            logger.error("Dataset validation failed. Cannot proceed.")
            logger.info("\nTo generate the required dataset, run:")
            logger.info("  uv run python scripts/test_dataset_og.py")
            logger.info("  uv run python scripts/test_comparator_og.py")
            raise FileNotFoundError(f"Required dataset not found: {dataset_file}")
        else:
            ds_is_file = False

    # Load the dataset
    try:
        if ds_is_file:
            dataset = Dataset.from_parquet(str(dataset_file))
        else:
            dataset = Dataset.load_from_disk(dataset_file.parent / dataset_file.stem)
        logger.info(f"Successfully loaded dataset with {len(dataset)} examples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Get the label names from the dataset (i.e. the criterion names)
    # TODO: this could be buggy, eg if column order changes
    label_names = [col for col in dataset.column_names if col.endswith("_average")]

    return dataset, label_names


def setup_model_and_tokenizer(args, label_names):
    """Setup model and tokenizer based on command line arguments."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        legacy=False,
    )
    tokenizer.pad_token_id = 0

    # Load config and model
    config = AutoConfig.from_pretrained(args.model_name)
    config.num_labels = len(label_names)
    config.pad_token_id = 0

    # Use model factory for Flash Attention preference with automatic fallback
    model = create_model(args.model_name, config=config, dtype=torch.bfloat16)

    return model, tokenizer


def setup_training_args_and_collator(args, label_names, tokenizer):
    """Setup training arguments and data collator based on command line arguments."""
    # Setup script arguments
    script_args = ScriptArguments(
        model_name_or_path=args.model_name,
        max_length=args.max_length,
        text_field="texts",
        label_field=label_names,
        single_label_ablation=-1,
        eval_split_size=args.eval_split,
        use_fast_tokenizer=False,
    )

    # Setup training arguments
    training_args = TrainingArguments(
        report_to=args.report_to,
        output_dir=args.output_dir,
        run_name=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_device,
        per_device_eval_batch_size=args.batch_size_per_device,
        gradient_accumulation_steps=int(
            args.batch_size_total
            / args.batch_size_per_device
            / args.num_gpus
            / args.num_nodes
        ),
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.1,
        max_grad_norm=1.0,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=args.save_steps,
        do_train=True,
        do_eval=True,
        overwrite_output_dir=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        log_level="info",
        disable_tqdm=False,
        seed=args.seed,
        # Confidence and labeling parameters
        confidence_threshold=args.confidence_threshold,
        label_temperature=1.0,
        log_confidences=[0.5, 0.8],
        greater_is_better=False,
        metric_for_best_model="eval_validation_loss",
        fsdp="auto_wrap",
        ddp_find_unused_parameters=False,
    )

    # Setup data collator
    data_collator = DataCollator(script_args, training_args, tokenizer)

    return script_args, training_args, data_collator


def prepare_datasets(dataset, script_args, training_args, label_names):
    """Prepare train and validation datasets."""
    print("Preparing datasets for training...")

    # Apply label filtering to remove invalid examples
    print(f"Original dataset size: {len(dataset)}")
    dataset_filtered = dataset.filter(
        LabelFilter(label_names), num_proc=1, keep_in_memory=True
    )
    print(f"After label filtering: {len(dataset_filtered)}")

    # Apply confidence filtering if specified
    if training_args.confidence_threshold > 0.0:
        dataset_filtered = dataset_filtered.filter(
            ConfidenceFilter(label_names, training_args.confidence_threshold),
            num_proc=1,
            keep_in_memory=True,
        )
        print(f"After confidence filtering: {len(dataset_filtered)}")

    # Split dataset into train and validation
    if len(dataset_filtered) > 10:  # Only split if we have enough data
        splits = dataset_filtered.train_test_split(
            test_size=script_args.eval_split_size, seed=42
        )
        train_dataset, eval_dataset = splits["train"], splits["test"]
    else:
        # If dataset is too small, use all for training and a subset for eval
        train_dataset = dataset_filtered
        eval_dataset = dataset_filtered.select(range(min(4, len(dataset_filtered))))

    print(f"Training examples: {len(train_dataset)}")
    print(f"Evaluation examples: {len(eval_dataset)}")

    # Create eval dataset dict as expected by trainer
    eval_datasets = {
        "validation": eval_dataset,
        "all": eval_dataset,  # Same dataset for both
    }

    return train_dataset, eval_datasets


def run_training(
    model, tokenizer, train_dataset, eval_datasets, training_args, data_collator
):
    """Run the training process."""
    print("\n" + "=" * 50)
    print("STARTING PREFERENCE MODEL TRAINING")
    print("=" * 50)

    # Set seed for reproducibility
    set_seed(training_args.seed if hasattr(training_args, "seed") else 42)

    # Initialize the PreferenceTrainer
    trainer = PreferenceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Trainer initialized with:")
    print(f"  - Model: {model.__class__.__name__}")
    print(f"  - Training examples: {len(train_dataset)}")
    print(f"  - Validation examples: {len(eval_datasets['validation'])}")
    print(f"  - Epochs: {training_args.num_train_epochs}")
    print(f"  - Batch size per device: {training_args.per_device_train_batch_size}")
    print(
        f"  - Gradient accumulation steps: {training_args.gradient_accumulation_steps}"
    )
    print(f"  - Learning rate: {training_args.learning_rate}")

    # Start training
    print("\nStarting training...")
    try:
        train_result = trainer.train()

        # Save the model
        trainer.save_model()
        print(f"Model saved to: {training_args.output_dir}")

        # Print training metrics
        print("\nTraining completed successfully!")
        print("Final training metrics:")
        for key, value in train_result.metrics.items():
            print(f"  {key}: {value:.4f}")

    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Run evaluation
    if training_args.do_eval:
        print("\nRunning evaluation...")
        try:
            eval_result = trainer.evaluate()
            print("Evaluation metrics:")
            for key, value in eval_result.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"Evaluation failed with error: {e}")
            import traceback

            traceback.print_exc()

    return True


def train():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()

    print("Starting preference model training with arguments:")
    print(f"  Model: {args.model_name}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size per device: {args.batch_size_per_device}")
    print(
        f"  Gradient accumulation steps: {int(
            args.batch_size_total
            / args.batch_size_per_device
            / args.num_gpus
            / args.num_nodes
        )}"
    )
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Output directory: {args.output_dir}")

    try:
        # Load dataset and setup
        dataset, label_names = load_dataset_and_setup_args(args)

        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(args, label_names)

        # Setup training arguments and data collator
        script_args, training_args, data_collator = setup_training_args_and_collator(
            args, label_names, tokenizer
        )

        # Prepare datasets
        train_dataset, eval_datasets = prepare_datasets(
            dataset, script_args, training_args, label_names
        )

        # Run training
        success = run_training(
            model, tokenizer, train_dataset, eval_datasets, training_args, data_collator
        )

        # Final status summary
        print("\n" + "=" * 60)
        print("SCRIPT EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Dataset loaded: {len(dataset)} examples")
        print(f"Label fields found: {len(label_names)} ({', '.join(label_names)})")

        if success:
            print("\n✅ TRAINING COMPLETED SUCCESSFULLY")
            print(f"   - Model: {model.__class__.__name__}")
            print(f"   - Epochs: {training_args.num_train_epochs}")
            print(f"   - Output directory: {training_args.output_dir}")
        else:
            print("\n❌ TRAINING FAILED")
            print("   - Check the error messages above")

        print("\nScript completed!")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Script failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = train()
    sys.exit(exit_code)
