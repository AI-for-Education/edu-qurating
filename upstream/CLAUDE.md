# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QuRating is a research project for selecting high-quality data for training language models using LLM quality judgments. The codebase implements:

1. **QuRater Models**: Preference models that rate text quality across 4 criteria (writing style, facts & trivia, educational value, required expertise)
2. **Data Selection Pipeline**: Tools to annotate datasets with quality scores and select high-quality subsets
3. **Language Model Training**: Scripts to train language models on selected data
4. **LLM Prompting**: Tools to collect pairwise and individual quality judgments from OpenAI models

## Key Commands

### Environment Setup
```bash
# Install dependencies in specific order (required)
pip install packaging==23.2
pip install torch==2.1.1 torchaudio==2.1.1 torchvision==0.16.1
pip install -r requirements.txt
```

### Training Commands
```bash
# Train QuRater preference model
./TrainQuRater.sh
# With custom parameters:
BSZ=512 SEQ=4 ./TrainQuRater.sh

# Train language model
DATASET=<path_to_dataset> BSZ=2048 SEQ=8 ./TrainLM.sh
# With curriculum learning:
DATASET=<path> ./TrainLM.sh --ordered --sort_by <column_name>
```

### Data Processing
```bash
# Annotate dataset with QuRater scores
python -m data_tools.qurater_annotate json <output_path> \
    -F <jsonl_files> \
    -M princeton-nlp/QuRater-1.3B \
    --text_field text \
    --labels writing_style required_expertise facts_and_trivia educational_value

# Select high-quality subset
python -m data_tools.select_subset <annotated_dataset> <output_path> \
    --metric_field educational_value_average \
    --seq_len_field <token_count_field> \
    --tokens 1_000_000_000 \
    --temperature 2.0 \
    --normalize

# Tokenize dataset for training
python -m data_tools.tokenize_dataset <input> <output> \
    --tokenizer <model_name> \
    --max_length 2048
```

### LLM Prompting for Judgments
```bash
# Collect pairwise comparisons
python prompting/score_pairwise.py <input> <output> \
    -n 1000 -k 2 \
    --model gpt-3.5-turbo \
    --template_file prompting/templates/pairwise_educational_value.txt \
    --tokens_min 256 --tokens_max 512
```

### Evaluation
```bash
# Use included lm-evaluation-harness
cd eval/lm-evaluation-harness
python main.py --model huggingface --model_args pretrained=<model_path> --tasks <task_list>
```

## Architecture Overview

### Core Components

**Training Pipeline (`training/`)**:
- `train_preference_model.py`: Trains QuRater models using pairwise preference data
- `train_language_model.py`: Trains language models with Flash Attention optimizations
- `trainer.py`: Custom trainer extending HuggingFace with FSDP support and custom learning rate schedules

**Data Processing (`data_tools/`)**:
- `qurater_annotate.py`: Adds quality ratings to text datasets using trained QuRater models
- `select_subset.py`: Selects data subsets based on quality scores with various sampling strategies
- `tokenize_dataset.py`: Tokenizes and chunks text data for training

**LLM Prompting (`prompting/`)**:
- `score_pairwise.py`: Collects pairwise quality comparisons from OpenAI models
- `score_individual.py`: Collects individual quality ratings
- `templates/`: Prompt templates for different quality criteria

**Custom Models (`modeling/`)**:
- `modeling_flash_llama.py`: LLaMA models with Flash Attention integration

### Data Flow

1. **Judgment Collection**: Use OpenAI models to rate text quality via `prompting/` scripts
2. **QuRater Training**: Train preference models on collected judgments via `TrainQuRater.sh`
3. **Data Annotation**: Apply trained QuRater to large datasets via `qurater_annotate.py`
4. **Data Selection**: Select high-quality subsets via `select_subset.py`
5. **LM Training**: Train language models on selected data via `TrainLM.sh`

### Quality Criteria

The system evaluates text across 4 dimensions:
- **Writing Style**: Grammar, clarity, structure
- **Facts & Trivia**: Factual accuracy and interesting information
- **Educational Value**: Learning potential and instructional quality  
- **Required Expertise**: Complexity and domain knowledge needed

## Configuration

### Training Scripts Environment Variables

**TrainLM.sh**:
- `ARCH`: Model architecture (default: princeton-nlp/Sheared-LLaMA-1.3b)
- `BSZ`: Total batch size (default: 2048)
- `SEQ`: Sequences per device (default: 16)
- `LR`: Learning rate (default: 5e-4)
- `EPOCHS`: Training epochs (default: 1)
- `DATASET`: Path to tokenized training dataset

**TrainQuRater.sh**:
- `MODEL`: Base model (default: princeton-nlp/Sheared-LLaMA-1.3b)
- `BSZ`: Batch size (default: 512)
- `CONFIDENCE`: Minimum prediction confidence (default: 0.5)
- `LABELTEMP`: Temperature for label processing (default: 1.0)

### Multi-GPU and Multi-Node Support

Both training scripts automatically detect available GPUs and support:
- FSDP (Fully Sharded Data Parallel) training
- Gradient accumulation for large effective batch sizes
- SLURM integration for multi-node training

## Important Notes

- **Flash Attention**: The codebase uses custom Flash Attention models for memory efficiency
- **FSDP**: Training uses FSDP with hybrid sharding strategy for large models
- **Weights & Biases**: Logging is configured for offline mode by default
- **HuggingFace Integration**: Models and datasets are downloaded from HuggingFace Hub
- **Mixed Precision**: Uses BF16 for training efficiency