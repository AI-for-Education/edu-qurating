# Gemini Code Assistant Guidance

This file provides guidance to the Gemini code assistant when working with the QuRating repository.

## Project Overview

QuRating is a research project focused on leveraging large language models (LLMs) to assess and select high-quality data for training other language models. The core idea is to use LLM-generated quality judgments to train "QuRater" models, which can then be used to score and filter large datasets.

The project is structured around the following key activities:

1.  **LLM-based Judgment Collection**: Using LLMs (e.g., GPT models) to generate pairwise comparisons and individual scores for text quality based on various criteria.
2.  **QuRater Model Training**: Training preference models (QuRaters) on the collected judgments to predict text quality.
3.  **Data Annotation and Selection**: Applying the trained QuRaters to score large datasets and then selecting high-quality subsets based on these scores.
4.  **Language Model Training**: Training language models on the curated, high-quality datasets.
5.  **Evaluation**: Assessing the performance of the trained language models.

## Key Files and Directories

*   `training/`: Contains scripts for training both the QuRater preference models and the final language models.
    *   `train_preference_model.py`: The core script for training QuRater models.
    *   `train_language_model.py`: The script for training language models on curated data.
    *   `trainer.py`: A custom Hugging Face Trainer with support for FSDP.
*   `data_tools/`: A collection of scripts for data processing.
    *   `qurater_annotate.py`: Annotates a dataset with quality scores from a trained QuRater model.
    *   `select_subset.py`: Selects a subset of a dataset based on quality scores.
    *   `tokenize_dataset.py`: Tokenizes a dataset for training.
*   `prompting/`: Scripts for generating quality judgments from LLMs.
    *   `score_pairwise.py`: Generates pairwise quality comparisons.
    *   `score_individual.py`: Generates individual quality scores.
    *   `templates/`: Prompt templates for different quality dimensions.
*   `modeling/`: Contains custom model implementations.
    *   `modeling_flash_llama.py`: A LLaMA model implementation with Flash Attention.
*   `eval/`: Contains the `lm-evaluation-harness` for evaluating language models.
*   `TrainQuRater.sh` and `TrainLM.sh`: Shell scripts for launching training jobs.

## Common Workflows

### Training a QuRater Model

1.  **Generate judgments**: Use the scripts in `prompting/` to create a dataset of pairwise comparisons.
2.  **Train the model**: Run the `TrainQuRater.sh` script, pointing it to the training data.

    ```bash
    ./TrainQuRater.sh
    ```

### Training a Language Model with Curated Data

1.  **Annotate a dataset**: Use `data_tools/qurater_annotate.py` to score a large, uncurated dataset with a trained QuRater model.
2.  **Select a high-quality subset**: Use `data_tools/select_subset.py` to filter the annotated dataset based on the quality scores.
3.  **Tokenize the subset**: Use `data_tools/tokenize_dataset.py` to prepare the data for training.
4.  **Train the language model**: Run the `TrainLM.sh` script, providing the path to the tokenized dataset.

    ```bash
    DATASET=<path_to_tokenized_dataset> ./TrainLM.sh
    ```

### Evaluating a Trained Language Model

The project uses a fork of the `lm-evaluation-harness` located in the `eval/` directory.

```bash
cd eval/lm-evaluation-harness
python main.py --model huggingface --model_args pretrained=<path_to_trained_model> --tasks <task_name>
```

## Technical Details

*   **Models**: The project primarily uses LLaMA-based architectures, with custom modifications for Flash Attention.
*   **Training**: Training is performed using PyTorch and Hugging Face Transformers, with support for Fully Sharded Data Parallel (FSDP) for large-scale training.
*   **Environment**: The required Python packages are listed in `requirements.txt`. It is recommended to install them in the specified order.
