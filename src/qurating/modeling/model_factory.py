"""Model factory for creating sequence classification models with Flash Attention preference."""

import logging
from transformers import AutoModelForSequenceClassification
import torch

logger = logging.getLogger(__name__)


def create_model(model_name, num_labels, reinit_score_layer=True, **kwargs):
    """
    Create sequence classification model with Flash Attention preference.

    Attempts to use Flash Attention LlamaForSequenceClassification first,
    falls back to standard AutoModelForSequenceClassification if Flash Attention
    is not available.

    Args:
        model_name: HuggingFace model name or path
        num_labels: number of labels for output layer
        **kwargs: Additional arguments for model creation

    Returns:
        Model instance (either Flash Attention or standard)
    """

    def init_score_layer(model, num_labels):
        model.score = torch.nn.Linear(
            in_features=model.score.in_features,
            out_features=num_labels,
            bias=model.score.bias,
            dtype=model.dtype,
        )
        model.config.num_labels = num_labels
        return model

    try:
        from .flash_llama import LlamaForSequenceClassification

        logger.info("Using Flash Attention LlamaForSequenceClassification")
        model = LlamaForSequenceClassification.from_pretrained(model_name, **kwargs)
        if reinit_score_layer:
            model = init_score_layer(model, num_labels)
        return model
    except ImportError as e:
        logger.info(
            f"Flash Attention unavailable ({e}), using standard AutoModelForSequenceClassification"
        )
        model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
        if reinit_score_layer:
            model = init_score_layer(model, num_labels)
        return model

    except Exception as e:
        logger.warning(
            f"Model is not LLama or otherwise failed to instantiate: {e}, using AutoModel with flash_attn"
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, attn_implementation="flash_attention_2", **kwargs
        )
        if reinit_score_layer:
            model = init_score_layer(model, num_labels)
        return model
