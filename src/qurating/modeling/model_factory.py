"""Model factory for creating sequence classification models with Flash Attention preference."""

import logging
from transformers import AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


def create_model(model_name, **kwargs):
    """
    Create sequence classification model with Flash Attention preference.

    Attempts to use Flash Attention LlamaForSequenceClassification first,
    falls back to standard AutoModelForSequenceClassification if Flash Attention
    is not available.

    Args:
        model_name: HuggingFace model name or path
        **kwargs: Additional arguments for model creation

    Returns:
        Model instance (either Flash Attention or standard)
    """
    try:
        from .flash_llama import LlamaForSequenceClassification

        logger.info("Using Flash Attention LlamaForSequenceClassification")
        return LlamaForSequenceClassification.from_pretrained(model_name, **kwargs)
    except ImportError as e:
        logger.info(
            f"Flash Attention unavailable ({e}), using standard AutoModelForSequenceClassification"
        )
        return AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
    except Exception as e:
        logger.warning(
            f"Flash Attention model failed to load: {e}, falling back to standard model"
        )
        return AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
