"""Modeling utilities for QuRating."""

try:
    from .flash_llama import LlamaForSequenceClassification
    _FLASH_AVAILABLE = True
except ImportError:
    _FLASH_AVAILABLE = False

from .model_factory import create_model

__all__ = ["create_model"]

if _FLASH_AVAILABLE:
    __all__.append("LlamaForSequenceClassification")