"""Utility functions for training preference models."""

import torch
from typing import List


def confidence_mask(labels, confidence):
    """Create a mask for labels based on confidence threshold."""
    return (labels - 0.5).abs() >= confidence / 2


def bce_with_temperature(probs, labels, temperature=1.0):
    """Binary cross-entropy loss with temperature scaling."""
    probs = probs.clamp(min=0.0, max=1.0)
    labels = labels.clamp(min=0.0, max=1.0)

    if temperature != 1.0:
        labels = (labels.logit() / temperature).sigmoid()

    return torch.nn.functional.binary_cross_entropy(probs, labels)


class LabelFilter:
    """Filter examples that have valid labels."""
    
    def __init__(self, label_field: List[str]):
        self.label_field = label_field

    def __call__(self, example):
        labels = torch.tensor([example[label] for label in self.label_field])
        return not (labels == -100).all().item()


class ConfidenceFilter:
    """Filter examples based on confidence threshold."""
    
    def __init__(self, label_field: List[str], confidence: float):
        self.label_field = label_field
        self.confidence = confidence

    def __call__(self, example):
        labels = torch.tensor([example[label] for label in self.label_field])
        return confidence_mask(labels[labels != -100], self.confidence).any().item()