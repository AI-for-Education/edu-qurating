"""Preference trainer for QuRating models."""

import torch
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from transformers import Trainer, TrainingArguments as BaseTrainingArguments

from .utils import confidence_mask, bce_with_temperature


@dataclass
class TrainingArguments(BaseTrainingArguments):
    """Training arguments with preference-specific parameters."""
    
    label_temperature: float = field(
        default=1.0,
        metadata={"help": "Label temperature"},
    )
    log_confidences: List[float] = field(
        default_factory=lambda: [0.5, 0.8],
        metadata={"help": "Confidence thresholds for logging accuracy"},
    )
    confidence_threshold: float = field(
        default=0.0,
        metadata={
            "help": "Confidence threshold for including data during training"
        },
    )


class PreferenceTrainer(Trainer):
    """Trainer for preference models using pairwise comparisons."""
    
    def compute_loss(self, model, inputs, return_outputs=False, return_output_and_metrics=False, num_items_in_batch=None):
        """
        Compute loss for preference model training.
        
        This implements the pairwise preference loss used in QuRating.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs, use_cache=False)
        
        # Convert logits to pairwise preference probabilities
        logit_diffs = outputs.logits.unsqueeze(0) - outputs.logits.unsqueeze(1)
        probs = logit_diffs.float().sigmoid()

        # Apply confidence filtering and compute loss
        valid_mask = (labels != -100) & confidence_mask(labels, self.args.confidence_threshold)
        loss = bce_with_temperature(probs[valid_mask], labels[valid_mask], self.args.label_temperature)

        if return_output_and_metrics:
            correct = torch.where(
                (labels != -100), ((probs >= 0.5) == (labels >= 0.5)).float(), float('nan')
            )

            metrics = {
                "acc": correct,
            }
            
            # Add confidence-based accuracy metrics
            for confidence in self.args.log_confidences:
                metrics[f"acc_confidence{confidence * 100}"] = torch.where(
                    confidence_mask(labels, confidence), correct, float('nan')
                )

            # Add per-label metrics
            for i in range(probs.shape[-1]):
                valid_mask_i = (labels[..., i] != -100)
                if valid_mask_i.any():
                    loss_i = bce_with_temperature(
                        probs[..., i][valid_mask_i], 
                        labels[..., i][valid_mask_i], 
                        self.args.label_temperature
                    )
                    
                    # Use upper triangular mask for pairwise comparisons
                    valid_mask_i = valid_mask_i & torch.triu(torch.ones_like(labels[..., i])).bool()
                    correct_i = torch.where(
                        valid_mask_i, 
                        ((probs[..., i] >= 0.5) == (labels[..., i] >= 0.5)).float(), 
                        float('nan')
                    )

                    metrics.update({
                        f"label{i}_loss": loss_i,
                        f"label{i}_acc": correct_i,
                    })
                    
                    # Add confidence-based per-label metrics
                    for confidence in self.args.log_confidences:
                        metrics[f"label{i}_acc_confidence{confidence * 100}"] = torch.where(
                            confidence_mask(labels[..., i], confidence), correct_i, float('nan')
                        )

            return (loss, outputs, metrics)
        
        if return_outputs:
            return (loss, outputs)
        else:
            return loss