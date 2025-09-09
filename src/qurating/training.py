"""Consolidated training utilities for QuRating preference models."""

import torch
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from transformers import Trainer, TrainingArguments as BaseTrainingArguments



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



def label_filter(example, label_field: List[str]):
    """Filter examples that have valid labels."""
    labels = torch.tensor([example[label] for label in label_field])
    return not (labels == -100).all().item()


def confidence_filter(example, label_field: List[str], confidence: float):
    """Filter examples based on confidence threshold."""
    labels = torch.tensor([example[label] for label in label_field])
    return confidence_mask(labels[labels != -100], confidence).any().item()


class LabelFilter:
    """Filter examples that have valid labels."""
    
    def __init__(self, label_field: List[str]):
        self.label_field = label_field

    def __call__(self, example):
        return label_filter(example, self.label_field)


class ConfidenceFilter:
    """Filter examples based on confidence threshold."""
    
    def __init__(self, label_field: List[str], confidence: float):
        self.label_field = label_field
        self.confidence = confidence

    def __call__(self, example):
        return confidence_filter(example, self.label_field, self.confidence)



class DataCollator:
    """Data collator for pairwise preference training."""
    
    def __init__(self, args, training_args, tokenizer):
        self.args = args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.pad_token_id = self.tokenizer.pad_token_id
        self.max_length = getattr(args, 'max_length', 512)

    @torch.no_grad()
    def __call__(self, features: Any) -> Dict[str, Any]:
        batch = self.tokenizer(
            sum([item[self.args.text_field] for item in features], []),
            add_special_tokens=False,
            truncation=True,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
        )

        bsz = batch.input_ids.size(0)
        num_labels = len(self.args.label_field)
        labels = -100 * torch.ones(bsz, bsz, num_labels, dtype=torch.float32)

        counter = 0
        for item in features:
            k = len(item[self.args.text_field])
            for i, label in enumerate(self.args.label_field):
                labels[counter : counter + k, counter : counter + k, i] = torch.tensor(
                    item[label], dtype=torch.float32
                )
            counter += k

        # Handle single label ablation if specified
        single_label_ablation = getattr(self.args, 'single_label_ablation', -1)
        for i in range(labels.size(-1)):
            if single_label_ablation >= 0 and i != single_label_ablation:
                labels[:, :, i] = -100

        labels[range(bsz), range(bsz)] = -100

        return dict(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=labels,
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