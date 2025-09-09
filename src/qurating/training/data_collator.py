"""Data collator for preference model training."""

import torch
from typing import Any, Dict


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