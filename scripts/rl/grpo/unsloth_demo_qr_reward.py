# %%
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from trl import GRPOConfig, GRPOTrainer, SFTTrainer, SFTConfig  # type: ignore
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
import torch
from transformers import TextStreamer
from vllm import SamplingParams

from qr_reward import QuratingReward
from qurating.constants import DATA_DIR

evals_dir = DATA_DIR / "education_evals"

max_seq_length = 2048
max_prompt_length = 256
lora_rank = 32

# %%
# load qurating reward functions
qr_reward = QuratingReward()

score_spec_core_primary = {
    "core_ed": {
        "pedagogical_structure": 1,
        "lesson_engagement": 0.25,
        "factual_accuracy": 1,
        "education_level_primary": 1,
    }
}

score_spec_fl_teacher = {
    **{"fl_teacher": {label: 1 for label in qr_reward.labels["fl_teacher"]}}
}

reward_fun_core_primary = qr_reward.reward_fun_generator(score_spec_core_primary)
reward_fun_fl_teacher = qr_reward.reward_fun_generator(score_spec_fl_teacher)

# %%
system_prompt = """You are an early grade primary teacher given a task to follow."""

# %%
dataset = Dataset.load_from_disk(
    str(evals_dir / "education_evals_combined_literacy_grade0-3.parquet")
)
dataset

# %%
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Base",
    max_seq_length=max_seq_length,
    load_in_4bit=False,  # False for LoRA 16bit
    fast_inference=True,  # Enable vllm fast inference
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.8,  # Reduce if out of memory
)
tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=lora_rank * 2,  # *2 speeds up training
    use_gradient_checkpointing="unsloth",  # Reduces memory usage
    random_state=3407,
)

tokenizer.apply_chat_template(
    [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": dataset[0]["Rendered Prompt"]},
        {
            "role": "assistant",
            "content": dataset[0]["Good Response"],
        },
    ],
    tokenize=False,
    add_generation_prompt=True,
)

# %%
dataset = dataset.map(
    lambda x: {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": x["Rendered Prompt"]},
        ]
    }
)
dataset[0]

reward_fun_core_primary(
    prompts=[[{"content": ""}]],
    completions=[[{"role": "assistant", "content": "ab " * 1000}]],
)

# %%
tokenized = dataset.map(
    lambda x: {
        "tokens": tokenizer.apply_chat_template(
            x["prompt"], add_generation_prompt=True, tokenize=True
        )
    },
    batched=True,
)
print(tokenizer.decode(tokenized[0]["tokens"]))
tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})

maximum_length = int(np.quantile(tokenized["L"], 0.9))
print("Max Length = ", maximum_length)

# # Filter only samples smaller than 90% max length
# dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
# del tokenized

# %%
max_prompt_length = maximum_length + 1  # + 1 just in case!
max_completion_length = max_seq_length - max_prompt_length

vllm_sampling_params = SamplingParams(
    min_p=0.1,
    top_p=1.0,
    top_k=-1,
    seed=3407,
    stop=[tokenizer.eos_token],
    include_stop_str_in_output=True,
)

training_args = GRPOConfig(
    vllm_sampling_params=vllm_sampling_params,
    temperature=1.0,
    learning_rate=5e-6,
    weight_decay=0.001,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,  # Increase to 4 for smoother training
    num_generations=4,  # Decrease if out of memory
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    num_train_epochs=1,  # Set to 1 for a full training run
    max_steps=2000,
    save_steps=100,
    report_to="none",  # Can use Weights & Biases
    output_dir="outputs",
    # For optional training + evaluation
    # fp16_full_eval = True,
    # per_device_eval_batch_size = 4,
    # eval_accumulation_steps = 1,
    # eval_strategy = "steps",
    # eval_steps = 1,
)

# For optional training + evaluation
# new_dataset = dataset.train_test_split(test_size = 0.01)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        reward_fun_core_primary,
        reward_fun_fl_teacher,
    ],  # type: ignore
    args=training_args,
    train_dataset=dataset,
    # For optional training + evaluation
    # train_dataset = new_dataset["train"],
    # eval_dataset = new_dataset["test"],
)
trainer.train()

# %%
model.save_lora("grpo_saved_lora")

# %%
from safetensors import safe_open

tensors = {}
with safe_open("grpo_saved_lora/adapter_model.safetensors", framework="pt") as f:
    # Verify both A and B are non zero
    for key in f.keys():
        tensor = f.get_tensor(key)
        n_zeros = (tensor == 0).sum() / tensor.numel()
        assert n_zeros.item() != tensor.numel()

# %%
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the sqrt of 101?"},
]

text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,  # Must add for generation
    tokenize=False,
)

sampling_params = SamplingParams(
    temperature=1.0,
    top_k=50,
    max_tokens=2048,
)
output = (
    model.fast_generate(
        text,
        sampling_params=sampling_params,
        lora_request=model.load_lora("grpo_saved_lora"),
    )[0]
    .outputs[0]
    .text
)

output

# %%
# Merge to 16bit
model.save_pretrained_merged(
    "qwen_finetune_16bit",
    tokenizer,
    save_method="merged_16bit",
)

# # Merge to 4bit
# model.save_pretrained_merged("qwen_finetune_4bit", tokenizer, save_method = "merged_4bit",)

# Just LoRA adapters
model.save_pretrained("qwen_lora")
tokenizer.save_pretrained("qwen_lora")

model.save_pretrained_gguf("qwen_finetune", tokenizer, quantization_method="f16")
model.save_pretrained_gguf("qwen_finetune", tokenizer, quantization_method="bf16")
