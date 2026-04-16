# %%
import re

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import GRPOConfig, GRPOTrainer  # type: ignore
import numpy as np
from datasets import Dataset
from vllm import SamplingParams
from safetensors import safe_open
from transformers import TextStreamer
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from qr_reward import QuratingReward
from qurating.constants import DATA_DIR

from chat_templates import ReasoningStudentMaterial

evals_dir = DATA_DIR / "education_evals"

max_seq_length = 2048
max_prompt_length = 256
lora_rank = 32

# %%
template_manager = ReasoningStudentMaterial()
print(template_manager.chat_template())
print(template_manager.render_system_prompt())

# %%
dataset = Dataset.load_from_disk(
    str(evals_dir / "education_evals_combined_literacy_grade0-3_train.parquet")
)
print(dataset)

test_ds = Dataset.load_from_disk(
    str(evals_dir / "education_evals_combined_literacy_grade0-3_test.parquet")
)
print(test_ds)

prompts = list(test_ds["Rendered Prompt"])
test_promptiter = iter(prompts)

# %%
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Base",
    max_seq_length=max_seq_length,
    load_in_4bit=False,  # False for LoRA 16bit
    fast_inference=True,  # Enable vllm fast inference
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.8,  # Reduce if out of memory
)
tokenizer.chat_template = template_manager.chat_template()
# tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")

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
        {"role": "system", "content": template_manager.render_system_prompt()},
        {"role": "user", "content": dataset[0]["Rendered Prompt"]},
    ],
    tokenize=False,
    add_generation_prompt=True,
)

# %%
test_str_1 = (
    f"Let me think!{template_manager.reasoning_end}"
    f"{template_manager.material_needed_start}yes{template_manager.material_needed_end}"
    f"{template_manager.student_material_start}a story{template_manager.student_material_end}"
    f"{template_manager.response_start}\n2\n{template_manager.response_end}"
)
test_str_2 = (
    f"{template_manager.reasoning_start}Let me think!{template_manager.reasoning_end}"
    f"{template_manager.material_needed_start}no{template_manager.material_needed_end}"
    f"{template_manager.student_material_start}a story{template_manager.student_material_end}"
    f"{template_manager.response_start}  2  {template_manager.response_end}\n\n"
)

match_format = template_manager.formatter(tokenizer)

print(match_format.findall(test_str_1))

print(match_format.findall(test_str_2))

print(match_format.search(test_str_1))

match = match_format.match(test_str_1)
if match is not None:
    print(match.groups()[3])

# %%
# load qurating reward functions
qr_reward = QuratingReward()

score_spec_reasoning = {
    "core_ed": {"pedagogical_structure": 1, "factual_accuracy": 1},
    "fl_teacher": {label: 1 for label in qr_reward.labels["fl_teacher"]},
}

score_spec_materials = {
    "core_ed": {
        "pedagogical_structure": 1,
        "factual_accuracy": 1,
        "lesson_engagement": 0.5,
        "education_level_primary": 0.5,
    },
    "fl_student": {label: 1 for label in qr_reward.labels["fl_student"]},
}

reward_fun_reasoning = qr_reward.reward_fun_generator(
    score_spec_reasoning, grouper=match_format, group_idx=0, verbose=True
)
reward_fun_materials = qr_reward.reward_fun_generator(
    score_spec_materials, grouper=match_format, group_idx=2
)


def length_target(completions, **kwargs):
    assert "Good Response" in kwargs
    max_len = max_seq_length
    len_score = []
    for completion, good_response in zip(completions, kwargs["Good Response"]):
        text = completion[0]["content"]
        match = match_format.match(text)
        if match is None:
            len_score.append(0)
        else:
            groups = match.groups()
            if len(groups) != 4:
                len_score.append(0)
            else:
                resp_len = len(groups[3])
                good_len = len(good_response)
                len_score.append(5 * (1 - np.abs(resp_len - good_len) / max_len) ** 0.5)
    return len_score


def material_needed_consistency(completions, **kwargs):
    scores = []
    for completion in completions:
        text = completion[0]["content"]
        match = match_format.match(text)
        if match is None:
            scores.append(0)
        else:
            groups = match.groups()
            if len(groups) != 4:
                scores.append(0.0)
            else:
                mat_needed = groups[1]
                if mat_needed not in ("yes", "no"):
                    scores.append(0.0)
                else:
                    if mat_needed == "yes" and len(groups[2]) > 0:
                        scores.append(3.0)
                    elif mat_needed == "no" and len(groups[2]) == 0:
                        scores.append(3.0)
                    else:
                        scores.append(0.0)
    return scores


def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None:
            score += 3.0
        scores.append(score)
    return scores


def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Count how many keywords are seen - we penalize if too many!
        # If we see 1, then plus some points!

        # No need to reward the opening tag since we always prepend it!
        # score += 0.5 if response.count(reasoning_start) == 1 else -1.0
        score += 0.5 if response.count(template_manager.reasoning_end) == 1 else -1.0
        score += 0.5 if response.count(template_manager.material_needed_start) == 1 else -1.0
        score += 0.5 if response.count(template_manager.material_needed_end) == 1 else -1.0
        score += 0.5 if response.count(template_manager.student_material_start) == 1 else -1.0
        score += 0.5 if response.count(template_manager.student_material_end) == 1 else -1.0
        score += 0.5 if response.count(template_manager.response_start) == 1 else -1.0
        score += 0.5 if response.count(template_manager.response_end) == 1 else -1.0
        scores.append(score)
    return scores


# %%
dataset = dataset.map(
    lambda x: {
        "prompt": [
            {"role": "system", "content": template_manager.system_prompt},
            {"role": "user", "content": x["Rendered Prompt"]},
        ]
    }
)
dataset[0]

prompts = [[{"content": ""}]]
completions = [
    [
        {
            "role": "assistant",
            "content": (
                f"{'ab ' * 1000}"
                f"{template_manager.reasoning_end}"
                f"{template_manager.material_needed_start}no{template_manager.material_needed_end}"
                f"{template_manager.student_material_start}ababa{template_manager.student_material_end}"
                f"{template_manager.response_start}ab{template_manager.response_end}"
            )
        }
    ]
]


print(reward_fun_reasoning(prompts=prompts, completions=completions))
print(reward_fun_materials(prompts=prompts, completions=completions))
print(match_format_exactly(completions=completions))
print(match_format_approximately(completions=completions))
print(material_needed_consistency(completions=completions))

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
    output_dir="reasoning/test1/outputs",
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
        reward_fun_reasoning,
        reward_fun_materials,
        match_format_exactly,
        match_format_approximately,
        material_needed_consistency,
        length_target,  # type: ignore
    ],
    args=training_args,
    train_dataset=dataset,
    # For optional training + evaluation
    # train_dataset = new_dataset["train"],
    # eval_dataset = new_dataset["test"],
)

# %%
trainer.train()

# %%
model.save_lora("reasoning/test1/grpo_saved_lora")

# %%
tensors = {}
with safe_open(
    "reasoning/test1/grpo_saved_lora/adapter_model.safetensors", framework="pt"
) as f:
    # Verify both A and B are non zero
    for key in f.keys():
        tensor = f.get_tensor(key)
        n_zeros = (tensor == 0).sum() / tensor.numel()
        assert n_zeros.item() != tensor.numel()


# %%
prompt = next(test_promptiter)
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt},
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
        lora_request=model.load_lora("reasoning/test1/grpo_saved_lora"),
    )[0]
    .outputs[0]
    .text
)

print(prompt)
print("-" * 50)
print(output)
print("#" * 50)

# %%
# Merge to 16bit
model.save_pretrained_merged(
    "reasoning/test1/qwen_finetune_16bit",
    tokenizer,
    save_method="merged_16bit",
)

# # Merge to 4bit
# model.save_pretrained_merged("qwen_finetune_4bit", tokenizer, save_method = "merged_4bit",)

# Just LoRA adapters
model.save_pretrained("reasoning/test1/qwen_lora")
tokenizer.save_pretrained("reasoning/test1/qwen_lora")

model.save_pretrained_gguf(
    "reasoning/test1/qwen_finetune", tokenizer, quantization_method="f16"
)
model.save_pretrained_gguf(
    "reasoning/test1/qwen_finetune", tokenizer, quantization_method="bf16"
)
