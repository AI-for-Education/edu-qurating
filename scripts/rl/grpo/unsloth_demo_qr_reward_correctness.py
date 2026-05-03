# %%
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import GRPOConfig, GRPOTrainer  # type: ignore
import numpy as np
from datasets import Dataset, load_from_disk
from fdllm import register_models, get_caller, LLMMessage
from safetensors import safe_open
from transformers import TextStreamer
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from vllm import SamplingParams
from pydantic import BaseModel, Field

from qr_reward import QuratingReward
from qurating.constants import DATA_DIR, ROOT

register_models(ROOT / "custom_models.yaml")

evals_dir = DATA_DIR / "education_evals"

max_seq_length = 2048
max_prompt_length = 256
lora_rank = 32

# content_column = "Rendered Prompt"
# GOOD_RESPONSE_COLUMN = "Good Response"
CONTENT_COLUMN = "content"
GOOD_RESPONSE_COLUMN = "example_of_good_answer"

# %%
# load qurating reward functions
qr_reward = QuratingReward()
score_cap = (-np.inf, 24.0)

score_spec_core_primary = {
    "core_ed": {
        "pedagogical_structure": 0.1,
        "lesson_engagement": 0.25,
        "factual_accuracy": 1,
        # "education_level_primary": 0.25,
    }
}

# score_spec_fl_teacher = {**{"fl_teacher": {"3_systematic_phonics": 1}}}
score_spec_fl_teacher = {
    **{
        "fl_teacher": {
            (
                "1_oral_language_vocabulary",
                (("criterion", "pairwise_1_oral_language_vocabulary"),),
            ): 1,
            (
                "2_phonological_awareness",
                (("criterion", "pairwise_2_phonological_awareness"),),
            ): 1,
            (
                "3_systematic_phonics",
                (("criterion", "pairwise_3_systematic_phonics"),),
            ): 1,
            ("4_reading_fluency", (("criterion", "pairwise_4_reading_fluency"),)): 1,
            (
                "5_reading_comprehension",
                (("criterion", "pairwise_5_reading_comprehension"),),
            ): 1,
            ("6_writing_encoding", (("criterion", "pairwise_6_writing_encoding"),)): 1,
            (
                "7_pedagogical_quality",
                (("criterion", "pairwise_7_pedagogical_quality"),),
            ): 1,
        }
    }
}

reward_fun_core_primary = qr_reward.reward_fun_generator(
    score_spec_core_primary, name="core_ed", score_cap=score_cap
)
reward_fun_fl_teacher = qr_reward.reward_fun_generator(
    score_spec_fl_teacher, name="fl_teacher_interleaved", score_cap=score_cap
)


class CorrectnessResponse(BaseModel):
    length_match: bool = Field(
        description="True if the length of the response matches the good response"
    )
    format_match: bool = Field(
        description=(
            "True if the overall formatting of the response matches the good response"
        )
    )
    instruction_following: bool = Field(
        description=(
            "True if the response follows the task instructions as well"
            " or almost as well as the good response"
        )
    )


caller = get_caller("gemma-4-E4B")
weight = 3


def correctness_reward(prompts, completions, **kwargs):
    assert GOOD_RESPONSE_COLUMN in kwargs
    prompts_text = [prompt[-1]["content"] for prompt in prompts]
    responses = [completion[0]["content"] for completion in completions]
    good_responses = kwargs[GOOD_RESPONSE_COLUMN]
    scores = []
    for prompt, resp, good_resp in zip(prompts_text, responses, good_responses):
        message_text = (
            "Below is a task instruction, a response to that task from a trainee assistant,"
            " and a corresponding good response from an expert assistant."
            f"\n<Task>\n{prompt}\n</Task>\n<Response>\n{resp}\n</Response>"
            f"\n<Good Response>\n{good_resp}\n</Good Response>"
            "\n\nI want you to judge the trainee response against the good response."
        )
        message = LLMMessage(Role="user", Message=message_text)
        out = caller.call(
            message, response_schema=CorrectnessResponse, max_tokens=64000
        )
        try:
            out_obj = CorrectnessResponse.model_validate_json(out.Message)
        except:
            scores.append(0.0)
            continue
        print(prompt)
        print(resp)
        print(out_obj.model_dump_json(indent=2))
        score = (
            float(out_obj.length_match)
            + float(out_obj.format_match)
            + float(out_obj.instruction_following)
        )
        scores.append(score * weight)
    return scores


# def length_target(completions, **kwargs):
#     assert GOOD_RESPONSE_COLUMN in kwargs
#     resp_len = np.array([len(completion[0]["content"]) for completion in completions])
#     good_len = np.array([len(gr) for gr in kwargs[GOOD_RESPONSE_COLUMN]])
#     max_len = max_seq_length
#     print(resp_len)
#     print(good_len)
#     print(max_len)
#     len_score = 5 * np.maximum((1 - np.abs(resp_len - good_len) / max_len), 0) ** 0.5
#     return len_score.tolist()


# %%
system_prompt = """/flteacher"""

# %%
dataset = load_from_disk(str(evals_dir / "flteach_grpo_dataset_train-test"))["train"]
dataset

# %%
test_ds = load_from_disk(str(evals_dir / "flteach_grpo_dataset_train-test"))["test"]
test_ds

prompts = list(test_ds[CONTENT_COLUMN])
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
        {"role": "user", "content": dataset[0][CONTENT_COLUMN]},
        {
            "role": "assistant",
            "content": dataset[0][GOOD_RESPONSE_COLUMN],
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
            {"role": "user", "content": x[CONTENT_COLUMN]},
        ]
    }
)
dataset[0]

reward_fun_core_primary(
    prompts=[[{"content": ""}]],
    completions=[[{"role": "assistant", "content": "ab " * 1000}]],
)

correctness_reward(
    prompts=[dataset[0]["prompt"]],
    completions=[[{"role": "assistant", "content": dataset[0][GOOD_RESPONSE_COLUMN]}]],
    **{GOOD_RESPONSE_COLUMN: [dataset[0][GOOD_RESPONSE_COLUMN]]},
)

reward_fun_fl_teacher(
    prompts=[[{"content": ""}]] * 3,
    completions=[[{"role": "assistant", "content": "ab " * 1000}]] * 3,
    criterion=[
        "pairwise_1_oral_language_vocabulary",
        "pairwise_2_phonological_awareness",
        "pairwise_3_systematic_phonics",
    ],
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

maximum_length = int(np.quantile(tokenized["L"], 0.99))
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
    output_dir="interleaved_scoring/test2/outputs",
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
        correctness_reward,  # type: ignore
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
model.save_lora("interleaved_scoring/test2/grpo_saved_lora")

# %%
tensors = {}
with safe_open(
    "interleaved_scoring/test2/grpo_saved_lora/adapter_model.safetensors",
    framework="pt",
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
        lora_request=model.load_lora("interleaved_scoring/test2/grpo_saved_lora"),
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
    "interleaved_scoring/test2/qwen_finetune_16bit",
    tokenizer,
    save_method="merged_16bit",
)

# # Merge to 4bit
# model.save_pretrained_merged("qwen_finetune_4bit", tokenizer, save_method = "merged_4bit",)

# Just LoRA adapters
model.save_pretrained("interleaved_scoring/test2/qwen_lora")
tokenizer.save_pretrained("interleaved_scoring/test2/qwen_lora")

model.save_pretrained_gguf(
    "interleaved_scoring/test2/qwen_finetune",
    tokenizer,
    quantization_method="f16",
)
model.save_pretrained_gguf(
    "interleaved_scoring/test2/qwen_finetune",
    tokenizer,
    quantization_method="bf16",
)
