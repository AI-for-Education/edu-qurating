# %%
from pathlib import Path
from itertools import chain
import logging
import os

from dotenv import load_dotenv
from unsloth import FastLanguageModel, FastVisionModel
from unsloth.chat_templates import get_chat_template
from trl import GRPOConfig, GRPOTrainer  # type: ignore
import numpy as np
from datasets import Dataset, load_from_disk
from fdllm import register_models, get_caller, LLMMessage
from safetensors import safe_open
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from vllm import SamplingParams
from pydantic import BaseModel, Field
import yaml

from qr_reward import QuratingReward
from qurating.constants import DATA_DIR, ROOT

# Monkey-patch to add the missing attribute
if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(  # type: ignore
        lambda self: self.all_special_tokens
    )

load_dotenv(override=True)
register_models(ROOT / "custom_models.yaml")

USE_CFG = "qwen35/test2"
NUM_GENERATIONS = 2
PER_DEVICE_TRAIN_BATCH_SIZE = 2
USE_WANDB = True

HERE = Path(__file__).resolve().parent
CHECKPOINTS_DIR = DATA_DIR / "grpo_checkpoints"
CHECKPOINTS_DIR.mkdir(exist_ok=True, parents=True)

### load test config
with open(HERE / "grpo_configs.yaml") as f:
    cfg = yaml.safe_load(f)

if USE_CFG not in cfg:
    raise ValueError(f"{USE_CFG} is not a valid test config")
cfg = cfg[USE_CFG]

os.environ["WANDB_PROJECT"] = "cc-qurating-rl"
os.environ["WANDB_NAME"] = USE_CFG

max_seq_length = 2048
max_prompt_length = 256
lora_rank = cfg["lora_rank"]

chat_template_name = cfg.get("chat_template_name", None)

system_prompt = cfg.get("system_prompt", """/flteacher""")

dataset_variant = cfg["dataset"]

if dataset_variant == "interleaved":
    evals_dir = DATA_DIR / "education_evals"
    CONTENT_COLUMN = "content"
    GOOD_RESPONSE_COLUMN = "example_of_good_answer"
elif dataset_variant == "original":
    evals_dir = DATA_DIR / "education_evals_original"
    CONTENT_COLUMN = "Rendered Prompt"
    GOOD_RESPONSE_COLUMN = "Good Response"
else:
    raise ValueError("DATASET must be one of 'interleaved' or 'original")

if "qwen3-" in cfg["base_model"].lower():
    FastModel = FastLanguageModel
    fast_inference = True
    learning_rate = 5e-6
    per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE
    max_steps = 2000
    save_steps = 100
    extra_grpo_kwargs = {}
    extra_peft_kwargs = {}
elif "qwen3.5-" in cfg["base_model"].lower():
    FastModel = FastVisionModel
    fast_inference = False
    learning_rate = 5e-6
    per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE
    max_steps = 2000
    save_steps = 100
    extra_grpo_kwargs = {}
    extra_peft_kwargs = {"finetune_vision_layers": False}
elif "gemma-4" in cfg["base_model"].lower():
    FastModel = FastVisionModel
    fast_inference = False
    learning_rate = 5e-5
    per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE
    max_steps = 2000
    save_steps = 100
    extra_grpo_kwargs = {}
    extra_peft_kwargs = {"finetune_vision_layers": False}
elif "gemma-3" in cfg["base_model"].lower():
    chat_template_name = "gemma-3"
    FastModel = FastModelUS
    fast_inference = False
    learning_rate = 5e-6
    per_device_train_batch_size = 4
    max_steps = 2000
    save_steps = 100
    extra_grpo_kwargs = {}
    extra_peft_kwargs = {}
    # extra_peft_kwargs = {"finetune_vision_layers": False}
    # extra_grpo_kwargs = dict(
    #     epsilon = 0.2,
    #     epsilon_high = 0.28, # one sided
    #     delta = 1.5, # two sided
    #     loss_type = 'bnpo',
    #     mask_truncated_completions = True
    # )
else:
    raise ValueError("base_model must be in qwen-3 or gemma-4 families")


# hide warning about processor_kwargs from transformers v5
logging.getLogger("transformers.processing_utils").setLevel(logging.ERROR)


# %%
# define and store extra non-qurating reward funs
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

verbose = False

def correctness_reward_half(prompts, completions, **kwargs):
    return correctness_reward(prompts, completions, **kwargs, scale=1.5)

def correctness_reward(prompts, completions, scale=3.0, **kwargs):
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
        if verbose:
            print(prompt)
            print(resp)
            print(out_obj.model_dump_json(indent=2))
        score = (
            float(out_obj.length_match)
            + float(out_obj.format_match)
            + float(out_obj.instruction_following)
        )
        scores.append(score * scale)
    return scores


def length_target(completions, **kwargs):
    assert GOOD_RESPONSE_COLUMN in kwargs
    resp_len = np.array([len(completion[0]["content"]) for completion in completions])
    good_len = np.array([len(gr) for gr in kwargs[GOOD_RESPONSE_COLUMN]])
    max_len = max_seq_length
    print(resp_len)
    print(good_len)
    print(max_len)
    len_score = 5 * np.maximum((1 - np.abs(resp_len - good_len) / max_len), 0) ** 0.5
    return len_score.tolist()


extra_functions = {
    "length_target": length_target,
    "correctness_reward": correctness_reward,
    "correctness_reward_half": correctness_reward_half,
}


# instantiate qurating reward functions generator object
qr_reward = QuratingReward()


# initialise preset score specs for $all and $all_interleaved
def preset_score_specs(preset_label, weight):
    preset_score_spec_dict = {
        "fl_teacher": {
            "all": {key: weight for key in qr_reward.labels["fl_teacher"]},
            "all_interleaved": {
                (
                    "1_oral_language_vocabulary",
                    (("criterion", "pairwise_1_oral_language_vocabulary"),),
                ): weight,
                (
                    "2_phonological_awareness",
                    (("criterion", "pairwise_2_phonological_awareness"),),
                ): weight,
                (
                    "3_systematic_phonics",
                    (("criterion", "pairwise_3_systematic_phonics"),),
                ): weight,
                (
                    "4_reading_fluency",
                    (("criterion", "pairwise_4_reading_fluency"),),
                ): weight,
                (
                    "5_reading_comprehension",
                    (("criterion", "pairwise_5_reading_comprehension"),),
                ): weight,
                (
                    "6_writing_encoding",
                    (("criterion", "pairwise_6_writing_encoding"),),
                ): weight,
                (
                    "7_pedagogical_quality",
                    (("criterion", "pairwise_7_pedagogical_quality"),),
                ): weight,
            },
        }
    }
    return preset_score_spec_dict["fl_teacher"][preset_label]


#### instantiate reward function from config
# first qr_reward fun spec we encounter, we set to be verbose, then we set the rest to be non-verbose
# NOTE: Update - changed this to always be False. Using log completions instead
verbose = False
qr_reward_funs = {}
for reward_cfg in cfg["reward"]["qurating"]:
    name = reward_cfg["name"]
    score_cap = reward_cfg["score_cap"]
    if score_cap[0] is None:
        score_cap[0] = -np.inf
    if score_cap[1] is None:
        score_cap[1] = np.inf
    spec = reward_cfg["spec"]
    use_spec = {}
    for key, val in spec.items():
        if len(val) == 1 and next(iter(val)).startswith("$"):
            preset_label, weight = next(iter(val.items()))
            if key != "fl_teacher":
                raise ValueError(
                    "preset_score_spec only supported for 'fl_teacher' qurating model"
                )
            use_spec[key] = preset_score_specs(preset_label[1:], weight)
        else:
            use_spec[key] = val
    reward_fun = qr_reward.reward_fun_generator(
        use_spec, name=name, score_cap=tuple(score_cap), verbose=verbose
    )
    # after the first one, the rest are not verbose
    verbose = False
    qr_reward_funs[name] = reward_fun

extra_reward_funs = {
    name: extra_functions[name] for name in cfg["reward"]["extra_functions"]
}

all_reward_funs = dict(chain(qr_reward_funs.items(), extra_reward_funs.items()))

# %%
if dataset_variant == "interleaved":
    dataset = load_from_disk(str(evals_dir / "flteach_grpo_dataset_train-test"))[
        "train"
    ]
elif dataset_variant == "original":
    dataset = load_from_disk(
        str(evals_dir / "education_evals_combined_literacy_grade0-3_train.parquet")
    )
assert isinstance(dataset, Dataset)

print(dataset)

dataset = dataset.map(
    lambda x: {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": x[CONTENT_COLUMN]},
        ]
    }
)
print(dataset[0])

# %%
if dataset_variant == "interleaved":
    test_ds = load_from_disk(str(evals_dir / "flteach_grpo_dataset_train-test"))["test"]
elif dataset_variant == "original":
    test_ds = load_from_disk(
        str(evals_dir / "education_evals_combined_literacy_grade0-3_test.parquet")
    )

prompts = list(test_ds[CONTENT_COLUMN])  # type: ignore
test_promptiter = iter(prompts)

print(test_ds)

# %%
### check that all reward functions run

for reward_fun in all_reward_funs.values():
    reward_fun(
        prompts=[dataset[0]["prompt"]] * 3,
        completions=[
            [{"role": "assistant", "content": dataset[0][GOOD_RESPONSE_COLUMN]}]
        ]
        * 3,
        **{GOOD_RESPONSE_COLUMN: [dataset[0][GOOD_RESPONSE_COLUMN]] * 3},
        criterion=[
            "pairwise_1_oral_language_vocabulary",
            "pairwise_2_phonological_awareness",
            "pairwise_3_systematic_phonics",
        ],
    )


# %%
model, tokenizer = FastModel.from_pretrained(
    model_name=cfg["base_model"],
    max_seq_length=max_seq_length,
    load_in_4bit=False,  # False for LoRA 16bit
    fast_inference=fast_inference,  # Enable vllm fast inference
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.8,  # Reduce if out of memory
)
if chat_template_name is not None:
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template_name)

model = FastModel.get_peft_model(
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
    **extra_peft_kwargs,  # type: ignore
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
tokenized = dataset.map(
    lambda x: {
        "tokens": tokenizer(
            text=tokenizer.apply_chat_template(
                x["prompt"], add_generation_prompt=True, tokenize=False
            )
        )["input_ids"]
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
# max_prompt_length = maximum_length + 1  # + 1 just in case!
max_completion_length = max_seq_length - max_prompt_length

grpo_kwargs = dict(
    temperature=1.0,
    learning_rate=learning_rate,
    weight_decay=0.001,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    logging_steps=10,
    log_completions=True,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps = int(np.ceil(16 / (NUM_GENERATIONS * PER_DEVICE_TRAIN_BATCH_SIZE))),  # Increase to 4 for smoother training
    num_generations=NUM_GENERATIONS,  # Decrease if out of memory
    # max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    num_train_epochs=1,  # Set to 1 for a full training run
    max_steps=max_steps,
    save_steps=save_steps,
    report_to="wandb" if USE_WANDB else "none",  # Can use Weights & Biases
    output_dir=str(CHECKPOINTS_DIR / USE_CFG),
    generation_kwargs={
        "stop": [tokenizer.eos_token],
        "include_stop_str_in_output": True,
    },
    **extra_grpo_kwargs,
    # For optional training + evaluation
    # fp16_full_eval = True,
    # per_device_eval_batch_size = 4,
    # eval_accumulation_steps = 1,
    # eval_strategy = "steps",
    # eval_steps = 1,
)
if fast_inference:
    grpo_kwargs = {
        **grpo_kwargs,
        **dict(min_p=0.1, top_p=1.0, top_k=-1),
    }
# add in settings from config if they exist
for key, val in cfg.get("grpo_kwargs", {}).items():
    if (
        isinstance(val, dict)
        and key in grpo_kwargs
        and isinstance(grpo_kwargs[key], dict)
    ):
        grpo_kwargs[key] = {**grpo_kwargs[key], **val} # type: ignore
    else:
        grpo_kwargs[key] = val
training_args = GRPOConfig(**grpo_kwargs)  # type: ignore

# For optional training + evaluation
# new_dataset = dataset.train_test_split(test_size = 0.01)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=list(all_reward_funs.values()),
    args=training_args,
    train_dataset=dataset,
    # For optional training + evaluation
    # train_dataset = new_dataset["train"],
    # eval_dataset = new_dataset["test"],
)

# %%
trainer.train()

# %%
model.save_lora(str(CHECKPOINTS_DIR / USE_CFG / "grpo_saved_lora"))

# %%
tensors = {}
with safe_open(
    str(CHECKPOINTS_DIR / USE_CFG / "grpo_saved_lora" / "adapter_model.safetensors"),
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
        lora_request=model.load_lora(
            str(CHECKPOINTS_DIR / USE_CFG / "grpo_saved_lora")
        ),
    )[0]
    .outputs[0]
    .text
)

print(prompt)
print("-" * 50)
print(output)
print("#" * 50)
