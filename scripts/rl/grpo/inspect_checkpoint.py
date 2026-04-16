# %%
import gc
from pathlib import Path
import re

import jsonlines
import pandas as pd
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import Dataset
from vllm import SamplingParams

from qurating.constants import DATA_DIR

from chat_templates import ReasoningStudentMaterial

evals_dir = DATA_DIR / "education_evals"

max_seq_length = 2048
max_prompt_length = 256
lora_rank = 128

qwen3_response_formatter = re.compile(
    r"(.+?)<|endoftext|>.*", flags=re.DOTALL | re.MULTILINE
)

pad_token = "<|PAD_TOKEN|>"


# %%
def parse_generated(generated, test_config, tokenizer):
    if test_config["formatter_requires_tokenizer"]:
        formatter = test_config["formatter"](tokenizer).match
    else:
        formatter = test_config["formatter"].match

    group_idx = test_config["group_idx"]
    match = formatter(generated)
    if match is None:
        return None
    groups = match.groups()
    if isinstance(group_idx, int):
        output = groups[group_idx]
    else:
        output = "\n\n".join(
            f"{key.upper()}\n{'#'*80}\n\n{groups[gidx]}"
            for key, gidx in group_idx.items()
        )
    return output


# %%
test_configs = {
    "test1": {
        "checkpoints": [
            "test1/outputs/checkpoint-1000",
            "test1/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flnteach",
        "chat_template": "qwen-3",
        "formatter_requires_tokenizer": False,
        "group_idx": 0,
    },
    "test2": {
        "checkpoints": [
            "test2/outputs/checkpoint-1000",
            "test2/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flnteach",
        "chat_template": "qwen-3",
        "formatter_requires_tokenizer": False,
        "group_idx": 0,
    },
    "test3": {
        "checkpoints": [
            "test3/outputs/checkpoint-1000",
            "test3/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flnteach",
        "chat_template": "qwen-3",
        "formatter_requires_tokenizer": False,
        "group_idx": 0,
    },
    "test4": {
        "checkpoints": [
            "test4/outputs/checkpoint-1000",
            "test4/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flnteach",
        "chat_template": "qwen-3",
        "formatter_requires_tokenizer": False,
        "group_idx": 0,
    },
    "test5": {
        "checkpoints": [
            "test5/outputs/checkpoint-1000",
            "test5/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flnteach",
        "chat_template": "qwen-3",
        "formatter_requires_tokenizer": False,
        "group_idx": 0,
    },
    "reasoning/test1": {
        "checkpoints": [
            "reasoning/test1/outputs/checkpoint-1000",
            "reasoning/test1/outputs/checkpoint-2000",
        ],
        "system_prompt": ReasoningStudentMaterial().render_system_prompt(),
        "chat_template": ReasoningStudentMaterial().chat_template,
        "formatter": ReasoningStudentMaterial().formatter,
        "formatter_requires_tokenizer": True,
        "group_idx": {"reasoning": 0, "student_material": 2, "response": 3},
    },
    "base_model": {
        "checkpoints": ["unsloth/Qwen3-4B-Base"],
        "chat_template": "qwen-3",
        "formatter_requires_tokenizer": False,
        "group_idx": 0,
    },
}

# %%
test_ds = Dataset.load_from_disk(
    str(evals_dir / "education_evals_combined_literacy_grade0-3_test.parquet")
)
test_ds

prompts = list(test_ds["Rendered Prompt"])

# %%
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Base",
    max_seq_length=max_seq_length,
    load_in_4bit=False,  # False for LoRA 16bit
    fast_inference=True,  # Enable vllm fast inference
    max_lora_rank=lora_rank,
)
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
)
# %%
cache_file = Path("GRPO_tests.jsonl")
res_list = []
if cache_file.exists():
    with jsonlines.open(cache_file) as reader:
        for obj in reader:
            res_list.append(obj)

resume_point = {}
for test_name, test_config in test_configs.items():
    resume_point[test_name] = {}
    for checkpoint in test_config["checkpoints"]:
        resume_point[test_name][checkpoint] = sum(
            True
            for rd in res_list
            if rd["test_name"] == test_name
            and rd["checkpoint"] == Path(checkpoint).name
        )

for test_name, test_config in test_configs.items():
    for checkpoint in test_config["checkpoints"]:
        resume_point_ckpt = resume_point[test_name][checkpoint]
        if resume_point_ckpt >= len(test_ds):
            continue
        chat_template = test_config["chat_template"]
        if isinstance(chat_template, str):
            tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")
        else:
            tokenizer.chat_template = chat_template()

        try:
            lora_request = model.load_lora(checkpoint)
        except Exception as e:
            lora_request = None
            
        system_prompt = test_config.get("system_prompt")
        resume_ds = test_ds.select(range(resume_point_ckpt, len(test_ds)))
        for batchi, batch_ds in enumerate(resume_ds.batch(batch_size=20)):
            print(f"Test Name: {test_name}; checkpoint: {checkpoint}; batch: {batchi}")

            prompts = batch_ds["Rendered Prompt"]
            texts = []
            for prompt in prompts:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,  # Must add for generation
                )
                texts.append(text)
            # tokenized = tokenizer(
            #     texts, return_tensors="pt", padding=True, padding_side="left"
            # ).to("cuda")

            sampling_params = SamplingParams(
                temperature=1.0,
                top_k=50,
                max_tokens=2048,
            )
            outputs = model.fast_generate(
                texts,
                sampling_params=sampling_params,
                lora_request=lora_request,
            )
            for response, prompt, text, prompt_index in zip(
                outputs, prompts, texts, batch_ds["index"]
            ):
                # response = response.replace(pad_token, "")
                response = response.outputs[0].text
                if response.startswith(text):
                    response = response[len(text) :]
                if test_config.get("formatter"):
                    parsed_response = parse_generated(response, test_config, tokenizer)
                else:
                    parsed_response = response
                print(f"\n{'#'*50}\n{parsed_response}\n{'#'*50}\n")
                row_dict = {
                    "test_name": test_name,
                    "checkpoint": Path(checkpoint).name,
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "response": parsed_response,
                }
                with jsonlines.open(cache_file, "a") as writer:
                    writer.write(row_dict)
                res_list.append(row_dict)

# %%
res_df = pd.DataFrame(res_list)
test_df = test_ds.to_pandas()
res_df["Good Response"] = (
    test_df.set_index("index").loc[res_df["prompt_index"], "Good Response"].to_list()
)

pivot_df = res_df.pivot(
    columns=["test_name", "checkpoint"],
    index=["prompt_index", "prompt", "Good Response"],
    values=["response"],
).reset_index(level=[0, 1, 2])


pivot_df.to_csv("GRPO_tests.csv", encoding="utf_8_sig")

# %%
