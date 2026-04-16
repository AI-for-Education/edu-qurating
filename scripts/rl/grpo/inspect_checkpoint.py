# %%
import gc
from pathlib import Path
import re

import pandas as pd
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import Dataset
from transformers import AutoTokenizer

from qurating.constants import DATA_DIR

from chat_templates import ReasoningStudentMaterial

evals_dir = DATA_DIR / "education_evals"

max_seq_length = 2048
max_prompt_length = 256
lora_rank = 32

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
        "formatter": qwen3_response_formatter,
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
        "formatter": qwen3_response_formatter,
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
        "formatter": qwen3_response_formatter,
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
        "formatter": qwen3_response_formatter,
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
}

# %%
test_ds = Dataset.load_from_disk(
    str(evals_dir / "education_evals_combined_literacy_grade0-3_test.parquet")
)
test_ds

prompts = list(test_ds["Rendered Prompt"])

# %%
base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Base",
    max_seq_length=max_seq_length,
    load_in_4bit=False,  # False for LoRA 16bit
    fast_inference=True,  # Enable vllm fast inference
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.8,  # Reduce if out of memory
)
base_tokenizer = get_chat_template(base_tokenizer, chat_template="qwen-3")

# %%
res_list = []
for test_name, test_config in test_configs.items():
    for checkpoint in test_config["checkpoints"]:
        model, tokenizer = FastLanguageModel.from_pretrained(model_name=checkpoint)
        chat_template = test_config["chat_template"]
        if isinstance(chat_template, str):
            tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")
        else:
            tokenizer.chat_template = chat_template()

        system_prompt = test_config["system_prompt"]

        for batchi, batch_ds in enumerate(test_ds.batch(batch_size=20)):
            print(f"Test Name: {test_name}; checkpoint: {checkpoint}; batch: {batchi}")

            prompts = batch_ds["Rendered Prompt"]
            texts = []
            for prompt in prompts:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,  # Must add for generation
                )
                texts.append(text)
            tokenized = tokenizer(
                texts, return_tensors="pt", padding=True, padding_side="left"
            ).to("cuda")

            output = model.generate(
                **tokenized,
                temperature=1,
                max_new_tokens=2048,
                # streamer=TextStreamer(tokenizer, skip_prompt=False),
            )
            output = tokenizer.batch_decode(output.tolist())
            for response, prompt, text, prompt_index in zip(
                output, prompts, texts, batch_ds["index"]
            ):
                response = response.replace(pad_token, "")
                if response.startswith(text):
                    response = response[len(text) :]
                parsed_response = parse_generated(response, test_config, tokenizer)
                print(f"\n{'#'*50}\n{parsed_response}\n{'#'*50}\n")
                row_dict = {
                    "test_name": test_name,
                    "checkpoint": Path(checkpoint).name,
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "response": parsed_response,
                }
                res_list.append(row_dict)

        del model
        gc.collect()
        torch.cuda.empty_cache()

# %%
## base model
for batchi, batch_ds in enumerate(test_ds.batch(batch_size=20)):
    print(f"Test Name: base_model; checkpoint: null; batch: {batchi}")

    prompts = batch_ds["Rendered Prompt"]
    texts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = base_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # Must add for generation
        )
        texts.append(text)
    tokenized = base_tokenizer(
        texts, return_tensors="pt", padding=True, padding_side="left"
    ).to("cuda")

    output = base_model.generate(
        **tokenized,
        temperature=1,
        max_new_tokens=2048,
        # streamer=TextStreamer(tokenizer, skip_prompt=False),
    )
    output = base_tokenizer.batch_decode(output.tolist())
    for response, prompt, text, prompt_index in zip(
        output, prompts, texts, batch_ds["index"]
    ):
        response = response.replace(pad_token, "")
        if response.startswith(text):
            response = response[len(text) :]
        parsed_response = parse_generated(
            response,
            {
                "formatter": qwen3_response_formatter,
                "formatter_requires_tokenizer": False,
                "group_idx": 0,
            },
            base_tokenizer,
        )
        row_dict = {
            "test_name": "base_model",
            "checkpoint": None,
            "prompt_index": prompt_index,
            "prompt": prompt,
            "response": parsed_response,
        }
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