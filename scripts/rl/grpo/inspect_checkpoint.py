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
max_lora_rank = 128

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
        "description": "core_ed-fl_teacher",
        "checkpoints": [
            "test1/outputs/checkpoint-1000",
            "test1/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flnteach",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "test2": {
        "description": "core_ed-fl_teacher-no_ed_level",
        "checkpoints": [
            "test2/outputs/checkpoint-1000",
            "test2/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flnteach",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "test3": {
        "description": "core_ed-fl_teacher-no_ed_level-length_target",
        "checkpoints": [
            "test3/outputs/checkpoint-1000",
            "test3/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flnteach",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "test4": {
        "description": "core_ed_reduced-fl_teacher-no_ed_level-length_target",
        "checkpoints": [
            "test4/outputs/checkpoint-1000",
            "test4/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flnteach",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "test5": {
        "description": "core_ed_reduced-phonological_awareness-length_target",
        "checkpoints": [
            "test5/outputs/checkpoint-1000",
            "test5/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flnteach",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "test6": {
        "description": "core_ed_reduced-systematic_phonics-length_target",
        "checkpoints": [
            "test6/outputs/checkpoint-1000",
            "test6/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flnteach",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "test7": {
        "description": "core_ed_reduced-reading_fluency-length_target",
        "checkpoints": [
            "test7/outputs/checkpoint-1000",
            "test7/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flnteach",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "test8": {
        "description": "core_ed_reduced-reading_comprehension-length_target",
        "checkpoints": [
            "test8/outputs/checkpoint-1000",
            "test8/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flnteach",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "test9": {
        "description": "core_ed_reduced-writing_encoding-length_target",
        "checkpoints": [
            "test9/outputs/checkpoint-1000",
            "test9/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flnteach",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "test10": {
        "description": "core_ed_reduced-oral_language_vocabulary-length_target",
        "checkpoints": [
            "test10/outputs/checkpoint-1000",
            "test10/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flnteach",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "instruction_following/test1": {
        "description": "core_ed_reduced-writing_encoding-instruction_following",
        "checkpoints": [
            "instruction_following/test1/outputs/checkpoint-1000",
            "instruction_following/test1/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flnteach",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "instruction_following/test2": {
        "description": "core_ed_reduced-writing_encoding-instruction_following-v2",
        "checkpoints": [
            "instruction_following/test2/outputs/checkpoint-1000",
            "instruction_following/test2/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flnteach",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "instruction_following/test3": {
        "description": "core_ed_reduced-phonological_awareness-instruction_following",
        "checkpoints": [
            "instruction_following/test3/outputs/checkpoint-1000",
            "instruction_following/test3/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flnteach",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "instruction_following/test4": {
        "description": "core_ed_reduced-reading_comprehension-instruction_following",
        "checkpoints": [
            "instruction_following/test4/outputs/checkpoint-1000",
            "instruction_following/test4/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flnteach",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "instruction_following/test5": {
        "description": "core_ed_reduced-systematic_phonics-instruction_following",
        "checkpoints": [
            "instruction_following/test5/outputs/checkpoint-1000",
            "instruction_following/test5/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flnteach",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "instruction_following/test6": {
        "description": "core_ed_reduced-phonological_awareness-instruction_following",
        "checkpoints": [
            "instruction_following/test6/outputs/checkpoint-1000",
            "instruction_following/test6/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flnteach",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "instruction_following/test7": {
        "description": "core_ed_reduced-oral_language_vocabulary-instruction_following",
        "checkpoints": [
            "instruction_following/test7/outputs/checkpoint-1000",
            "instruction_following/test7/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flnteach",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "instruction_following/test8": {
        "description": "core_ed_reduced-reading_fluency-instruction_following",
        "checkpoints": [
            "instruction_following/test8/outputs/checkpoint-1000",
            "instruction_following/test8/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flnteach",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "instruction_following_128/test1": {
        "description": "core_ed_reduced-systematic_phonics-instruction_following_rank-128",
        "checkpoints": [
            "instruction_following_128/test1/outputs/checkpoint-1000",
            "instruction_following_128/test1/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flnteach",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "reasoning/test1": {
        "description": "reasoning-core_ed_fl_teacher-material-core_ed_fl_student",
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
        "description": "Qwen3-4B-Base",
        "checkpoints": ["unsloth/Qwen3-4B-Base"],
        "chat_template": "qwen-3",
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
base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Base",
    max_seq_length=max_seq_length,
    load_in_4bit=False,  # False for LoRA 16bit
    fast_inference=True,  # Enable vllm fast inference
    max_lora_rank=max_lora_rank,
)
base_tokenizer = get_chat_template(base_tokenizer, chat_template="qwen-3")

# %%
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Base",
    max_seq_length=max_seq_length,
    load_in_4bit=False,  # False for LoRA 16bit
    fast_inference=True,  # Enable vllm fast inference
    max_lora_rank=max_lora_rank,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=max_lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
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
        if test_name == "base_model":
            tokenizer = base_tokenizer
        else:
            if isinstance(chat_template, str):
                tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
            else:
                tokenizer.chat_template = chat_template()
            lora_request = model.load_lora(checkpoint)

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
            if test_name == "base_model":
                tokenized = tokenizer(
                    texts, return_tensors="pt", padding=True, padding_side="left"
                ).to("cuda")
                outputs = base_model.generate(
                    **tokenized, temperature=1.0, top_k=50, max_new_tokens=2048
                )
                outputs = tokenizer.batch_decode(outputs.tolist())
            else:
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
                if not isinstance(response, str):
                    response = response.outputs[0].text
                else:
                    response = response.replace(pad_token, "").replace(
                        tokenizer.eos_token, ""
                    )
                    if response.startswith(text):
                        response = response[len(text) :]
                if test_config.get("formatter"):
                    parsed_response = parse_generated(response, test_config, tokenizer)
                else:
                    parsed_response = response
                print(f"\n{'#'*50}\n{parsed_response}\n{'#'*50}\n")
                row_dict = {
                    "test_name": test_name,
                    "description": test_config["description"],
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

merge_cols = ["Task Name", "Subtask Name", "Topic", "Good Response"]
res_df[merge_cols] = (
    test_df.set_index("index").loc[res_df["prompt_index"], merge_cols].to_numpy()
)

pivot_df = res_df.pivot(
    columns=["test_name", "checkpoint", "description"],
    index=["prompt_index", "prompt", *merge_cols],
    values=["response"],
).reset_index()


pivot_df.to_csv("GRPO_tests_FLN.csv", encoding="utf_8_sig")

# %%
