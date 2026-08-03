# %%
import gc
from pathlib import Path
import re
from hashlib import shake_256

import jsonlines
import pandas as pd
import numpy as np
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from datasets import Dataset, load_from_disk
from vllm import SamplingParams

from qurating.constants import DATA_DIR

from chat_templates import ReasoningStudentMaterial

# Monkey-patch to add the missing attribute
if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(  # type: ignore
        lambda self: self.all_special_tokens
    )

HERE = Path(__file__).resolve().parent
DATA_DIR_GRPO_EVALS = DATA_DIR / "grpo_evals"

checkpoint_dir_local = DATA_DIR / "grpo_checkpoints"

max_seq_length = 2048
max_prompt_length = 256
max_lora_rank = 32

qwen3_response_formatter = re.compile(
    r"(.+?)<|endoftext|>.*", flags=re.DOTALL | re.MULTILINE
)

pad_token = "<|PAD_TOKEN|>"

DATASET = "original_full"


if DATASET == "interleaved":
    evals_dir = DATA_DIR / "education_evals"
    CONTENT_COLUMN = "content"
    GOOD_RESPONSE_COLUMN = "example_of_good_answer"
    TASK_COLUMN = "criterion"
    SUBTASK_COLUMN = "family"
    TOPIC_COLUMN = "category"
    INDEX_COLUMN = "index"
elif DATASET == "original":
    evals_dir = DATA_DIR / "education_evals_original"
    CONTENT_COLUMN = "Rendered Prompt"
    GOOD_RESPONSE_COLUMN = "Good Response"
    TASK_COLUMN = "Task Name"
    SUBTASK_COLUMN = "Subtask Name"
    TOPIC_COLUMN = "Topic"
    INDEX_COLUMN = "index"
elif DATASET == "original_full":
    evals_dir = DATA_DIR / "education_evals_original_full"
    CONTENT_COLUMN = "Rendered Prompt"
    GOOD_RESPONSE_COLUMN = "Good Response"
    TASK_COLUMN = "Task Name"
    SUBTASK_COLUMN = "Subtask Name"
    TOPIC_COLUMN = "Topic"
    INDEX_COLUMN = "index"
else:
    raise ValueError("DATASET must be one of 'interleaved', 'original', or 'original_full")

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
    # "test1": {
    #     "description": "core_ed-fl_teacher",
    #     "checkpoints": [
    #         "test1/outputs/checkpoint-1000",
    #         "test1/outputs/checkpoint-2000",
    #     ],
    #     "system_prompt": "/flteacher",
    #     "chat_template": "qwen-3",
    #     "group_idx": 0,
    # },
    # "test2": {
    #     "description": "core_ed-fl_teacher-no_ed_level",
    #     "checkpoints": [
    #         "test2/outputs/checkpoint-1000",
    #         "test2/outputs/checkpoint-2000",
    #     ],
    #     "system_prompt": "/flteacher",
    #     "chat_template": "qwen-3",
    #     "group_idx": 0,
    # },
    # "test3": {
    #     "description": "core_ed-fl_teacher-no_ed_level-length_target",
    #     "checkpoints": [
    #         "test3/outputs/checkpoint-1000",
    #         "test3/outputs/checkpoint-2000",
    #     ],
    #     "system_prompt": "/flteacher",
    #     "chat_template": "qwen-3",
    #     "group_idx": 0,
    # },
    # "test4": {
    #     "description": "core_ed_reduced-fl_teacher-no_ed_level-length_target",
    #     "checkpoints": [
    #         "test4/outputs/checkpoint-1000",
    #         "test4/outputs/checkpoint-2000",
    #     ],
    #     "system_prompt": "/flteacher",
    #     "chat_template": "qwen-3",
    #     "group_idx": 0,
    # },
    "test5": {
        "description": "core_ed_reduced-phonological_awareness-length_target",
        "checkpoints": [
            "test5/outputs/checkpoint-1000",
            "test5/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flteacher",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "test6": {
        "description": "core_ed_reduced-systematic_phonics-length_target",
        "checkpoints": [
            "test6/outputs/checkpoint-1000",
            "test6/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flteacher",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "test7": {
        "description": "core_ed_reduced-reading_fluency-length_target",
        "checkpoints": [
            "test7/outputs/checkpoint-1000",
            "test7/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flteacher",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "test8": {
        "description": "core_ed_reduced-reading_comprehension-length_target",
        "checkpoints": [
            "test8/outputs/checkpoint-1000",
            "test8/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flteacher",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "test9": {
        "description": "core_ed_reduced-writing_encoding-length_target",
        "checkpoints": [
            "test9/outputs/checkpoint-1000",
            "test9/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flteacher",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "test10": {
        "description": "core_ed_reduced-oral_language_vocabulary-length_target",
        "checkpoints": [
            "test10/outputs/checkpoint-1000",
            "test10/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flteacher",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "instruction_following/test1": {
        "description": "core_ed_reduced-writing_encoding-instruction_following",
        "checkpoints": [
            "instruction_following/test1/outputs/checkpoint-1000",
            "instruction_following/test1/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flteacher",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "instruction_following/test2": {
        "description": "core_ed_reduced-writing_encoding-instruction_following-v2",
        "checkpoints": [
            "instruction_following/test2/outputs/checkpoint-1000",
            "instruction_following/test2/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flteacher",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "instruction_following/test3": {
        "description": "core_ed_reduced-phonological_awareness-instruction_following",
        "checkpoints": [
            "instruction_following/test3/outputs/checkpoint-1000",
            "instruction_following/test3/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flteacher",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "instruction_following/test4": {
        "description": "core_ed_reduced-reading_comprehension-instruction_following",
        "checkpoints": [
            "instruction_following/test4/outputs/checkpoint-1000",
            "instruction_following/test4/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flteacher",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "instruction_following/test5": {
        "description": "core_ed_reduced-systematic_phonics-instruction_following",
        "checkpoints": [
            "instruction_following/test5/outputs/checkpoint-1000",
            "instruction_following/test5/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flteacher",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "instruction_following/test6": {
        "description": "core_ed_reduced-phonological_awareness-instruction_following",
        "checkpoints": [
            "instruction_following/test6/outputs/checkpoint-1000",
            "instruction_following/test6/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flteacher",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "instruction_following/test7": {
        "description": "core_ed_reduced-oral_language_vocabulary-instruction_following",
        "checkpoints": [
            "instruction_following/test7/outputs/checkpoint-1000",
            "instruction_following/test7/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flteacher",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "instruction_following/test8": {
        "description": "core_ed_reduced-reading_fluency-instruction_following",
        "checkpoints": [
            "instruction_following/test8/outputs/checkpoint-1000",
            "instruction_following/test8/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flteacher",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "instruction_following/test9": {
        "description": "core_ed_reduced-flteach_all-instruction_following",
        "checkpoints": [
            "instruction_following/test9/outputs/checkpoint-1000",
            "instruction_following/test9/outputs/checkpoint-2000",
        ],
        "system_prompt": "/flteacher",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "instruction_following/test10": {
        "description": "core_ed-instruction_following",
        "checkpoints": [
            "instruction_following/test10/checkpoint-1000",
            "instruction_following/test10/checkpoint-2000",
        ],
        "system_prompt": "/flteacher",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "instruction_following/test11": {
        "description": "core_ed-flteach_all-instruction_following",
        "checkpoints": [
            "instruction_following/test11/checkpoint-1000",
            "instruction_following/test11/checkpoint-2000",
        ],
        "system_prompt": "/flteacher",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    # "instruction_following_128/test1": {
    #     "description": "core_ed_reduced-systematic_phonics-instruction_following_rank-128",
    #     "checkpoints": [
    #         "instruction_following_128/test1/outputs/checkpoint-1000",
    #         "instruction_following_128/test1/outputs/checkpoint-2000",
    #     ],
    #     "system_prompt": "/flteacher",
    #     "chat_template": "qwen-3",
    #     "group_idx": 0,
    # },
    # "interleaved_scoring/test1": {
    #     "description": "core_ed_reduced-flteach_interleaved_scoring-instruction_following",
    #     "checkpoints": [
    #         "interleaved_scoring/test1/outputs/checkpoint-1000",
    #         "interleaved_scoring/test1/outputs/checkpoint-2000",
    #     ],
    #     "system_prompt": "/flteacher",
    #     "chat_template": "qwen-3",
    #     "group_idx": 0,
    # },
    # "interleaved_scoring/test2": {
    #     "description": "core_ed_reduced-flteach_interleaved_scoring_cap24-instruction_following",
    #     "checkpoints": [
    #         "interleaved_scoring/test2/outputs/checkpoint-1000",
    #         "interleaved_scoring/test2/outputs/checkpoint-2000",
    #     ],
    #     "system_prompt": "/flteacher",
    #     "chat_template": "qwen-3",
    #     "group_idx": 0,
    # },
    # "reasoning/test1": {
    #     "description": "reasoning-core_ed_fl_teacher-material-core_ed_fl_student",
    #     "checkpoints": [
    #         "reasoning/test1/outputs/checkpoint-1000",
    #         "reasoning/test1/outputs/checkpoint-2000",
    #     ],
    #     "system_prompt": ReasoningStudentMaterial().render_system_prompt(),
    #     "chat_template": ReasoningStudentMaterial().chat_template,
    #     "formatter": ReasoningStudentMaterial().formatter,
    #     "formatter_requires_tokenizer": True,
    #     "group_idx": {"reasoning": 0, "student_material": 2, "response": 3},
    # },
    "qwen3/test2": {
        "description": "reading_comprehension-instruction_following",
        "checkpoints": [
            "qwen3/test2/checkpoint-1000",
            "qwen3/test2/checkpoint-2000",
        ],
        "system_prompt": "/flteacher",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "qwen3/test3": {
        "description": "writing_encoding-instruction_following",
        "checkpoints": [
            "qwen3/test3/checkpoint-1000",
            "qwen3/test3/checkpoint-2000",
        ],
        "system_prompt": "/flteacher",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "qwen3/test4": {
        "description": "systematic_phonics-instruction_following",
        "checkpoints": [
            "qwen3/test4/checkpoint-1000",
            "qwen3/test4/checkpoint-2000",
        ],
        "system_prompt": "/flteacher",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "qwen3/test5": {
        "description": "oral_language_vocabulary-instruction_following",
        "checkpoints": [
            "qwen3/test5/checkpoint-1000",
            "qwen3/test5/checkpoint-2000",
        ],
        "system_prompt": "/flteacher",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "qwen3/test6": {
        "description": "phonological_awareness-instruction_following",
        "checkpoints": [
            "qwen3/test6/checkpoint-1000",
            "qwen3/test6/checkpoint-2000",
        ],
        "system_prompt": "/flteacher",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "qwen3/test7": {
        "description": "reading_fluency-instruction_following",
        "checkpoints": [
            "qwen3/test7/checkpoint-1000",
            "qwen3/test7/checkpoint-2000",
        ],
        "system_prompt": "/flteacher",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "qwen3/test8": {
        "description": "pedagogical_quality-instruction_following",
        "checkpoints": [
            "qwen3/test8/checkpoint-1000",
            "qwen3/test8/checkpoint-2000",
        ],
        "system_prompt": "/flteacher",
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
    "base_model": {
        "description": "Qwen3-4B-Base",
        "checkpoints": ["unsloth/Qwen3-4B-Base"],
        "chat_template": "qwen-3",
        "group_idx": 0,
    },
}


# %%
def hash_extra(row):
    return shake_256(
        (
            f"{row['rep']}"
            f"{row[CONTENT_COLUMN]}"
            f"{row[GOOD_RESPONSE_COLUMN]}"
            f"{row[TOPIC_COLUMN]}"
        ).encode("utf-8")
    ).hexdigest(8)

if DATASET == "interleaved":
    test_ds = load_from_disk(str(evals_dir / "flteach_grpo_dataset_train-test"))["test"]
    test_ds = test_ds.map(lambda row: {INDEX_COLUMN: f"{row['hash']}_{hash_extra(row)}"})
    test_ds = test_ds.select(range(1000))
elif DATASET == "original":
    test_ds = load_from_disk(
        str(evals_dir / "education_evals_combined_literacy_grade0-3_test.parquet")
    )
elif DATASET == "original_full":
    test_ds = load_from_disk(
        str(evals_dir / "education_evals_combined_literacy_grade0-3_test.parquet")
    )

prompts = list(test_ds[CONTENT_COLUMN])

print(test_ds)

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
if DATASET == "interleaved":
    cache_file = DATA_DIR_GRPO_EVALS / "GRPO_tests_flteach_grpo_dataset.jsonl"
elif DATASET == "original":
    cache_file = DATA_DIR_GRPO_EVALS / "GRPO_tests.jsonl"
elif DATASET == "original_full":
    cache_file = DATA_DIR_GRPO_EVALS / "GRPO_tests_full.jsonl"

    
res_list = []
if cache_file.exists():
    with jsonlines.open(cache_file) as reader:
        try:
            for obj in reader:
                res_list.append(obj)
        except:
            pass
with jsonlines.open(cache_file, "w") as writer:
    writer.write_all(res_list)


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

BATCH_SIZE = 40
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
            lora_request = model.load_lora(str(checkpoint_dir_local / checkpoint))

        system_prompt = test_config.get("system_prompt")
        resume_ds = test_ds.select(range(resume_point_ckpt, len(test_ds)))
        nbatches = int(np.ceil(len(resume_ds) / BATCH_SIZE))
        for batchi, batch_ds in enumerate(resume_ds.batch(batch_size=BATCH_SIZE)):
            print(
                f"Test Name: {test_name}; checkpoint: {checkpoint}; batch: {batchi} / {nbatches}"
            )

            prompts = batch_ds[CONTENT_COLUMN]
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
                outputs, prompts, texts, batch_ds[INDEX_COLUMN]
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

merge_cols = [TASK_COLUMN, SUBTASK_COLUMN, TOPIC_COLUMN, GOOD_RESPONSE_COLUMN]
res_df[merge_cols] = (
    test_df.set_index(INDEX_COLUMN).loc[res_df["prompt_index"], merge_cols].to_numpy()
)

pivot_df = res_df.pivot(
    columns=["test_name", "checkpoint", "description"],
    index=["prompt_index", "prompt", *merge_cols],
    values=["response"],
).reset_index().rename(columns={GOOD_RESPONSE_COLUMN: "Good Response"})


if DATASET == "interleaved":
    pivot_df.to_csv(DATA_DIR_GRPO_EVALS / "GRPO_tests_FLN_flteach_grpo_dataset.csv", encoding="utf_8_sig")
elif DATASET == "original":
    pivot_df.to_csv(DATA_DIR_GRPO_EVALS / "GRPO_tests_FLN.csv", encoding="utf_8_sig")
elif DATASET == "original_full":
    pivot_df.to_csv(DATA_DIR_GRPO_EVALS / "GRPO_tests_full_FLN.csv", encoding="utf_8_sig")

# %%
