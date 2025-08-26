from typing import List, Literal
import os
import time
import json
from filelock import FileLock
import random

from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import openai
import tiktoken
from fdllm import LLMMessage, get_caller
from fdllm.llmtypes import LLMCaller
from pydantic import BaseModel


class ResponseFormat(BaseModel):
    choice: Literal["A", "B"]


# MODELS = {
#     "gpt-3.5-turbo-16k": {
#         "api_base": "https://pnlpopenai2.openai.azure.com/",
#         "api_type": "azure",
#         "api_version": "2023-05-15",
#         "deployment_name": "gpt-35-turbo-0613",
#         "enc": tiktoken.encoding_for_model("gpt-3.5-turbo"),
#         "prompt_cost_per_token": 0.003 / 1000,
#         "response_cost_per_token": 0.004 / 1000,
#     },
#     "gpt-3.5-turbo": {
#         "api_base": "https://pnlpopenai2.openai.azure.com/",
#         "api_type": "azure",
#         "api_version": "2023-05-15",
#         "deployment_name": "gpt-35-turbo-0613",
#         "enc": tiktoken.encoding_for_model("gpt-3.5-turbo"),
#         "prompt_cost_per_token": 0.0015 / 1000,
#         "response_cost_per_token": 0.002 / 1000,
#     },
#     "gpt-4": {
#         "api_base": "https://pnlpopenai3.openai.azure.com/",
#         "api_type": "azure",
#         "api_version": "2023-05-15",
#         "deployment_name": "gpt-4",
#         "enc": tiktoken.encoding_for_model("gpt-4"),
#         "prompt_cost_per_token": 0.03 / 1000,
#         "response_cost_per_token": 0.06 / 1000,
#     },
# }

RANDOM = random.Random()


async def aquery_model(
    prompt: str,
    model: str | LLMCaller,
    system_prompt: str = None,
    generations: int = 1,
    retries: int = 1,
    log_file_path: str = "openai_api_cost.jsonl",
) -> List[str]:
    if generations > 1:
        return [
            (
                await aquery_model(
                    prompt=prompt,
                    model=model,
                    system_prompt=system_prompt,
                    generations=1,
                    retries=retries,
                    log_file_path=log_file_path,
                )
            )[0]
            for _ in range(generations)
        ]
    #############################################################

    if isinstance(model, str):
        caller = get_caller(model)
    elif isinstance(model, LLMCaller):
        caller = model
    model = caller.Model.Name

    # enc = MODELS[model]["enc"]

    is_ok = False
    retry_count = 0

    messages = []
    if system_prompt is not None:
        messages.append(LLMMessage(Role="system", Message=system_prompt))
    messages.append(LLMMessage(Role="user", Message=prompt))

    while not is_ok:
        try:
            response = await caller.acall(
                messages,
                max_tokens=None,
                response_schema=ResponseFormat,
                temperature=None,
            )
            response_obj = ResponseFormat.model_validate_json(response.Message)
            choice = response_obj.choice
            is_ok = True
        except Exception as error:
            if "Please retry after" in str(error):
                timeout = (
                    int(str(error).split("Please retry after ")[1].split(" second")[0])
                    + 5 * RANDOM.random()
                )
                print(f"Wait {timeout}s before OpenAI API retry ({error})")
                time.sleep(timeout)
            elif retry_count < retries:
                print(f"OpenAI API retry for {retry_count} times ({error})")
                time.sleep(2)
                retry_count += 1
            else:
                print(f"OpenAI API failed for {retry_count} times ({error})")
                return []

    generations = [choice]

    usage = {
        "prompt_tokens": response.TokensUsed - response.TokensUsedCompletion,
        "completion_tokens": response.TokensUsedCompletion,
    }
    # usage["prompt_cost"] = (
    #     MODELS[model]["prompt_cost_per_token"] * usage["prompt_tokens"]
    # )
    # usage["response_cost"] = (
    #     MODELS[model]["response_cost_per_token"] * usage["completion_tokens"]
    # )
    # usage["cost"] = usage["prompt_cost"] + usage["response_cost"]
    usage["model"] = model

    with FileLock(log_file_path + ".lock"):
        with open(log_file_path, "a") as f:
            f.write(json.dumps(usage) + "\n")

    return generations


def query_model(
    prompt: str,
    model: str | LLMCaller,
    system_prompt: str = None,
    generations: int = 1,
    retries: int = 1,
    log_file_path: int = "openai_api_cost.jsonl",
) -> List[str]:
    if generations > 1:
        return [
            query_model(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                generations=1,
                retries=retries,
                log_file_path=log_file_path,
            )[0]
            for _ in range(generations)
        ]
    #############################################################

    if isinstance(model, str):
        caller = get_caller(model)
    elif isinstance(model, LLMCaller):
        caller = model
    model = caller.Model.Name

    # enc = MODELS[model]["enc"]

    is_ok = False
    retry_count = 0

    messages = []
    if system_prompt is not None:
        messages.append(LLMMessage(Role="system", Message=system_prompt))
    messages.append(LLMMessage(Role="user", Message=prompt))

    while not is_ok:
        try:
            response = caller.call(
                messages,
                max_tokens=None,
                response_schema=ResponseFormat,
                temperature=None,
            )
            response_obj = ResponseFormat.model_validate_json(response.Message)
            choice = response_obj.choice
            is_ok = True
        except Exception as error:
            if "Please retry after" in str(error):
                timeout = (
                    int(str(error).split("Please retry after ")[1].split(" second")[0])
                    + 5 * RANDOM.random()
                )
                print(f"Wait {timeout}s before OpenAI API retry ({error})")
                time.sleep(timeout)
            elif retry_count < retries:
                print(f"OpenAI API retry for {retry_count} times ({error})")
                time.sleep(2)
                retry_count += 1
            else:
                print(f"OpenAI API failed for {retry_count} times ({error})")
                return []

    generations = [choice]

    usage = {
        "prompt_tokens": response.TokensUsed - response.TokensUsedCompletion,
        "completion_tokens": response.TokensUsedCompletion,
    }
    # usage["prompt_cost"] = (
    #     MODELS[model]["prompt_cost_per_token"] * usage["prompt_tokens"]
    # )
    # usage["response_cost"] = (
    #     MODELS[model]["response_cost_per_token"] * usage["completion_tokens"]
    # )
    # usage["cost"] = usage["prompt_cost"] + usage["response_cost"]
    usage["model"] = model

    with FileLock(log_file_path + ".lock"):
        with open(log_file_path, "a") as f:
            f.write(json.dumps(usage) + "\n")

    return generations


def query_anthropic(prompt: str, model: str = "claude-2") -> List[str]:
    is_ok = False
    retry_count = 0

    while not is_ok:
        retry_count += 1
        try:
            anthropic = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            completion = anthropic.completions.create(
                model=model,
                max_tokens_to_sample=5,
                prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}",
            )
            return [completion.completion]
        except Exception as error:
            if retry_count <= 2:
                print(f"OpenAI API retry for {retry_count} times ({error})")
                time.sleep(2)
            else:
                print(f"OpenAI API failed for {retry_count} times ({error})")
                return []
