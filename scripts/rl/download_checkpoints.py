# %%
import subprocess

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(override=True)

from qurating.constants import ROOT, DATA_DIR

checkpoints_dir = (
    "s3://qurating-checkpoints-183631302286-eu-west-2-an/checkpoints_grpo/"
)

checkpoint_dir_local = DATA_DIR / "grpo_checkpoints"

HERE = Path(__file__).resolve().parent

# %%
run_names = [
    "test1",
    "test2",
    "test3",
    "test4",
    "test5",
    "test6",
    "test7",
    "test8",
    "test9",
    "test10",
    "instruction_following",
    # "reasoning",
    # "interleaved_scoring",
    "gemma4",
    "qwen3",
]

s5cmd_path = str(ROOT / ".venv/bin/s5cmd")

for run_name in run_names:
    cmd = [s5cmd_path, "--json"]
    cmd += ["cp", "-n", "-s", "-u", "-sp"]
    cmd += [f"{checkpoints_dir}{run_name}/*", str(checkpoint_dir_local / f"{run_name}")]
    subprocess.run(cmd)
