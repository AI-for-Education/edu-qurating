# %%
import subprocess

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(override=True)

from qurating.constants import ROOT, DATA_DIR

checkpoints_dir = (
    "s3://qurating-checkpoints-183631302286-eu-west-2-an/checkpoints_grpo/"
)

HERE = Path(__file__).resolve().parent

# %%
# run_names = [
#     "test1",
#     "test2",
#     "test3",
#     "test4",
#     "test5",
#     "test6",
#     "test7",
#     "test8",
#     "test9",
#     "test10",
#     "reasoning",
#     "instruction_following",
#     "instruction_following_128",
#     "interleaved_scoring",
# ]

s5cmd_path = str(ROOT / ".venv/bin/s5cmd")
CHECKPOINTS_BASE = DATA_DIR / "grpo_checkpoints"

run_names = [d for d in CHECKPOINTS_BASE.glob("*") if d.is_dir()]
print(run_names)
for run_name in run_names:
    cmd = [s5cmd_path, "--json"]
    cmd += ["cp", "-s", "-u", "-sp"]
    cmd += [str(f"{run_name}/*"), f"{checkpoints_dir}{run_name}/"]
    subprocess.run(cmd)
