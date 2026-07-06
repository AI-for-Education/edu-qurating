# %%
import subprocess
from pathlib import Path

from dotenv import load_dotenv

from qurating.constants import DATA_DIR

ROOT = Path(__file__).resolve().parents[2]

load_dotenv(override=True)

s5cmd_path = str(ROOT / ".venv/bin/s5cmd")
cmd = [s5cmd_path, "--json"]
cmd += ["cp", "-s", "-u", "-sp"]
cmd += [
    f"{str(ROOT)}/logs/*",
    "s3://qurating-checkpoints-183631302286-eu-west-2-an/qurating-scoring-logs/",
]
print(" ".join(cmd))
output = subprocess.run(cmd)  # , capture_output=True)
