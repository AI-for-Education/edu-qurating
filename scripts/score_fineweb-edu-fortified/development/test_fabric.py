# %%
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from fabric import Connection, SerialGroup, ThreadingGroup
from dotenv import load_dotenv

load_dotenv(override=True)

# %%
user = "ogarrod"
hosts = os.getenv("FABRIC_TEST_HOSTS")
key_filename = os.getenv("FABRIC_TEST_HOST_KEY")
dvc_key = os.getenv("FABRIC_TEST_DVC_KEY")
assert hosts is not None
assert key_filename is not None

hosts = hosts.split(";")

PROJECT_PATH = "~/projects"
REPO_PATH = f"{PROJECT_PATH}/fab-qurating"
SCRIPT_PATH = "./scripts/score_fineweb-edu-fortified/score_fineweb-edu-fortified.py"
UV_PATH = "~/.local/bin/uv"
REV = "do-adapt-oli-run-minimal-data"


def run_node(host, start_partition, end_partition, n_partitions):
    with Connection(
        host, connect_kwargs={"key_filename": key_filename}, inline_ssh_env=True
    ) as conn:
        conn.config.run.env = {
            "AZURE_STORAGE_KEY": os.getenv("FABRIC_TEST_AZURE_STORAGE_KEY"),
            "HF_TOKEN": os.getenv("HF_TOKEN"),
        }
        result = conn.run(
            " && ".join(
                [
                    "echo $0",
                    'eval "$(ssh-agent -s)"',
                    f"cd {PROJECT_PATH}",
                    "pwd",
                    "ssh-keyscan -H github.com >> ~/.ssh/known_hosts",
                    "git clone git@github.com:Fab-Inc/fab-qurating.git",
                    f"cd {REPO_PATH}",
                    f"git switch {REV}",
                    "git pull",
                    f"{UV_PATH} sync --group linux-gpu",
                    f"{UV_PATH} run dvc remote modify --local azure account_key {dvc_key}",
                    f"{UV_PATH} run dvc pull",
                    ## run again to improve chance of getting everything
                    f"{UV_PATH} run dvc pull",
                    f"{UV_PATH} run python {SCRIPT_PATH} {start_partition} {end_partition} --n-partitions {n_partitions}",
                ]
            )
        )


# %%
njobs = len(hosts)
with ThreadPoolExecutor(max_workers=njobs) as executor:
    futures = [
        executor.submit(
            run_node,
            host=host,
            start_partition=start_partition,
            end_partition=start_partition + 1,
            n_partitions=32,
        )
        for start_partition, host in enumerate(hosts)
    ]
    res = [future.result() for future in as_completed(futures)]