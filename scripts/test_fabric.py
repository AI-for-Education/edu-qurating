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
assert hosts is not None
assert key_filename is not None

hosts = hosts.split(";")

PROJECT_PATH = "~/projects/fab-qurating"
SCRIPT_PATH = "./scripts/score_fineweb-edu-fortified/score_fineweb-edu-fortified.py"
UV_PATH = "~/.local/bin/uv"
REV = "do-adapt-oli"


def run_node(host, start_partition, end_partition, n_partitions):
    with Connection(
        host, connect_kwargs={"key_filename": key_filename}, inline_ssh_env=True
    ) as conn:
        conn.config.run.env = {
            "AZURE_STORAGE_KEY": os.getenv("FABRIC_TEST_AZURE_STORAGE_KEY")
        }
        result = conn.run(
            " && ".join(
                [
                    "echo $0",
                    f"cd {PROJECT_PATH}",
                    "pwd",
                    "git remote -v",
                    "git remote remove origin",
                    "git remote add origin git@github.com:Fab-Inc/fab-qurating.git",
                    "git remote -v",
                    "ssh-keyscan -H github.com >> ~/.ssh/known_hosts",
                    "git fetch",
                    f"git branch --set-upstream-to=origin/{REV} {REV}",
                    f"git switch {REV}",
                    "git stash",
                    "git pull",
                    f"{UV_PATH} run dvc pull",
                    f"{UV_PATH} sync --group linux-gpu",
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