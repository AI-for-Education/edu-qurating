# %%
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, UTC
import time

import digitalocean
from digitalocean.baseapi import DataReadError
from fabric import Connection
from invoke.watchers import StreamWatcher
from dotenv import load_dotenv

from qurating.constants import LOG_DIR

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


def create_and_run_node(start_partition, end_partition, n_partitions):
    logfile = (
        LOG_DIR / f"score_node_log_{start_partition}_{end_partition}_{n_partitions}.txt"
    )
    with open(logfile, "a+") as logf:
        node_name = (
            f"cc-qurating-scorer-{start_partition}-{end_partition}-{n_partitions}"
        )
        dp = create_node(node_name)
        if dp is None:
            print(f"FAILED TO CREATE NODE: {node_name}", file=logf)
            print("TRYING AGAIN", file=logf)
            dp = create_node(node_name)
        if dp is not None:
            try:
                run_node(
                    dp.ip_address, start_partition, end_partition, n_partitions, logf
                )
            except Exception:
                try:
                    run_node(
                        dp.ip_address,
                        start_partition,
                        end_partition,
                        n_partitions,
                        logf,
                    )
                except Exception as e:
                    dp.destroy()
                    print(e)


def create_node(name):
    manager = digitalocean.Manager()
    target_project_name = "Content Curation"
    do_project = [
        proj for proj in manager.get_all_projects() if proj.name == target_project_name
    ]
    if len(do_project) == 1:
        do_project = do_project[0]
    else:
        raise ValueError(f"Couldn't find project named {target_project_name}")

    regions = ["ams3", "atl1", "nyc2", "sfo3", "tor1"]
    region_idx = 0
    created = False
    while not created:
        droplet = digitalocean.Droplet(
            name=name,
            size_slug="gpu-h200x1-141gb",
            image="215315195",
            region=regions[region_idx],
            ssh_keys=["3f:7b:15:32:65:f7:8d:7e:b5:1d:10:83:a6:d9:e4:2f"],
        )
        try:
            droplet.create()
            created = True
        except DataReadError as e:
            created = False
            region_idx += 1
            if region_idx > len(regions):
                raise e

    sleep_time = 30
    if created:
        while True:
            print(f"Checking droplet status: {droplet.name}")
            try:
                dp = manager.get_droplet(droplet.id)
            except DataReadError:
                time.sleep(5)
                dp = manager.get_droplet(droplet.id)
            if dp.status is not None:
                print(f"{dp.status}")
                if dp.status == "active":
                    print("Success")
                    print(f"id: {dp.id}")
                    print(f"size: {dp.size_slug}")
                    print(f"ip: {dp.ip_address}")
                    break
                else:
                    print("Not ready")
                    print(f"Checking again in {sleep_time}s")
                    time.sleep(sleep_time)

    return dp


def run_node(host, start_partition, end_partition, n_partitions, logf):
    connectable = False
    max_retries = 10
    n_retry = 0
    while not connectable:
        try:
            with Connection(
                host, connect_kwargs={"key_filename": key_filename}, inline_ssh_env=True
            ) as conn:
                conn.run("pwd")
            connectable = True
            time.sleep(60)
        except Exception as e:
            if n_retry > max_retries:
                raise e
            n_retry += 1
            time.sleep(10)
    #################
    with Connection(
        host, connect_kwargs={"key_filename": key_filename}, inline_ssh_env=True
    ) as conn:

        class OutputWatcher(StreamWatcher):
            pos = 0

            def submit(self, stream):
                print(
                    f"{host} - {datetime.now(UTC).isoformat()}",
                    file=logf,
                )
                for i, ln in enumerate(stream.splitlines()):
                    if i >= self.pos:
                        print(f"{ln}", file=logf)
                        self.pos += 1
                return []

        conn.config.run.watchers = [OutputWatcher()]
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
                    "rm -fr ./fab-qurating",
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
n_partitions = 32
njobs = 1
with ThreadPoolExecutor(max_workers=njobs) as executor:
    futures = [
        executor.submit(
            create_and_run_node,
            start_partition=start_partition,
            end_partition=start_partition + 1,
            n_partitions=32,
        )
        for start_partition in range(njobs)
    ]
    res = [future.result() for future in as_completed(futures)]
