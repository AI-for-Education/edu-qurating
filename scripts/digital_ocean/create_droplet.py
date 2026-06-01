import time
from typing import Annotated
from pathlib import Path

import typer
import yaml
import digitalocean
from digitalocean.baseapi import DataReadError, JSONReadError, NotFoundError
from dotenv import load_dotenv

load_dotenv(override=True)

HERE = Path(__file__).resolve().parent

CONFIG_FILE = HERE / "crane_config.yaml"

app = typer.Typer()


def instantiate_manager(target_project_name: str, verbose: bool = True):

    manager = digitalocean.Manager()

    do_project = [
        proj for proj in manager.get_all_projects() if proj.name == target_project_name
    ]
    if len(do_project) == 1:
        do_project = do_project[0]
    else:
        raise ValueError(f"Couldn't find project named {target_project_name}")

    droplet_sizes = [sz for sz in manager.get_all_sizes() if sz.slug.startswith("gpu-")]
    droplet_snapshots = manager.get_all_snapshots()
    droplet_images = manager.get_all_images()

    if verbose:
        print("\nSizes:")
        for sz in droplet_sizes:
            print(f"{sz.slug} - available: {sz.available}")
        print("\nSnapshots:")
        for snapshot in droplet_snapshots:
            print(snapshot)
        print("\nImages:")
        for image in droplet_images:
            if "1-Click".lower() in image.name.lower():
                print(image)
        print()

    return manager, do_project


@app.command()
def main(
    config_file: Path = CONFIG_FILE,
    list_resources: Annotated[bool, typer.Option("--list-resources")] = False,
    destroy: str | None = None,
):

    if not config_file.exists():
        raise OSError(f"config_file: {config_file} doesn't exist")

    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    target_project_name = cfg["project_name"]
    manager, do_project = instantiate_manager(
        target_project_name=target_project_name, verbose=list_resources
    )
    if destroy is not None:
        try:
            dp = manager.get_droplet(destroy)
            dp.destroy()
        except NotFoundError:
            print(f"\ndroplet: {destroy} could not be found\n")
            return

    if list_resources:
        return

    name = cfg["droplet_name"]
    size_slug = cfg["size_slug"]
    image = cfg["image"]  # single GPU inference ready image

    ssh_keys = cfg["ssh_keys"]

    created_outer = False

    while not created_outer:
        regions = ["nyc2", "sfo3", "tor1", "ams3", "atl1"]
        region_idx = 0
        created = False
        errors = {}
        while not created:
            droplet = digitalocean.Droplet(
                name=name,
                size_slug=size_slug,
                image=image,
                region=regions[region_idx],
                ssh_keys=ssh_keys,
            )
            try:
                droplet.create()
                created = True
            except DataReadError as e:
                errors[regions[region_idx]] = e
                created = False
                region_idx += 1
                if region_idx >= len(regions):
                    print(f"{[f'{reg}: {e_}' for reg, e_ in errors.items()]}")
                    break
            except (JSONReadError, NotFoundError):
                continue
        if created:
            created_outer = True
        else:
            time.sleep(60)

        sleep_time = 30
        if created_outer:
            try:
                while True:
                    print(f"Checking droplet status: {droplet.name}")
                    dp = manager.get_droplet(droplet.id)
                    if dp.status is not None:
                        print(f"{dp.status}")
                        if dp.status == "active":
                            do_project.assign_resource([f"do:droplet:{droplet.id}"])
                            print("Success")
                            print(f"id: {dp.id}")
                            print(f"size: {dp.size_slug}")
                            print(f"ip: {dp.ip_address}")
                            break
                    else:
                        print("Not ready")
                        print(f"Checking again in {sleep_time}s")
                    time.sleep(sleep_time)
            except Exception:
                created_outer = False


if __name__ == "__main__":
    app()
