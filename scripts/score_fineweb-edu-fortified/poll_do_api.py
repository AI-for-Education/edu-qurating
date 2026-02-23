# %%
import time
from datetime import datetime, UTC

import jsonlines
import digitalocean
from digitalocean.baseapi import DataReadError
from dotenv import load_dotenv

from qurating.constants import LOG_DIR

load_dotenv(override=True)


# %%
def create_node(
    name: str, size_slug: str = "gpu-h200x1-141gb", image: str = "215315195"
) -> tuple[bool, digitalocean.Droplet | dict]:
    manager = digitalocean.Manager()
    target_project_name = "Content Curation"
    do_project = [
        proj for proj in manager.get_all_projects() if proj.name == target_project_name
    ]
    if len(do_project) == 1:
        do_project = do_project[0]
    else:
        raise ValueError(f"Couldn't find project named {target_project_name}")

    regions = ["nyc2", "tor1", "ams3", "atl1", "sfo3"]
    region_idx = 0
    created = False
    errors = {}
    while not created:
        droplet = digitalocean.Droplet(
            name=name,
            size_slug=size_slug,
            image=image,
            region=regions[region_idx],
            ssh_keys=["3f:7b:15:32:65:f7:8d:7e:b5:1d:10:83:a6:d9:e4:2f"],
        )
        try:
            droplet.create()
            created = True
        except DataReadError as e:
            errors[regions[region_idx]] = str(e)
            created = False
            region_idx += 1
            if region_idx >= len(regions):
                return False, errors

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

    return True, dp


# %%
target_project_name = "Content Curation"

manager = digitalocean.Manager()

do_project = [
    proj for proj in manager.get_all_projects() if proj.name == target_project_name
]
if len(do_project) == 1:
    do_project = do_project[0]
else:
    raise ValueError(f"Couldn't find project named {target_project_name}")

droplet_sizes = [sz for sz in manager.get_all_sizes() if sz.slug.startswith("gpu-")]
for sz in droplet_sizes:
    print(f"{sz.slug} - available: {sz.available}")

droplet_snapshots = manager.get_all_snapshots()
for snapshot in droplet_snapshots:
    print(snapshot)

droplet_images = manager.get_all_images()
for image in droplet_images:
    if "1-Click".lower() in image.name.lower():
        print(image)

# %%
size_slug = "gpu-h100x8-640gb"
image = "194099177"

logfile = LOG_DIR / "api_polling" / f"{size_slug}_{image}.jsonl"
logfile.parent.mkdir(exist_ok=True, parents=True)

wait_time = 60
while True:
    with jsonlines.open(logfile, mode="a") as writer:
        resobj = {"size_slug": size_slug, "image": image}
        try:
            created, dp = create_node("test_droplet", size_slug=size_slug, image=image)
            if created:
                dp.destroy()
                writer.write(
                    {
                        **resobj,
                        "created": True,
                        "attempted_at": datetime.now(UTC).isoformat(timespec="seconds"),
                        "error": None
                    }
                )
            else:
                writer.write(
                    {
                        **resobj,
                        "created": False,
                        "attempted_at": datetime.now(UTC).isoformat(timespec="seconds"),
                        "error": dp
                    }
                )
        except Exception as e:
            writer.write(
                {
                    **resobj,
                    "created": False,
                    "attempted_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "error": str(e)
                }
            )
    time.sleep(wait_time)
