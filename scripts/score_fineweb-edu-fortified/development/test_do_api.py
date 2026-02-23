# %%
import time

import digitalocean
from digitalocean.baseapi import DataReadError
from dotenv import load_dotenv

load_dotenv(override=True)

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

# %%
droplet_sizes = [sz for sz in manager.get_all_sizes() if sz.slug.startswith("gpu-")]
for sz in droplet_sizes:
    print(
        f"{sz.slug} - available: {sz.available}"
    )

droplet_snapshots = manager.get_all_snapshots()
for snapshot in droplet_snapshots:
    print(snapshot)

droplet_images = manager.get_all_images()
for image in droplet_images:
    if "1-Click".lower() in image.name.lower():
        print(image)

# %%
regions = ["nyc2", "sfo3", "tor1", "ams3", "atl1"]
region_idx = 0
created = False
errors = {}
while not created:
    droplet = digitalocean.Droplet(
        name="test-droplet-1",
        size_slug="gpu-6000adax1-48gb",
        # size_slug="gpu-h100x8-640gb",
        image="214151830",
        # image="201061338",
        region=regions[region_idx],
        ssh_keys=["3f:7b:15:32:65:f7:8d:7e:b5:1d:10:83:a6:d9:e4:2f"],
    )
    try:
        droplet.create()
        created = True
    except DataReadError as e:
        errors[regions[region_idx]] = e
        created = False
        region_idx +=1
        if region_idx >= len(regions):
            for reg, e_ in errors.items():
                print(f"{reg}: {e_}")
            break

sleep_time = 3
if created:
    while True:
        print(f"Checking droplet status: {droplet.name}")
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

# %%
dp.destroy()