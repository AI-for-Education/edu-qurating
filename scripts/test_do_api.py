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
print(droplet_snapshots)

# %%
droplet = digitalocean.Droplet(
    name="test-droplet-1",
    size_slug="gpu-h200x1-141gb",
    image="214386455",
    region="nyc2",
    ssh_keys=["3f:7b:15:32:65:f7:8d:7e:b5:1d:10:83:a6:d9:e4:2f"],
)

try:
    droplet.create()
    created = True
except DataReadError as e:
    created = False
    print(e)

sleep_time = 3
if created:
    while True:
        print(f"Checking droplet status: {droplet.name}")
        dp = manager.get_droplet(droplet.id)
        if dp.status is not None:
            print(f"{dp.status}")
            if dp.status == "active":
                print(f"Success, destroying droplet in {sleep_time}s")
                time.sleep(sleep_time)
                dp.destroy()
                break
        else:
            print("Not ready")
            print(f"Checking again in {sleep_time}s")
            time.sleep(sleep_time)
