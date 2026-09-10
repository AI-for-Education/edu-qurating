WHAT FOLLOWS IS A SET OF NOTES I TOOK WHEN PLANNING THIS PROCESS AND COPY/PASTED HERE. MOSTLY IT ENDED UP GOING AS PLANNED BUT THE PLAN IS INCOMPLETE AND HAS SOME DIFFERENCES TO WHAT WE ACTUALLY DID. IN THE END WE USED 32x SINGLE H200 MACHINES USING THE [[#Less complex automated]] IDEA LAID OUT IN THESE NOTES

## Context

**Single scoring pass of QuRating model on full FineWeb-Edu-Fortified dataset**  
Number of rows: 322,250,000  
Number of rows processed per second (single H200 GPU): 40  
&ensp;&ensp;&ensp;&ensp;_(Number of rows processed per second (single RTX6000Ada GPU): 10)_  
Number of hours to process all rows: 2238  
Cost per hour (single H200 GPU): $3.50  
**Total cost: $7,832**  
Total running time (8 x H200 GPU in parallel): 11.66 days

## Hardware options

#### Preferred option:
- 4 x H200x8 machines
#### Other options:
- N x H200x8 + (8 x (4-N)) x H200x1
- N x H200x8 + M x MI325x8 + (8 x (4-(N+M))) X H200x1
	- Update: just discovered this option isn't possible actually as MI325 are only on 12 month contracts

## Process design

#### Easiest to set up
- Manually assign different partitions of the FineWeb-Edu-Fortified dataset to each machine
	- This can be a script parameter or just an editable constant in the script
	- Dataset is naturally split into 95 different subsets (by crawl)
	- If we need extra granularity, each subset is split into ~20-50 shards
	- I think both levels of division allow random access for streaming dataset (need to confirm that for shards but pretty sure)
- This option is only really easy if we can get the preferred hardware configuration (4 x H200x8)
	- If we need to run on more machines (potentially up to 32 separate single H100 machines) then it becomes impractical

#### More complex but automated
- Set up a compute cluster
	- SLURM
	- Ray
	- Kubernetes?
	- Some combination?
- Resources:
	- SLURM
		- https://github.com/SergioMEV/slurm-for-dummies
		- https://slurm.schedmd.com/documentation.html
		- https://medium.com/@satishdotpatel/setup-slurm-cluster-for-hpc-bfa73364706a
	- Ray:
		- https://docs.ray.io/en/latest/cluster/getting-started.html
		- We would be "bare-metal machines":
			- https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html#on-prem
- Extra considerations:
	- VPC network?
	- Scripting the creation / destruction of machines? ^e5ca06
		- https://docs.digitalocean.com/reference/api/digitalocean/#tag/Droplets

#### Less complex automated
- Automated creation of droplets combined with automated ssh-ing into each one and launching script with partition parameter
- This would be controlled from either local machine at home or a cheap cloud node

- See above ([[#^e5ca06]]) for scripting the creation / destruction of machines
- Parameterisation of partitions
	- inputs:
		- start partition
		- end partition
	- output:
		- list of dataset subset strings (i.e. crawl dump IDs like "CC-MAIN-2024-10")

#### Common considerations
- Output data
	- Estimated size per row:
		- 6 floats + subset_id (16 chars) + row_id (47 chars) = ~111B
		- Per 1 million rows = ~111MB
	- Strategy:
		- Every 1m rows, upload new chunk per subset (e.g. "CC-MAIN-2024-10_chunk00", "CC-MAIN-2024-10_chunk01", etc.)

#### Plan of action
- Node-level process:
	- Get 1 H200 machines
	- Create some minimal test script that covers:
		- Parameterization of partitions
		- Loading the streaming datasets (i.e. Fineweb-Edu-Fortified subsets)
		- Saving results
		- Replicates intended process but over only 1 or 2 batches
		- Hard-code the partition parameters
	- Ideally get a 8xH200 (or 8xH100) to test the options for multi-GPU parallelism:
		- e.g.:
			- https://huggingface.co/docs/transformers/v4.47.0/en/perf_infer_gpu_multi
			- https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
- Orchestration process