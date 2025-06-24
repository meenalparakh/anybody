#!/usr/bin/env bash

# in the case you need to load specific modules on the cluster, add them here
# e.g., `module load eth_proxy`

# to run this script, provide the following arguments:
# 1. task name (e.g., "train")
# 2. path to the anybody directory (e.g., "/home/user/anybody")
# 3. wandb key
# 4. arguments for the script "scripts/run.py"

# create job script with compute demands
### MODIFY HERE FOR YOUR JOB ###
mkdir slurm
cat <<EOT > slurm/job_$1.sh
#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=4048
#SBATCH --job-name="$1-$(date +"%Y-%m-%dT%H:%M")"
#SBATCH --output=slurm/outputs/$1_%j.txt \


bash "$2/docker/cluster/run_singularity.sh" "$3" "${@:4}"
EOT

sbatch < slurm/job_$1.sh
rm slurm/job_$1.sh
