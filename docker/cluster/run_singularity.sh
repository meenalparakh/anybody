#!/usr/bin/env bash

echo "(run_singularity.py): Called on compute node from current isaaclab directory wandb key and arguments ${@:2}"

#==
# Helper functions
#==
# Clusters often have /scratch for this purpose. 
# Check if /scratch exists, otherwise use /tmp/$USER

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

if [ -d "/scratch" ]; then
    WORK_DIR="/scratch/$USER/$SLURM_JOB_ID"
else
    WORK_DIR="/tmp/$USER/"
    echo "Using /tmp/$USER as the working directory since /scratch does not exist."
fi

# clean the working directory if it exists
if [ -d "$WORK_DIR" ]; then
    echo "Deleting existing working directory: $WORK_DIR"
    rm -rf "$WORK_DIR"
fi

mkdir -p "$WORK_DIR"

# caches cause issues with multiple jobs running at the same time
# define cache dir uniquely if possible: base cache dir + job id
CLUSTER_ISAAC_SIM_CACHE_DIR="${WORK_DIR}/docker-isaac-sim"

setup_directories() {
    # Check and create directories
    for dir in \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/kitdata" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/nv_shadercache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/kit" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/ov" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/pip" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/glcache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/computecache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/logs" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/data" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/documents"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo "Created directory: $dir"
        fi
    done
}


#==
# Main
#==


# get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# load variables to set the Isaac Lab path on the cluster
source $SCRIPT_DIR/.env.cluster
source $SCRIPT_DIR/../.env.base

# make sure that all directories exists in cache directory
setup_directories


touch "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/kitdata/user.config.json"

# create a Kit/shared directory in documents if it does not exist
if [ ! -d "${CLUSTER_ISAAC_SIM_CACHE_DIR}/documents/Kit/shared" ]; then
    mkdir -p "${CLUSTER_ISAAC_SIM_CACHE_DIR}/documents/Kit/shared"
    echo "Created directory: ${CLUSTER_ISAAC_SIM_CACHE_DIR}/documents/Kit/shared"
fi



# copy all cache files
# cp -r $CLUSTER_ISAAC_SIM_CACHE_DIR $WORK_DIR

# make sure logs directory exists (in the permanent isaaclab directory)
mkdir -p "$CLUSTER_ROOT_DIR/logs"
touch "$CLUSTER_ROOT_DIR/logs/.keep"

mkdir -p "$CLUSTER_PROJECT_DIR/anybody/envs/problem_specs"
touch "$CLUSTER_PROJECT_DIR/anybody/envs/problem_specs/.keep"

# copy the temporary isaaclab directory with the latest changes to the compute node
cp -r $CLUSTER_PROJECT_DIR $WORK_DIR
# Get the directory name
dir_name=$(basename "$CLUSTER_PROJECT_DIR")

# copy container to the compute node
tar -xf $CLUSTER_SIF_PATH/isaac-lab-anybody.tar  -C $WORK_DIR

# execute command in singularity container
# NOTE: ISAACLAB_PATH is normally set in `isaaclab.sh` but we directly call the isaac-sim python because we sync the entire
# Isaac Lab directory to the compute node and remote the symbolic link to isaac-sim

singularity_cmd=(
    singularity exec
    -B $CLUSTER_ISAAC_SIM_CACHE_DIR/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw 
    -B $CLUSTER_ISAAC_SIM_CACHE_DIR/cache/kitdata:${DOCKER_ISAACSIM_ROOT_PATH}/kit/data:rw 
    -B $CLUSTER_ISAAC_SIM_CACHE_DIR/cache/nv_shadercache:${DOCKER_ISAACSIM_ROOT_PATH}/kit/exts/omni.gpu_foundation/cache/nv_shadercache:rw 
    -B $CLUSTER_ISAAC_SIM_CACHE_DIR/cache/ov:${DOCKER_USER_HOME}/.cache/ov:rw 
    -B $CLUSTER_ISAAC_SIM_CACHE_DIR/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw 
    -B $CLUSTER_ISAAC_SIM_CACHE_DIR/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw 
    -B $CLUSTER_ISAAC_SIM_CACHE_DIR/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw 
    -B $CLUSTER_ISAAC_SIM_CACHE_DIR/logs:${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw 
    -B $CLUSTER_ISAAC_SIM_CACHE_DIR/data:${DOCKER_USER_HOME}/.local/share/ov/data:rw 
    -B $CLUSTER_ISAAC_SIM_CACHE_DIR/documents:${DOCKER_USER_HOME}/Documents:rw 
    -B $WORK_DIR/$dir_name:/workspace/anybody:rw 
    -B ${CLUSTER_PROJECT_DIR}/anybody/envs/problem_specs:/workspace/anybody/anybody/envs/problem_specs:rw 
    -B $CLUSTER_ROOT_DIR/logs:/workspace/logs:rw 
    --bind /usr/bin/scontrol:/usr/bin/scontrol 
    --bind /usr/lib64/slurm:/usr/lib64/slurm 
    --bind /dev/null:/var/run/nvidia-persistenced/socket 
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES 
    --env SLURM_JOB_ID=$SLURM_JOB_ID 
    --env SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST 
    --env CHECK_OVERHEAT_FOLDER="/workspace/check_overheat" 
    --env ISAACLAB_PATH=/workspace/anybody/isaaclab 
    --env PROJECT_PATH=/workspace/anybody 
    --env WANDB_API_KEY=$1
    --nv --containall
)

# Conditionally bind CLUSTER_OVERHEAT_CHECK_DIR
if [[ -n "$CLUSTER_OVERHEAT_CHECK_DIR" ]]; then
    singularity_cmd+=(--bind ${CLUSTER_OVERHEAT_CHECK_DIR}:/workspace/check_overheat:rw)
fi

    
# Add the SIF file and the command to run

singularity_cmd+=("$WORK_DIR/isaac-lab-anybody.sif")
singularity_cmd+=("bash" "-c" "cd /workspace/anybody && /isaac-sim/python.sh ${CLUSTER_PYTHON_EXECUTABLE} ${@:2}")

# copy resulting cache files back to host
# rsync -azPv $WORK_DIR/docker-isaac-sim $CLUSTER_ISAAC_SIM_CACHE_DIR/..

"${singularity_cmd[@]}"

echo "(run_singularity.py): Return"
