
## 🐳 Running on Cluster with Singularity

This guide explains how to run the benchmark on a compute cluster using Singularity (Apptainer). This guide explains process involves building a Docker image, converting it to a Singularity image, and running jobs on the cluster.

---

### 1. Build Docker Image

We base our Docker image on the NVIDIA Isaac Sim image. For detailed steps, refer to the [Isaac Lab Docker guide](https://isaac-sim.github.io/IsaacLab/main/source/deployment/index.html).

> **Note:** Some clusters require Docker images built with specific Docker versions for compatibility with Singularity. Hence, you may need to manually install the required Docker version and components.

**Prerequisites:**
- [Docker and Docker Compose](https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html#docker-and-docker-compose) (check version requirements).

**Build and Test the Docker Image:**

```bash
# Build and start the Docker container
./docker/container.py start

# Enter the running container
./docker/container.py enter anybody

# Inside the container, test the environment (headless mode)
python scripts/env_utils/load_env.py --headless BENCHMARK_TASK intra_simple_bot_reach

# If the simulator GUI crashes on first run, try again:
python scripts/env_utils/load_env.py BENCHMARK_TASK intra_simple_bot_reach
```

---

### 2. Build Singularity (Apptainer) Image

1. **Install Apptainer:**  
  Follow the instructions [here](https://isaac-sim.github.io/IsaacLab/main/source/deployment/cluster.html#setup-instructions).

2. **Configure Cluster Environment:**  
  In `anybody/docker/cluster`, rename `.env.cluster.template` to `.env.cluster` and update the paths as needed. See the [Isaac Lab cluster configuration guide](https://isaac-sim.github.io/IsaacLab/main/source/deployment/cluster.html#configuring-the-cluster-parameters) for details.

3. **Export and Push the Image:**  
  This command builds the Singularity image and pushes it to the remote cluster (authentication may be required):

  ```bash
  ./docker/cluster/cluster_interface.sh push anybody
  ```

---

### 3. Run Interactive Apptainer Shell

To test the image interactively on the cluster:

1. Change to the `anybody` directory (ensure robot morphs are also downloaded, or pre-generated).
2. Run:

  ```bash
  # wandb key needed only for run.py script
  ./docker/cluster/run_singularity_shell.sh <WANDB_API_KEY>
  ```

3. In the Apptainer shell:

  ```bash
  cd /workspace/anybody
  /isaac-sim/python.sh scripts/env_utils/load_env.py --headless BENCHMARK_TASK intra_simple_bot_reach
  /isaac-sim/python.sh scripts/run.py --headless BENCHMARK_TASK intra_simple_bot_reach OVERRIDE_CFGNAME experiment_cfgs/mt_mlp_reach.yaml
  ```

---

### 4. Run Batch Jobs

To submit a batch job with Slurm from the `anybody` directory on the cluster, use:

```bash
./docker/cluster/submit_job_slurm.sh <slurm_job_name> /path/on/cluster/to/anybody <WANDB_API_KEY> <all arguments for run.py script>
```

**Example:**

```bash
./docker/cluster/submit_job_slurm.sh <slurm_job_name> /path/on/cluster/to/anybody <WANDB_API_KEY> --headless BENCHMARK_TASK intra_simple_bot_reach OVERRIDE_CFGNAME experiment_cfgs/mt_mlp_reach.yaml
```

---

