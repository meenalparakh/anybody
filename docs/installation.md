## 📦 Installation

This benchmark uses [MT-Isaaclab](https://github.com/princeton-vl/MT-IsaacLab/tree/develop), a multi-task learning framework built on top of [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html).

### Prerequisites

This repository has been tested with the following setup:

- **Ubuntu 22.04**
- **Python 3.10.14**
- **NVIDIA GPU with CUDA 11.8**
- [**Isaac Sim 4.5.0**](https://developer.nvidia.com/isaac-sim)

### Installation Steps

To set up the benchmark environment, follow these steps:

1. **Install Isaac Sim**:  
   This code has been tested with the binary version 4.5.0 (January Release).  
   Instructions can be found [here](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html).  
   Extract the binaries to your home directory (or any other suitable location).

2. **Clone this repository:**

   ```bash
   git clone --recurse-submodules https://github.com/meenalparakh/anybody.git
   cd anybody
   ```

3. **Create a symbolic link to the installed Isaac Sim:**

   ```bash
   # Assuming the Isaac Sim binaries are extracted at ~/isaacsim
   ln -s ~/isaacsim _isaac_sim
   ```

4. **Create and activate a conda environment:**

   ```bash
   # This command creates a conda environment named `anybody`
   ./anybody.sh --conda
   conda activate anybody
   ```

5. **Install Isaac Lab, the benchmark, and other dependencies:**

   ```bash
   ./anybody.sh --install
   ```

6. **Verify Isaac Sim installation:**

   ```bash
   ${ISAACSIM_PATH}/isaac-sim.sh
   ```

   This should open the Isaac Sim simulator window.

7. **Verify Isaac Lab and benchmark installation:**

   ```bash
   # This should launch the simulator and display a window with a black ground plane.
   python isaaclab/source/standalone/tutorials/00_sim/create_empty.py

   # Print the base configuration file of the benchmark.
   python -c 'import anybody; from anybody.cfg import cfg; print(cfg)'
   ```