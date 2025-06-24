
## 🧪 Benchmark Overview

### Robot Set

1. **Download Robot Morphologies**  
  Download the robot morphologies for the AnyBody benchmark (as used in the [paper](http://arxiv.org/abs/2505.14986)) from [this link](https://drive.google.com/file/d/1YmtDxPNNHh_k0Xzk4MxBB0nJWIx6Z_EI/view?usp=sharing).

2. **Place Morphologies**  
  Extract and place the downloaded morphologies in the `anybody/morphs/robots/` directory. The structure should look like:

  ```
  anybody/
  └── morphs/
     └── robots/
        ├── __init__.py
        ├── arm_ur5/
        │   ├── r1/
        │   ├── r2/
        │   └── ...
        ├── chain/
        │   ├── r1/
        │   ├── r2/
        │   └── ...
        ├── ...
        └── real/
          ├── kinova_gen3/
          ├── jaco2/
          ├── panda/
          └── lwr/
  ```

3. **Procedurally Generate Robot Variants**  
  To generate additional robot variants (beyond those provided), run:

  ```bash
  python scripts/env_utils/create_robots.py --artificial --real --randomized
  ```

  - Each robot morphology has a **robot category** (`robo-cat`) and a **robot name** (`robo-name`).
  - `robo-cat` can be one of: `stick`, `nlink`, `arm_ur5`, `simple_bot`, `prims`, `chain`, `panda_variations`, or `real`.
  - `robo-name` is typically of the form `r{number}` (e.g., `r1`, `r2`, etc.) and can be found under the respective category directory.
  - For the `real` category, `robo-name` can be `kinova_gen3`, `jaco2`, `panda`, `lwr`, etc.
  - **Variations**: Robot names are followed by a version suffix (e.g., `r40_v1`, `r40_v2`). `v1` indicates environments without obstacles, while `v2` includes obstacles.

  To generate specific morphologies, specify the categories:

  ```bash
  python scripts/env_utils/create_robots.py --morphs stick prims ...
  ```

4. **Visualize Robot Morphologies**  
  To visualize morphologies using Meshcat (ensure `meshcat-server` is running):

  ```bash
  python source/ua_bench/bench/create_morphs/view_real.py <robo-cat> <robo-name>
  ```

  Replace `<robo-cat>` and `<robo-name>` with the desired category and robot name.


  ### Configs

  We use [yacs](https://github.com/rbgirshick/yacs) to manage system-wide parameters. All parameters can be set via config files or overridden directly from the command line—command line arguments always take precedence.
  The benchmark uses following config files:

  - **Universal config file** (`anybody/cfg.py`): Default parameters for the benchmark.
  - **Benchmark scenario configs** (`anybody/envs/benchmark_cfgs/<benchmark-name>.yaml`): YAML files specifying tasks, robots, and environments for each benchmark.
  - **Base config file** (`anybody/global_cfgs/base.yaml`): Shared base parameters for all experiments.
  - **Experiment configs** (`anybody/global_cfgs/experiment_configs/*.yaml`): YAML files for training, evaluation, and fine-tuning hyperparameters. These override defaults as needed.

  ### Benchmark Scenarios

  Each benchmark scenario YAML file defines:

  - **Task type**: The task to perform (e.g., `reach`, `push`).
  - **Train robots**: Robot categories and names used for training. Names are suffixed with `_v1`, `_v2`, etc., to indicate environment variations (e.g., `r40_v1`, `r40_v2`).
  - **Test robots**: Robot categories and names for evaluation, using the same naming convention as train robots.
  - **Seeds**: Random seeds for generating initial and goal targets (e.g., for `reach` tasks). Defaults to `0` unless specified.

  For multi-task scenarios, each of the above fields is a list corresponding to the multi-task environments in order. 

  #### Adding a New Benchmark Scenario

  1. Copy an existing `.yaml` file from `anybody/envs/benchmark_tasks/` and place it in the same directory. The filename is used for the benchmark task name.
  2. Edit the file to specify your desired task, robots, and environment.

  You can now run scripts using your new benchmark scenario.

  #### Loading Environments

  Use the `scripts/env_utils/load_env.py` script to load any environment. This script runs random actions in the environment.

  - **Load a benchmark task** (e.g., `intra_simple_bot_reach`):

    ```bash
    python scripts/env_utils/load_env.py BENCHMARK_TASK intra_simple_bot_reach
    ```
    This loads the scenario from `anybody/envs/benchmark_tasks/intra_simple_bot_reach.yaml`.

  - **Load a specific robot morphology**:

    ```bash
    python scripts/env_utils/load_env.py SE_TASK simple_bot/r40_v1/reach
    ```
    Here, `SE_TASK` is specified as `<robot_category>/<robot_name>/<task>`, such as `simple_bot/r40_v1/reach`.

  - See `anybody/cfg.py` for a complete list of parameters that can be passed to `load_env.py` or `run.py`.

  #### Pre-Generating Environments

  Each task variation involves random, non-colliding start and goal poses for robot joints and objects. Generating these configurations may take several minutes. It is recommended to pre-generate them for each task variation:

  ```bash
  python scripts/env_utils/create_problem_specs.py --headless
  ```

  Subsequent runs will use these pre-generated configurations for faster startup.

