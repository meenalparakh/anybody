
## ⚙️ Training Configuration

Training configuration files are located in `anybody/global_cfgs/experiment_cfgs`. Each YAML file specifies hyperparameters for different model architectures and tasks.

These config files override the default parameters in `anybody/cfg.py`. Below is an example configuration:

```yaml
GROUP_RUN_NAME: Tr0-c0-s0-t-nt-h0               # WandB group/run name

# ===========================
# Environment Settings
# ===========================

OBSERVATION:
    MASK_ROBO_MORPH: false
    PREV_ACTION: false
    HIGH_DIM: false            # Set to true for high-dimensional point cloud observations

ACTION:
    DISCRETE: false

# ===========================
# Training Parameters
# ===========================

TRAINER:
    TIMESTEPS: 1_000_000

TRAIN:
    EPISODE_LENGTH_S: 6.0
    NUM_ENVS: 128

SEARCH_CHECKPOINT: true
RUN_SEED: 42                 # Change to use a different RL seed

# ===========================
# Agent Configuration
# ===========================

AGENT_NAME: ppo

AGENT:
    EMA_CRITIC: false
    PCGRAD: false
    SYMLOG_RETURNS: false
    PPO:
        MINI_BATCHES: 16

# ===========================
# Model Architecture
# ===========================

MODEL:
    TYPE: transformer
    TRANSFORMER:
        DIM_FEEDFORWARD: 512
        DECODER_DIM: 256
        N_DECODER_LAYERS: 2
        NLAYERS: 4
        NHEAD: 4
    LIMB_EMBED_SIZE: 64
```

---

## 🚀 Running RL Training

You can run RL training using the provided config files. The `run.py` script supports both single-embodiment and multi-embodiment training.

### Single-Embodiment Training

```bash
python scripts/run.py --headless OVERRIDE_CFGNAME experiment_cfgs/se.yaml SE_TASK simple_bot/r40_v1/reach RUN_SEED 42 PROJECT_NAME isbrv EXPERIMENT_NAME simple_bot-r40_v1-reach-42 
```

- Trains a single robot embodiment (e.g., `r40` in the `simple_bot` category) for the `reach` task using the `se.yaml` configuration.

### Multi-Embodiment Training

```bash
python scripts/run.py --headless BENCHMARK_TASK intra_simple_bot_reach OVERRIDE_CFGNAME experiment_cfgs/mt_mlp_reach.yaml
```

- The `--headless` flag disables the GUI.
- `OVERRIDE_CFGNAME` specifies the training configuration file.
- Command-line arguments override both default and config file parameters.
- `OBSERVATION.HIGH_DIM: true` is used for high-dimensional input tasks.
- To run ablations, override `AGENT.EMA_CRITIC`, `AGENT.SYMLOG_RETURNS`, or `ACTION.DISCRETE` in the config or via command line.

### Fine-Tuning a Multi-Embodiment Policy

Use `ft_mt.yaml` to fine-tune a policy from a saved checkpoint:

```bash
python scripts/run.py --headless --enable_cameras OVERRIDE_CFGNAME experiment_cfgs/ft_mt.yaml EVAL_CHECKPOINT /path/to/checkpoints/agent_{timestep}.pt
```
- The checkpoint path can be absolute or relative to `LOGS_PATH`.
- Use `--enable_cameras` to record evaluation videos. To disable, remove the flag and add `TRAINER.VIDEO_RENDER False`.

---

## 📝 Evaluating Trained Models

The evaluation script loads the config associated with the trained model and runs the evaluation task. It supports both multi-task and zero-shot evaluation.

### Multi-Task Evaluation

Evaluate a trained policy on the morphologies it was trained on (works for both single- and multi-embodiment):

```bash
python scripts/run.py --headless --enable_cameras OVERRIDE_CFGNAME experiment_cfgs/eval_mt.yaml EVAL_CHECKPOINT LOGS_PATH/intra_simple_bot_reach/Tr0-c0-s0-t-nt-h0_42/checkpoints/agent_1000000.pt
```

### Zero-Shot Evaluation

Evaluate multi-embodiment or fine-tuned policies on test morphologies of the benchmark task:

```bash
python scripts/run.py --headless --enable_cameras OVERRIDE_CFGNAME experiment_cfgs/eval_zs.yaml EVAL_CHECKPOINT /path/to/agent.pt
```


