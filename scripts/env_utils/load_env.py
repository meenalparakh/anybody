from anybody.utils.start_sim import args_cli, simulation_app
import os
from pathlib import Path

import wandb
import torch

from anybody.utils.utils import set_seed, is_none
from anybody.utils.path_utils import (
    get_logs_dir,
    get_benchmark_cfgs_dir,
    get_global_cfgs_dir,
)
from anybody.cfg import cfg, update_values

from anybody.envs.sim.mtrl_cfg import BenchmarkRLCfg
from isaaclab.envs import ManagerBasedMTRLEnv

from anybody.morphs.generate_morphs import create_real_robot_usd

torch.autograd.set_detect_anomaly(True)

# set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid cuda out of memory error
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def format_ckpt_path(checkpoint_path):
    if not is_none(checkpoint_path):
        if "LOGS_PATH" in checkpoint_path:
            # replace logs_path with logs dir path
            checkpoint_path = checkpoint_path.replace("LOGS_PATH", str(get_logs_dir()))
        return checkpoint_path


def update_real_robot_usd():
    if ("real" in cfg.MULTIENV.ROBOTS) or ("panda_variations" in cfg.MULTIENV.ROBOTS):
        for idx, robo_name_ver in enumerate(cfg.MULTIENV.VARIATIONS):
            # robo_name_ver_split = robo_name_ver.split("_")
            # robo_name = '_'.join(robo_name_ver_split[:-1])
            robo_cat = cfg.MULTIENV.ROBOTS[idx]
            if robo_cat != "real":
                if robo_cat != "panda_variations":
                    continue
            robo_name = robo_name_ver[:-3]
            print(
                f"################### Converting to usd: {robo_name} ###################"
            )
            create_real_robot_usd(robo_name)


def set_env_options():
    # if benchmark task is given, set the multi-env fields accordingly
    if cfg.BENCHMARK_TASK:
        # load the benchmark config
        benchmark_cfg = get_benchmark_cfgs_dir() / f"{cfg.BENCHMARK_TASK}.yaml"
        if not benchmark_cfg.exists():
            raise FileNotFoundError(f"Benchmark config not found at {benchmark_cfg}")
        cfg.merge_from_file(benchmark_cfg)

    if cfg.SE_TASK:
        try:
            robo_cat, robo_variation, robo_task = cfg.SE_TASK.split("/")
        except ValueError:
            raise ValueError(
                "SE_TASK must be in the format 'robot_category/robot_variation/task_name'"
            )
        cfg.MULTIENV.ROBOTS = [robo_cat]
        cfg.MULTIENV.VARIATIONS = [robo_variation]
        cfg.MULTIENV.TASKS = [robo_task]
        cfg.MULTIENV.SEEDS = [0]

    if cfg.EVAL_ON_TEST or cfg.IS_FINETUNING:
        # if eval is true or finetuning, need to replace the multienvs with the eval envs
        cfg.MULTIENV.TASKS = cfg.TEST_ENVS.TASKS
        cfg.MULTIENV.ROBOTS = cfg.TEST_ENVS.ROBOTS
        cfg.MULTIENV.SEEDS = cfg.TEST_ENVS.SEEDS
        cfg.MULTIENV.VARIATIONS = cfg.TEST_ENVS.VARIATIONS

    if isinstance(cfg.MULTIENV.TASKS, list):
        n = len(cfg.MULTIENV.TASKS)
    elif isinstance(cfg.MULTIENV.ROBOTS, list):
        n = len(cfg.MULTIENV.ROBOTS)
    elif isinstance(cfg.MULTIENV.SEEDS, list):
        n = len(cfg.MULTIENV.SEEDS)
    elif isinstance(cfg.MULTIENV.VARIATIONS, list):
        n = len(cfg.MULTIENV.VARIATIONS)
    else:
        # assume that all are single values
        n = 1
        # raise ValueError("None of the cfg.MULTIENV.* is a list.")

    n = max(n, 1)

    # convert any string to a list
    if isinstance(cfg.MULTIENV.TASKS, str):
        cfg.MULTIENV.TASKS = [cfg.MULTIENV.TASKS] * n
    if isinstance(cfg.MULTIENV.ROBOTS, str):
        cfg.MULTIENV.ROBOTS = [cfg.MULTIENV.ROBOTS] * n
    if isinstance(cfg.MULTIENV.SEEDS, int):
        cfg.MULTIENV.SEEDS = [cfg.MULTIENV.SEEDS] * n
    if isinstance(cfg.MULTIENV.VARIATIONS, str):
        cfg.MULTIENV.VARIATIONS = [cfg.MULTIENV.VARIATIONS] * n

    if cfg.IS_FINETUNING:
        # finetune on only one test env
        cfg.MULTIENV.TASKS = [cfg.MULTIENV.TASKS[-1]]
        cfg.MULTIENV.ROBOTS = [cfg.MULTIENV.ROBOTS[-1]]
        cfg.MULTIENV.SEEDS = [cfg.MULTIENV.SEEDS[-1]]
        cfg.MULTIENV.VARIATIONS = [cfg.MULTIENV.VARIATIONS[-1]]


def set_cfg_options():
    """
    Set the derived options in the config
    """
    # turn off visualization if running headless
    if args_cli.headless and (not cfg.TRAINER.VIDEO_RENDER):
        cfg.COMMAND.DEBUG_VIS = False
    else:
        cfg.COMMAND.DEBUG_VIS = True

    if "push_simple" in cfg.MULTIENV.TASKS:
        # set the reward related configs appropriately
        cfg.REWARD.OBJ_SIMPLE_REWARD = True

    cfg.LOGGER = None

    # update the values in the config
    update_values(cfg)


def run():
    # Configure the CUDNN backend
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

    env_cfg = BenchmarkRLCfg()
    # env = ManagerBasedMTRLEnv(cfg=env_cfg, render_mode="rgb_array" if cfg.TRAINER.VIDEO_RENDER else None)
    env = ManagerBasedMTRLEnv(cfg=env_cfg, render_mode="rgb_array")
    env.reset()
    # training seed, independent of the environment seed
    set_seed(cfg.RUN_SEED)

    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
            # sample random actions
            actions = []
            for task_env in env.envs.values():
                joint_efforts = torch.randn_like(task_env.action_manager.action)
                actions.append(joint_efforts)

            actions = torch.cat(actions, dim=0)  # concatenate actions for all tasks
            # step the environment
            obs, rew, terminated, truncated, info = env.step(actions)
            # print current observations

            if count % 100 == 0:
                print("-" * 80)
                print(
                    f"[INFO]: Shapes - Rew: {rew.shape}, Terminated: {terminated.shape}, Truncated: {truncated.shape}, Info: {len(info)}"
                )

            # update counter
            count += 1

    # close the simulator
    env.close()


def load_cfg():
    # loads default arguments
    base_cfgname = "base.yaml"
    cfg.merge_from_file(get_global_cfgs_dir() / base_cfgname)

    # load command line arguments
    cfg.merge_from_list(args_cli.opts)

    # load override configs (useful for running experiments with different configurations)
    if not is_none(cfg.OVERRIDE_CFGNAME):
        # if override_cfgname path is absolute, then use it as is
        # else use the global_cfgs_dir as the base directory
        cfg.OVERRIDE_CFGNAME = format_ckpt_path(cfg.OVERRIDE_CFGNAME)
        if Path(cfg.OVERRIDE_CFGNAME).is_absolute():
            cfg.merge_from_file(cfg.OVERRIDE_CFGNAME)
        else:
            cfg.merge_from_file(get_global_cfgs_dir() / cfg.OVERRIDE_CFGNAME)
    # override the config with the command line arguments if specified
    # this will give preference to the command line arguments
    cfg.merge_from_list(args_cli.opts)


if __name__ == "__main__":
    load_cfg()
    set_env_options()
    set_cfg_options()
    set_seed(cfg.RUN_SEED)

    cfg.freeze()

    # update real robot usd files if any
    update_real_robot_usd()

    run()
    simulation_app.close()
