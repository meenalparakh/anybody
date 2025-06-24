from anybody.utils.start_sim import args_cli, simulation_app
import os
from pathlib import Path

import wandb
import torch

from anybody.algos.multi_task_rl.agents.base import get_agent_cfg_and_memory, get_models
from anybody.algos.multi_task_rl.agents import agents
from anybody.algos.multi_task_rl.trainer import MySequentialLogTrainer

from anybody.cfg import cfg, dump_cfg, update_values, get_lower_case_cfg

from anybody.envs.sim.mtrl_cfg import BenchmarkRLCfg
from isaaclab.envs import ManagerBasedMTRLEnv
from anybody.envs.sim.gym_wrapper import MT_SKRLWrapper, VideoWrapper

from anybody.morphs.generate_morphs import create_real_robot_usd

from anybody.utils.utils import set_seed, is_none
from anybody.utils.path_utils import (
    get_wandb_fname,
    get_logs_dir,
    get_benchmark_cfgs_dir,
    get_global_cfgs_dir,
)

torch.autograd.set_detect_anomaly(True)

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


def set_logger_options():
    # hack for running within docker on the cluster. Unsafe!!
    if get_wandb_fname().exists():
        with open(get_wandb_fname(), "r") as f:
            key = f.read().strip()
        wandb.login(key=key)

    experiment_name = cfg.EXPERIMENT_NAME
    if is_none(experiment_name):
        experiment_name = cfg.GROUP_RUN_NAME + f"_{cfg.RUN_SEED}"

    if cfg.EVAL:
        ckpt_exp_name = Path(cfg.EVAL_CHECKPOINT).parents[1].name
        # get agent time step from the checkpoint name: agent_{t}.pt
        try:
            agent_time_step = int(
                Path(cfg.EVAL_CHECKPOINT).name.split("_")[-1].split(".")[0]
            )
            suffix = "_" + str(agent_time_step)
        except ValueError:
            suffix = ""

        if not cfg.IS_FINETUNING:
            if cfg.EVAL_ON_TEST:
                experiment_name = ckpt_exp_name + "_eval" + suffix
            else:
                experiment_name = ckpt_exp_name + "_mt-eval" + suffix  # multi-task eval

        elif cfg.IS_FINETUNING:
            experiment_name = ckpt_exp_name + "_ft" + suffix  # finetuning log dir

    if cfg.LOGGER == "wandb_offline":
        os.environ["WANDB_MODE"] = "offline"
        cfg.LOGGER = "wandb"

    if cfg.LOGGER == "wandb":
        cfg.AGENT.EXPERIMENT.EXPERIMENT_NAME = experiment_name
        cfg.AGENT.EXPERIMENT.DIRECTORY = cfg.AGENT.EXPERIMENT.BASE_DIRECTORY = (
            os.path.join(get_logs_dir(), cfg.PROJECT_NAME)
        )
        cfg.AGENT.EXPERIMENT.WANDB = True
        cfg.AGENT.EXPERIMENT.WANDB_KWARGS.PROJECT = cfg.PROJECT_NAME
        cfg.AGENT.EXPERIMENT.WANDB_KWARGS.GROUP = cfg.GROUP_RUN_NAME
        cfg.AGENT.EXPERIMENT.WANDB_KWARGS.DIR = os.path.join(
            cfg.AGENT.EXPERIMENT.DIRECTORY, experiment_name
        )
        cfg.TRAINER.VIDEO_DIR = os.path.join(
            cfg.AGENT.EXPERIMENT.DIRECTORY, experiment_name, "videos"
        )
        wandb.tensorboard.patch(
            root_logdir=os.path.join(cfg.AGENT.EXPERIMENT.DIRECTORY, experiment_name)
        )


def set_ckpts():
    if cfg.SEARCH_CHECKPOINT:
        possible_dir = os.path.join(
            cfg.AGENT.EXPERIMENT.DIRECTORY,
            cfg.AGENT.EXPERIMENT.EXPERIMENT_NAME,
            "checkpoints",
        )

        print(f"Searching for checkpoint in: {possible_dir}")
        if os.path.exists(possible_dir):
            all_agents = os.listdir(possible_dir)
            ts = [int(x[:-3].split("_")[-1]) for x in all_agents]
            if len(ts) == 0:
                print("No checkpoints found in the directory.")
                return
            ts.sort()
            agent = f"agent_{ts[-1]}.pt"
            cfg.TRAIN_CHECKPOINT = os.path.join(possible_dir, agent)

            print("*" * 60)
            print("*" * 60)
            print("Checkpoint found: ", cfg.TRAIN_CHECKPOINT)
            print("*" * 60)
            print("*" * 60)


def set_cfg_options():
    """
    Set the derived options in the config
    """
    cfg.TRAIN_CHECKPOINT = format_ckpt_path(cfg.TRAIN_CHECKPOINT)
    cfg.EVAL_CHECKPOINT = format_ckpt_path(cfg.EVAL_CHECKPOINT)

    # turn off visualization if running headless
    if args_cli.headless and (not cfg.TRAINER.VIDEO_RENDER):
        cfg.COMMAND.DEBUG_VIS = False
    else:
        cfg.COMMAND.DEBUG_VIS = True

    if "push_simple" in cfg.MULTIENV.TASKS:
        # set the reward related configs appropriately
        cfg.REWARD.OBJ_SIMPLE_REWARD = True

    if cfg.AGENT_NAME == "random":
        cfg.AGENT.EXPERIMENT.WRITE_INTERVAL = 500
        cfg.TRAINER.TIMESTEPS = 10000

    if cfg.EVAL:
        if not cfg.IS_FINETUNING:
            cfg.MODEL.POLICY.MIN_LOG_STD = -9  # std 10^-6, that is very small std
            cfg.MODEL.POLICY.MAX_LOG_STD = -3  # std 10^-3, that is small std
        else:
            cfg.AGENT.RANDOM_TIMESTEPS = 0
            cfg.AGENT.LEARNING_STARTS = 0

    # update the values in the config
    update_values(cfg)


def load_env():
    env_cfg = BenchmarkRLCfg()
    env = ManagerBasedMTRLEnv(
        cfg=env_cfg, render_mode="rgb_array" if cfg.TRAINER.VIDEO_RENDER else None
    )

    # if eval, then wrap env in video recorder
    if cfg.TRAINER.VIDEO_RENDER:
        video_kwargs = {
            "video_folder": cfg.AGENT.EXPERIMENT.WANDB_KWARGS.DIR + "/videos",
            "step_trigger": lambda step: step % cfg.TRAINER.EVAL_TIMESTEPS_INTERVAL
            == 0,
            "video_length": cfg.TRAINER.EVAL_TIMESTEPS,
            "name_prefix": "video",
            "disable_logger": True,
        }
        env = VideoWrapper(env, **video_kwargs)

    env = MT_SKRLWrapper(env)

    print(f"[INFO]: Observation space: {env.observation_space}")
    print(f"[INFO]: Action space: {env.action_space}")

    return env


def load_checkpoint(agent):
    if not cfg.EVAL:
        # In training mode
        if not is_none(cfg.TRAIN_CHECKPOINT):
            print("Loading checkpoint:", cfg.TRAIN_CHECKPOINT)
            # finetuning models further on train env
            # Choose whether to skip loading the optimizer state
            skip_optimizer = cfg.IS_FINETUNING
            agent.resume_from_checkpoint(
                cfg.TRAIN_CHECKPOINT, skip_optimizer=skip_optimizer
            )

    else:
        # In evaluation mode
        print("Loading checkpoint:", cfg.EVAL_CHECKPOINT)

        if cfg.IS_FINETUNING:  # finetuning on test env
            # Load only model weights (not optimizer)
            agent.resume_from_checkpoint(cfg.EVAL_CHECKPOINT, skip_optimizer=True)
        else:
            # Load entire model for evaluation
            print("Evaluating the model at location:", cfg.EVAL_CHECKPOINT)
            agent.load(cfg.EVAL_CHECKPOINT)


def load_agent(env):
    device = "cuda"
    agent_cfg, memory = get_agent_cfg_and_memory(env, device, only_agent=False)
    models = get_models(env, device)

    # if eval, set the random timesteps in agent_cfg to 0, and learning starts to a very large value
    if cfg.EVAL and (not cfg.IS_FINETUNING):
        agent_cfg["random_timesteps"] = 0
        agent_cfg["learning_starts"] = 1e8

    elif cfg.IS_FINETUNING and cfg.EVAL:
        agent_cfg["random_timesteps"] = 0
        agent_cfg["learning_starts"] = 0

    # ////////////////////////  Initialize the agent  ////////////////////////
    agent = agents[cfg.AGENT_NAME](
        env_name_list=list(env.__getattr__("envs").keys()),
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    return agent


def run():
    """
    Run the RL training or evaluation loop.
    """
    env = load_env()

    # load the agent
    agent = load_agent(env)

    # load the checkpoint if specified
    load_checkpoint(agent)

    # initialize the trainer
    trainer_cfg = get_lower_case_cfg(cfg.TRAINER)
    trainer = MySequentialLogTrainer(cfg=trainer_cfg, env=env, agents=agent)

    set_seed(cfg.RUN_SEED)

    # dump cfg to wandb run
    if cfg.LOGGER == "wandb":
        wandb.config.update(cfg)

    # run the training or evaluation loop
    if cfg.IS_FINETUNING or (not cfg.EVAL):
        # train mode
        print("Running training...")
        trainer.train()
    else:
        print("Running evaluation...")
        trainer.eval()

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

    if cfg.EVAL:
        # the corresponding config.yaml
        assert not is_none(cfg.EVAL_CHECKPOINT), (
            "EVAL_CHECKPOINT must be provided if cfg.EVAL is True."
        )

        cfg.EVAL_CHECKPOINT = format_ckpt_path(cfg.EVAL_CHECKPOINT)
        config_path = Path(cfg.EVAL_CHECKPOINT).parents[1] / "config.yaml"
        cfg.merge_from_file(config_path)

    # override the config with the command line arguments if specified
    # this will give preference to the command line arguments
    cfg.merge_from_list(args_cli.opts)


if __name__ == "__main__":
    load_cfg()
    set_env_options()
    set_cfg_options()
    set_logger_options()
    set_ckpts()
    set_seed(cfg.RUN_SEED)

    cfg.freeze()

    dump_cfg()

    # update real robot usd files if any
    update_real_robot_usd()

    run()

    if cfg.LOGGER == "wandb":
        wandb.finish()

    simulation_app.close()
