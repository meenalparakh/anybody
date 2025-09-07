from anybody.utils.start_sim import args_cli, simulation_app
import os
from pathlib import Path
import numpy as np
import wandb
import torch

from anybody.algos.multi_task_rl.agents.base import get_agent_cfg_and_memory, get_models
from anybody.algos.multi_task_rl.agents import agents
from anybody.algos.multi_task_rl.trainer import MySequentialLogTrainer
from isaaclab.managers import SceneEntityCfg

from anybody.cfg import cfg, dump_cfg, update_values, get_lower_case_cfg
from isaaclab.utils.math import quat_from_matrix, subtract_frame_transforms

from anybody.envs.sim.mtrl_cfg import BenchmarkRLCfg
from isaaclab.envs import ManagerBasedMTRLEnv, ManagerBasedRLEnv
from anybody.envs.sim.gym_wrapper import MT_SKRLWrapper, VideoWrapper
import anybody.envs.sim.utils as iu
from anybody.morphs.generate_morphs import create_real_robot_usd
from anybody.envs.tasks import generate_problem_spec

from anybody.utils.utils import set_seed, is_none
from anybody.utils.path_utils import (
    get_wandb_fname,
    get_logs_dir,
    get_benchmark_cfgs_dir,
    get_global_cfgs_dir,
)
from isaaclab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
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

    # if cfg.EVAL: defaulting to eval mode
    
    ckpt_exp_name = "diff-ik"
    # get agent time step from the checkpoint name: agent_{t}.pt
    group_name = "diff-ik"
    cfg.GROUP_RUN_NAME = group_name

    if cfg.LOGGER == "wandb_offline":
        os.environ["WANDB_MODE"] = "offline"
        cfg.LOGGER = "wandb"

    if cfg.LOGGER == "wandb":
        cfg.AGENT.EXPERIMENT.EXPERIMENT_NAME = ckpt_exp_name
        cfg.AGENT.EXPERIMENT.DIRECTORY = cfg.AGENT.EXPERIMENT.BASE_DIRECTORY = (
            os.path.join(get_logs_dir(), cfg.PROJECT_NAME)
        )
        cfg.AGENT.EXPERIMENT.WANDB = True
        cfg.AGENT.EXPERIMENT.WANDB_KWARGS.PROJECT = cfg.PROJECT_NAME
        cfg.AGENT.EXPERIMENT.WANDB_KWARGS.GROUP = cfg.GROUP_RUN_NAME
        cfg.AGENT.EXPERIMENT.WANDB_KWARGS.DIR = os.path.join(
            cfg.AGENT.EXPERIMENT.DIRECTORY, ckpt_exp_name
        )
        cfg.TRAINER.VIDEO_DIR = os.path.join(
            cfg.AGENT.EXPERIMENT.DIRECTORY, ckpt_exp_name, "videos"
        )
        wandb.tensorboard.patch(
            root_logdir=os.path.join(cfg.AGENT.EXPERIMENT.DIRECTORY, ckpt_exp_name)
        )


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
        cfg.TRAIN.EPISODE_LENGTH_S = 5.0
    else:
        cfg.TRAIN.EPISODE_LENGTH_S = 3.0

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


def set_eval_cfg():
    cfg.EVAL = True
    cfg.EVAL_ON_TEST = False
    cfg.IS_FINETUNING = False
    
    cfg.FORCE_PROBLEM_SPEC_GEN = False
    cfg.SEARCH_CHECKPOINT = False
    
    cfg.TRAINER.TIMESTEPS = 10000
    cfg.TRAINER.EVAL_TIMESTEPS_INTERVAL = 5000
    cfg.TRAINER.EVAL_TIMESTEPS = 500
    cfg.TRAINER.VIDEO_RENDER = True  # if running headless, need `
    
    cfg.AGENT.EXPERIMENT.CHECKPOINT_INTERVAL = 12000  # Skip checkpointing during evaluation
    cfg.AGENT.EXPERIMENT.WRITE_INTERVAL = 500
    cfg.CURRICULUM.ACTIVE = False
    

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


def get_problem_specs():
    probs = {}
    for i in range(len(cfg.MULTIENV.TASKS)):
        robo_task = cfg.MULTIENV.TASKS[i]
        robo_type = cfg.MULTIENV.ROBOTS[i]
        seed = cfg.MULTIENV.SEEDS[i]
        robot_env_variation = cfg.MULTIENV.VARIATIONS[i]

        # task_name = f"Task_{i}"
        # for logging, need unique task names for each task
        task_name = f"{robo_type}_{robot_env_variation}_{robo_task}"

        prob = generate_problem_spec(
            benchmark_task=cfg.BENCHMARK_TASK,
            robo_cat=robo_type,
            robo_task=robo_task,
            variation=robot_env_variation,
            seed=seed,
            save_if_not_exist=True,
        )
        probs[task_name] = prob

    return probs


def load_diff_ik_module(env, agent):
    # 1. initialize the diff-ik module in the agent.
    #    assume there are multiple envs.
    # 2. override the act method of the agent.

    # it has the following description:        
    # actions = self.agents.act(
    #     states, timestep=timestep, timesteps=self.timesteps
    # )[0]
    

    def X_to_pose(X: np.ndarray):
        pos = torch.from_numpy(X[:3, 3])
        mat = torch.from_numpy(X[:3, :3])
        quat = quat_from_matrix(mat)
        pose = torch.cat([pos, quat])
        return pose

    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type='position', use_relative_mode=False, ik_method="dls", 
        # ik_params={"lambda_val": 0.1}
    )
    
    diff_iks = {}
    robots = {}
    robo_cfgs = {}
    probs = get_problem_specs()
    ee_jacobi_indices = {}
    robo_base_poses = {}
    robo_info_dict = {}
    prob_robots = {}

    for env_name, task_env in env.__getattr__("envs").items():
        task_env: ManagerBasedRLEnv
        diff_ik = DifferentialIKController(
            cfg=diff_ik_cfg, num_envs=task_env.scene.num_envs, device=env.device
        )
        diff_iks[env_name] = diff_ik
        
        
        prob = probs[env_name]
        robo_id = list(prob.robot_dict.keys())[0]
        robo = prob.robot_dict[robo_id]
        prob_robots[env_name] = robo

        entity_cfg = SceneEntityCfg(
            f"robot_{robo_id}",
            joint_names=prob.robot_dict[robo_id].act_info["joint_names"],
            body_names=[robo.ee_link],
            preserve_order=True,
        )
        robo_cfgs[env_name] = entity_cfg
        entity_cfg.resolve(task_env.scene)
        
        robots[env_name] = task_env.scene[f"robot_{robo_id}"]

        if robots[env_name].is_fixed_base:
            ee_jacobi_idx = entity_cfg.body_ids[0] - 1
        else:
            ee_jacobi_idx = entity_cfg.body_ids[0]

        ee_jacobi_indices[env_name] = ee_jacobi_idx
        base_pose = prob.robot_dict[robo_id].pose
        robo_base_poses[env_name] = X_to_pose(base_pose).to(env.device)
        robo_info_dict[env_name] = iu.precompute_robo_info(prob)[robo_id]


    return {
        'diff_iks': diff_iks,
        'robots': robots,
        'prob_robots': prob_robots,
        'robo_cfgs': robo_cfgs,
        'ee_jacobi_indices': ee_jacobi_indices,
        'robo_base_poses': robo_base_poses,
        "robo_info_dicts": robo_info_dict
    }


def diff_ik_step(env, states, diff_ik_info):
    # first extract the goal (target position) from the states    
    
    target_pose = states['robo_goal']  # (num_envs, 7) pos(3) + quat(4) relative to ee.
    
    # command type is position
    ik_commands = target_pose[:, 0, :]
    
    actions = []

    prev_env_idx = 0
    for task_idx, (env_name, task_env) in enumerate(env.__getattr__("envs").items()):
        ik_commands_env = ik_commands[prev_env_idx: prev_env_idx + task_env.num_envs]
        prev_env_idx += task_env.num_envs
        diff_ik_module = diff_ik_info['diff_iks'][env_name]
        robot = diff_ik_info['robots'][env_name]
        robot_scene_cfg = diff_ik_info['robo_cfgs'][env_name]
        ee_jacobi_idx = diff_ik_info['ee_jacobi_indices'][env_name]
        robo_info = diff_ik_info['robo_info_dicts'][env_name]
        prob_robot = diff_ik_info['prob_robots'][env_name]
        # robo_base_pose = diff_ik_info['robo_base_poses'][env_name]
        
        diff_ik_module.reset()
        diff_ik_module.set_command(ik_commands_env[:, :3], ee_quat=ik_commands_env[:, 3:].clone())
        
        jacobian = robot.root_physx_view.get_jacobians()[
            :, ee_jacobi_idx, :, robot_scene_cfg.joint_ids
        ]
        ee_pose_w = robot.data.body_state_w[
            :, robot_scene_cfg.body_ids[0], 0:7
        ]
        root_pose_w = robot.data.root_state_w[:, 0:7]
        joint_pos = robot.data.joint_pos[:, robot_scene_cfg.joint_ids]

        # ee_pos_b, ee_quat_b = subtract_frame_transforms(
        #     root_pose_w[:, 0:3],
        #     root_pose_w[:, 3:7],
        #     ee_pose_w[:, 0:3],
        #     ee_pose_w[:, 3:7],
        # )
        ee_pos_b = ee_pose_w[:, :3] - task_env.scene.env_origins
        ee_quat_b = ee_pose_w[:, 3:7]
        
        # compute the joint commands
        jpos_des = diff_ik_module.compute(
            ee_pos_b, ee_quat_b, jacobian, joint_pos
        )

        # the rl env takes as actions, the delta joint positions
        if cfg.ACTION.ABSOLUTE:
            jpos_diff = jpos_des
        else:               
            jpos_diff = jpos_des - joint_pos
        
        jnames = robot_scene_cfg.joint_names
        robot_actions = {
            jname: jpos_diff[:, i].unsqueeze(-1) for i, jname in enumerate(jnames)
        }

        # actions are absolute action values (not deltas)
        env_actions = iu.map_trajectory_to_actions_batched(
            prob_robot, robo_info, robot_actions, 0, task_env.scene.num_envs
        )
        actions.append(env_actions)
        
    action = torch.cat(actions, dim=0)
    
    return action.to(env.device)

# need to override the act method of the agent
# it has the following description:
# the last argument timesteps is not used in PPO
# actions = self.agents.act(
#     states, timestep=timestep, timesteps=self.timesteps
# )[0]



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
    ik_info = load_diff_ik_module(env, agent)

    def new_act(states, role='policy'):
        # print("Using diff-ik based action")
        action = diff_ik_step(env, states['states'], ik_info)
        logprob = torch.zeros(action.shape[0], 1).to(action.device)
        # print(f"Action: {action.shape}, Logprob: {logprob.shape}")
        
        return action, logprob, {}
    
    agent.policy.act = new_act

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
    # the order of loading config settings is very important.
    
    # loads default arguments
    base_cfgname = "base.yaml"
    cfg.merge_from_file(get_global_cfgs_dir() / base_cfgname)

    # load command line arguments
    cfg.merge_from_list(args_cli.opts)

    if not is_none(cfg.EVAL_CHECKPOINT):
        # the corresponding config.yaml
        # assert cfg.EVAL, (
        #     "cfg.EVAL is True must be true. (just a sanity check)"
        # )
        cfg.EVAL = True

        cfg.EVAL_CHECKPOINT = format_ckpt_path(cfg.EVAL_CHECKPOINT)
        config_path = Path(cfg.EVAL_CHECKPOINT).parents[1] / "config.yaml"
        cfg.merge_from_file(config_path)
        
        
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
    set_eval_cfg()
    set_env_options()
    set_cfg_options()
    set_logger_options()
    set_seed(cfg.RUN_SEED)

    cfg.freeze()

    dump_cfg()

    # update real robot usd files if any
    update_real_robot_usd()

    run()

    if cfg.LOGGER == "wandb":
        wandb.finish()

    simulation_app.close()
