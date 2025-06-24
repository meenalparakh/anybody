# This file is part of AnyBody (BSD-3-Clause License).
#
# Copyright (c) Meenal Parakh, 2025
# All rights reserved.
#
# This file includes modified code from:
# SKRL (https://github.com/Toni-SM/skrl)
# Licensed under the MIT License (see LICENSES/MIT_LICENSE.txt)
#
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
from collections import deque
from packaging import version

# import collections
import torch
from torch import nn
from anybody.utils.utils import mydequedict
from skrl.agents.torch.base import Agent
from typing import Any, Optional
from anybody.algos.multi_task_rl.memory.buffer import CustomMemory
from anybody.algos.multi_task_rl.models.shared_model import (
    StochasticPolicy_DiscreteAction,
    StochasticPolicy,
    DeterministicPolicy,
    Value,
    QValueFunc
)
from skrl.resources.noises.torch import GaussianNoise
from anybody.algos.multi_task_rl.models.rnn_model import PolicyRNN, CriticRNN
# monkey patch the agent's record_transition method
from skrl import logger

from copy import copy

from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC_DEFAULT_CONFIG
from skrl.agents.torch.td3 import TD3_DEFAULT_CONFIG
from anybody.cfg import cfg as global_cfg, get_lower_case_cfg
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, _LRScheduler

import math


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def process_skrl_cfg(cfg: dict) -> dict:
    """Convert simple YAML types to skrl classes/components.

    Args:
        cfg: A configuration dictionary.

    Returns:
        A dictionary containing the converted configuration.
    """
    _direct_eval = [
        "learning_rate_scheduler",
        "state_preprocessor",
        "value_preprocessor",
        "input_shape",
        "output_shape",
    ]

    def reward_shaper_function(scale):
        def reward_shaper(rewards, timestep, timesteps):
            return rewards * scale

        return reward_shaper

    def update_dict(d):
        for key, value in d.items():
            if isinstance(value, dict):
                update_dict(value)
            else:
                if key in _direct_eval:
                    d[key] = eval(value)
                elif key.endswith("_kwargs"):
                    d[key] = value if value is not None else {}
                elif key in ["rewards_shaper_scale"]:
                    d["rewards_shaper"] = reward_shaper_function(value)

        return d

    # parse agent configuration and convert to classes
    return update_dict(cfg)


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, base_lr, warmup_steps, warmup_factor, total_steps, eta_min=0, last_epoch=-1):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        self.total_steps = total_steps
        self.eta_min = eta_min
        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warm-up phase
            warmup_factor = self.warmup_factor + (1 - self.warmup_factor) * (self.last_epoch / self.warmup_steps)
            return [self.base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            cosine_steps = self.last_epoch - self.warmup_steps
            cosine_total_steps = self.total_steps - self.warmup_steps
            cosine_factor = 0.5 * (1 + math.cos(math.pi * cosine_steps / cosine_total_steps))
            warmup_end_lr = self.base_lr * (1 - self.warmup_factor) + self.base_lr * self.warmup_factor
            return [self.eta_min + (warmup_end_lr - self.eta_min) * cosine_factor for base_lr in self.base_lrs]


class AdaptiveRewardScaler_MovingWindow:
    def __init__(self, device):
        self.mean = torch.tensor(0.0, device=device)
        self.std = torch.tensor(1.0, device=device)  
        self.device = device      
        self.rewards = deque(maxlen=global_cfg.AGENT.SMOOTHING_WINDOW)

    def update(self, rewards):
        batch_mean = rewards.mean().detach().cpu().numpy()
        self.rewards.append(batch_mean)
        self.mean = torch.tensor(np.mean(self.rewards), device=self.device)
        self.std = torch.tensor(np.std(self.rewards),   device=self.device)
        
    def scale(self, rewards):
        # mean = self.mean.to(rewards.device)
        # std = self.std.to(rewards.device)
        mean, std = self.mean, self.std
        
        return (rewards - mean) / (std + 1e-4)


class AdaptiveRewardScaler:
    def __init__(self, device):
        self.mean = torch.tensor(0.0, device=device)
        self.std = torch.tensor(1.0, device=device)
        self.count = 0
        
        # also maintain a queue if we want to consider only the last N
        # self.rewards = deque(maxlen=global_cfg.AGENT.SMOOTHING_WINDOW)

    def update(self, rewards):

        # print("Rewards shape: ", rewards.shape)
        batch_mean = rewards.mean()
        batch_std = rewards.std()
        batch_count = rewards.size(0)

        if self.count == 0:
            self.mean = batch_mean
            self.std = batch_std
            self.count = batch_count
            return

        self.mean = (self.mean + batch_mean * batch_count / self.count) / (1 + batch_count / self.count)
        self.std = (self.std + batch_std * batch_count / self.count) / (1 + batch_count / self.count)
        self.count += batch_count

    def scale(self, rewards):
        return (rewards - self.mean) / (self.std + 1e-3)


def reward_shaper_fn(reward, timestep, timesteps):
    # use symlog function to shape the reward
    # symlog = sign(x) * log(1 + abs(x))
    return torch.sign(reward) * torch.log1p(torch.abs(reward))


def _my_record_transition(
    self,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    infos: Any,
    timestep: int,
    timesteps: int,
    is_train: Optional[bool] = True,
) -> None:
    """Record an environment transition in memory (to be implemented by the inheriting classes)

    Inheriting classes must call this method to record episode information (rewards, timesteps, etc.).
    In addition to recording environment transition (such as states, rewards, etc.), agent information can be recorded.

    :param states: Observations/states of the environment used to make the decision
    :type states: torch.Tensor
    :param actions: Actions taken by the agent
    :type actions: torch.Tensor
    :param rewards: Instant rewards achieved by the current actions
    :type rewards: torch.Tensor
    :param next_states: Next observations/states of the environment
    :type next_states: torch.Tensor
    :param terminated: Signals to indicate that episodes have terminated
    :type terminated: torch.Tensor
    :param truncated: Signals to indicate that episodes have been truncated
    :type truncated: torch.Tensor
    :param infos: Additional information about the environment
    :type infos: Any type supported by the environment
    :param timestep: Current timestep
    :type timestep: int
    :param timesteps: Number of timesteps
    :type timesteps: int
    """

    if self.write_interval > 0:
        n_mt = len(infos)
        n_envs = rewards.shape[0] // n_mt
        assert (
            rewards.shape[0] % n_mt == 0
        ), f"Number of environments: {rewards.shape[0]} is not a multiple of the number of tasks: {n_mt}"

        finished_episodes = terminated + truncated

        for task_idx, (task_name, task_info) in enumerate(infos.items()):
            if (task_name not in self._cumulative_rewards) or (
                self._cumulative_rewards[task_name] is None
            ):
                self._cumulative_rewards[task_name] = torch.zeros_like(
                    rewards[task_idx * n_envs : (task_idx + 1) * n_envs], dtype=torch.float32
                )
                self._cumulative_timesteps[task_name] = torch.zeros_like(
                    rewards[task_idx * n_envs : (task_idx + 1) * n_envs], dtype=torch.int32
                )
                self._track_rewards[task_name] = deque(maxlen=100)
                self._track_timesteps[task_name] = deque(maxlen=100)

            self._cumulative_rewards[task_name].add_(
                rewards[task_idx * n_envs : (task_idx + 1) * n_envs]
            )
            self._cumulative_timesteps[task_name].add_(1)

            # check ended episodes
            idx_finished_episodes = finished_episodes[
                task_idx * n_envs : (task_idx + 1) * n_envs
            ].nonzero(as_tuple=False)
            if idx_finished_episodes.numel():
                # storage cumulative rewards and timesteps
                self._track_rewards[task_name].extend(
                    self._cumulative_rewards[task_name][idx_finished_episodes][:, 0]
                    .reshape(-1)
                    .tolist()
                )
                self._track_timesteps[task_name].extend(
                    self._cumulative_timesteps[task_name][idx_finished_episodes][:, 0]
                    .reshape(-1)
                    .tolist()
                )

                # reset the cumulative rewards and timesteps
                self._cumulative_rewards[task_name][idx_finished_episodes] = 0
                self._cumulative_timesteps[task_name][idx_finished_episodes] = 0

            # get env name to append to the right environment data

            # record data
            self.tracking_data[
                f"Additional Info / {task_name} / Inst. reward (max)"
            ].append(torch.max(rewards[task_idx * n_envs : (task_idx + 1) * n_envs]).item())
            self.tracking_data[
                f"Additional Info / {task_name} / Inst. reward (min)"
            ].append(torch.min(rewards[task_idx * n_envs : (task_idx + 1) * n_envs]).item())
            self.tracking_data[
                f"Additional Info / {task_name} / Reward / Inst. reward (mean)"
            ].append(torch.mean(rewards[task_idx * n_envs : (task_idx + 1) * n_envs]).item())

            if len(self._track_rewards[task_name]):
                track_rewards = np.array(self._track_rewards[task_name])
                track_timesteps = np.array(self._track_timesteps[task_name])

                self.tracking_data[
                    f"Additional Info / {task_name} / Total reward (max)"
                ].append(np.max(track_rewards))
                self.tracking_data[
                    f"Additional Info / {task_name} / Total reward (min)"
                ].append(np.min(track_rewards))
                self.tracking_data[f"{task_name} / Total reward (mean)"].append(
                    np.mean(track_rewards)
                )

                if global_cfg.AGENT.ADAPTIVE_REWARD_NORMALIZATION:
                    m = self.reward_scalers[task_idx].mean.item()
                    s = self.reward_scalers[task_idx].std.item()
                    
                    self.tracking_data[f"{task_name} / Adaptive Reward"].extend(
                        [m, m+s, m-s]
                    )
                
                max_timesteps = global_cfg.TRAIN.EPISODE_LENGTH_S / (global_cfg.TRAIN.SIM_DT * global_cfg.TRAIN.DECIMATION)
                success_rate = np.sum(track_timesteps < max_timesteps) / len(track_timesteps)
                
                # 5 / (0.005 * 4) = 1 / 0.004 = 250
                self.tracking_data[
                    f"Additional Info / {task_name} / Episode / Total timesteps (max)"
                ].append(np.max(track_timesteps))
                self.tracking_data[
                    f"Additional Info / {task_name} / Episode / Total timesteps (min)"
                ].append(np.min(track_timesteps))
                self.tracking_data[
                    f"{task_name} / Episode / Total timesteps (mean)"
                ].append(np.mean(track_timesteps))
                self.tracking_data[
                    f"{task_name} / Episode / Success rate"
                ].append(success_rate)


def _my_post_interaction(self, timestep: int, timesteps: int) -> None:
    """Callback called after the interaction with the environment

    :param timestep: Current timestep
    :type timestep: int
    :param timesteps: Number of timesteps
    :type timesteps: int
    """
    timestep += 1

    # update best models and write checkpoints
    if (
        timestep > 1
        and self.checkpoint_interval > 0
        and not timestep % self.checkpoint_interval
    ):
        # update best models
        reward = np.mean(
            self.tracking_data.get("Reward / Total reward (mean)", -(2**31))
        )
        if reward > self.checkpoint_best_modules["reward"]:
            self.checkpoint_best_modules["timestep"] = timestep
            self.checkpoint_best_modules["reward"] = reward
            self.checkpoint_best_modules["saved"] = False
            self.checkpoint_best_modules["modules"] = {
                k: copy.deepcopy(self._get_internal_value(v))
                for k, v in self.checkpoint_modules.items()
            }
        # write checkpoints
        self.write_checkpoint(timestep, timesteps)

    # write to tensorboard
    # if timestep > 1 and self.write_interval > 0 and not timestep % self.write_interval:
    # logging starts after 5000 timesteps, so initial interactions are not logged.
    if (
        timestep >= 500
        and self.write_interval > 0
        and not timestep % self.write_interval
    ):
        # print(f"Logging data at timestep: {timestep}")
        self.write_tracking_data(timestep, timesteps)


def resume_from_checkpoint(self, path: str, skip_optimizer=False) -> None:
    """Load the model from the specified path

    The final storage device is determined by the constructor of the model

    :param path: Path to load the model from
    :type path: str
    """
    if version.parse(torch.__version__) >= version.parse("1.13"):
        modules = torch.load(path, map_location=self.device, weights_only=False)  # prevent torch:FutureWarning
    else:
        modules = torch.load(path, map_location=self.device)
    if type(modules) is dict:
        for name, data in modules.items():
            
            if skip_optimizer and "optimizer" in name:
                continue
            
            module = self.checkpoint_modules.get(name, None)
            if module is not None:
                if hasattr(module, "load_state_dict"):
                    module.load_state_dict(data)
                    # if hasattr(module, "eval"):
                    #     module.eval()
                else:
                    raise NotImplementedError
            else:
                logger.warning(f"Cannot load the {name} module. The agent doesn't have such an instance")


Agent.record_transition = _my_record_transition
Agent.post_interaction = _my_post_interaction
setattr(Agent, "resume_from_checkpoint", resume_from_checkpoint)

class MyAgent(Agent):
    def __init__(
        self,
        models,
        memory=None,
        observation_space=None,
        action_space=None,
        device=None,
        cfg=None,
        env_name_lst=[],
    ):
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=cfg,
        )
        self._track_rewards = mydequedict(
            {env_name: deque(maxlen=100) for env_name in env_name_lst}
        )
        self._track_timesteps = mydequedict(
            {env_name: deque(maxlen=100) for env_name in env_name_lst}
        )
        self._cumulative_rewards = {env_name: None for env_name in env_name_lst}
        self._cumulative_timesteps = {env_name: None for env_name in env_name_lst}
        
        if global_cfg.AGENT.ADAPTIVE_REWARD_NORMALIZATION:    
            self.reward_scalers = [AdaptiveRewardScaler(device=device) for _ in range(len(env_name_lst))]


def update_learning_rate_scheduler(agent_cfg, experiment_cfg):
    if experiment_cfg["learning_rate_scheduler"] == "LinearLR":
        agent_cfg["learning_rate_scheduler"] = LinearLR

        kwargs_list = experiment_cfg["learning_rate_scheduler_kwargs"]
        if len(kwargs_list) > 0:
            keys = kwargs_list[0::2]
            values = kwargs_list[1::2]
            kwargs_dict = dict(zip(keys, values))

            if "total_iters" in kwargs_dict:
                kwargs_dict['total_iters'] = global_cfg.TRAINER.TIMESTEPS 

            agent_cfg["learning_rate_scheduler_kwargs"] = kwargs_dict

    elif experiment_cfg["learning_rate_scheduler"] == "LinearWarmstartCosineLR":
        # the arguments come from separate set of configs
        agent_cfg["learning_rate_scheduler"] = LinearWarmupCosineAnnealingLR
        kwargs_dict = {
            "base_lr": global_cfg.LEARNING_RATE.BASE_LR,
            "warmup_steps": global_cfg.LEARNING_RATE.WARMUP_STEPS,
            "total_steps": global_cfg.TRAINER.TIMESTEPS,
            "warmup_factor": global_cfg.LEARNING_RATE.WARMUP_FACTOR,
        }
        agent_cfg['learning_rate_scheduler_kwargs'] = kwargs_dict
        
        
    # /////////////////////////////////////////////////////////////////
    experiment_cfg.pop("learning_rate_scheduler")
    experiment_cfg.pop("learning_rate_scheduler_kwargs")

    return agent_cfg, experiment_cfg


def update_noise_cfg(experiment_cfg, device):
    agent_type = global_cfg.AGENT_NAME
    if agent_type == "td3":
        if "GaussianNoise" in experiment_cfg["exploration"]["noise"]:
            experiment_cfg["exploration"]["noise"] = GaussianNoise(0, 0.1, device=device)
        if "GaussianNoise" in experiment_cfg["smooth_regularization_noise"]:
            experiment_cfg["smooth_regularization_noise"] = GaussianNoise(0, 0.2, device=device)
            
    return experiment_cfg

def update_train_cfg(experiment_cfg):
    agent_type = global_cfg.AGENT_NAME
    
    experiment_cfg["learning_starts"] = global_cfg.AGENT.LEARNING_STARTS    
    experiment_cfg["random_timesteps"] = global_cfg.AGENT.RANDOM_TIMESTEPS

    if agent_type == "random":
        experiment_cfg["learning_starts"] = global_cfg.TRAINER.TIMESTEPS
        experiment_cfg["random_timesteps"] = global_cfg.TRAINER.TIMESTEPS
        experiment_cfg["checkpoint_interval"] = global_cfg.TRAINER.TIMESTEPS
        
    return experiment_cfg


def get_agent_cfg_and_memory(env, device, only_agent=False):
    agent_type = global_cfg.AGENT_NAME
    
    if agent_type == "ppo":
        base_cfg = PPO_DEFAULT_CONFIG.copy()
        global_agent_cfg = global_cfg.AGENT.PPO
    elif agent_type == "ppo_rnn":
        base_cfg = PPO_DEFAULT_CONFIG.copy()
        global_agent_cfg = global_cfg.AGENT.PPO
    elif agent_type == "sac":
        base_cfg = SAC_DEFAULT_CONFIG.copy()
        global_agent_cfg = global_cfg.AGENT.SAC
    elif agent_type == "td3":
        base_cfg = TD3_DEFAULT_CONFIG.copy()
        global_agent_cfg = global_cfg.AGENT.TD3
    elif agent_type == "random":
        base_cfg = PPO_DEFAULT_CONFIG.copy()
        global_agent_cfg = global_cfg.AGENT.PPO
        
    experiment_cfg = get_lower_case_cfg(global_agent_cfg)
    experiment_cfg["experiment"] = get_lower_case_cfg(global_cfg.AGENT.EXPERIMENT)

    experiment_cfg = update_train_cfg(experiment_cfg)
    experiment_cfg = update_noise_cfg(experiment_cfg, device)
    
    base_cfg, experiment_cfg = update_learning_rate_scheduler(base_cfg, experiment_cfg)
    base_cfg.update(process_skrl_cfg(experiment_cfg))
     
    if only_agent:
        return base_cfg
       
    memory_size = base_cfg.get("rollouts", global_cfg.AGENT.ROLLOUTS)   # for sac, no rollout size is there
    print("Memory size: ", memory_size)
    memory = CustomMemory(
        memory_size=memory_size,
        num_envs=env.__getattr__("num_envs"),
        device=device,
    )
    
    return base_cfg, memory
    

def get_models(env, device):
    agent_type = global_cfg.AGENT_NAME
    
    obs_space = env.observation_space
    act_space = env.action_space
    
    if global_cfg.ACTION.DISCRETE:
        if agent_type == "ppo" or agent_type == "random":
            models = {
                "policy": StochasticPolicy_DiscreteAction(obs_space, act_space, device=device),
                "value": Value(obs_space, act_space, device=device)
            }
        else:
            raise NotImplementedError("Discrete action space not supported for SAC and TD3")

    else:    
        if agent_type == "ppo" or agent_type == "random":
            models = {
                "policy": StochasticPolicy(obs_space, act_space, device=device),
                "value": Value(obs_space, act_space, device=device)
            }
        elif agent_type == "ppo_rnn":
            models = {
                "policy": PolicyRNN(obs_space, act_space, device=device),
                "value": CriticRNN(obs_space, act_space, device=device)
            }
        elif agent_type == "sac":
            models = {
                "policy": StochasticPolicy(obs_space, act_space, device=device),
                "critic_1": QValueFunc(obs_space, act_space, device=device),
                "critic_2": QValueFunc(obs_space, act_space, device=device),
                "target_critic_1": QValueFunc(obs_space, act_space, device=device),
                "target_critic_2": QValueFunc(obs_space, act_space, device=device)
            }    
        elif agent_type == "td3":
            models = {
                "policy": DeterministicPolicy(obs_space, act_space, device=device),
                "target_policy": DeterministicPolicy(obs_space, act_space, device=device),
                "critic_1": QValueFunc(obs_space, act_space, device=device),
                "critic_2": QValueFunc(obs_space, act_space, device=device),
                "target_critic_1": QValueFunc(obs_space, act_space, device=device),
                "target_critic_2": QValueFunc(obs_space, act_space, device=device)
            }

    return models
        