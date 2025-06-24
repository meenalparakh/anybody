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

import torch
import tqdm
import copy
import sys

# from skrl.agents.torch import Agent
# from anybody.algos.multi_task_rl.agent import Agent
from skrl.envs.wrappers.torch import Wrapper
from skrl.trainers.torch.sequential import SEQUENTIAL_TRAINER_DEFAULT_CONFIG
from skrl.trainers.torch import Trainer
from anybody.utils.path_utils import import_overheat_module
# from gymnasium.wrappers.monitoring import video_recorder

import wandb
import os

from anybody.cfg import cfg as global_cfg


if global_cfg.CHECK_OVERHEATING:
    check_overheat = import_overheat_module()
else:
    check_overheat = None

class MySequentialLogTrainer(Trainer):
    """Sequential trainer with logging of episode information.

    This trainer inherits from the :class:`skrl.trainers.base_class.Trainer` class. It is used to
    train agents in a sequential manner (i.e., one after the other in each interaction with the
    environment). It is most suitable for on-policy RL agents such as PPO, A2C, etc.

    It modifies the :class:`skrl.trainers.torch.sequential.SequentialTrainer` class with the following
    differences:

    * It also log episode information to the agent's logger.
    * It does not close the environment at the end of the training.

    Reference:
        https://skrl.readthedocs.io/en/latest/api/trainers.html#base-class
    """

    def __init__(
        self,
        env: Wrapper,
        agents,
        agents_scope: list[int] | None = None,
        cfg: dict | None = None,
    ):
        """Initializes the trainer.

        Args:
            env: Environment to train on.
            agents: Agents to train.
            agents_scope: Number of environments for each agent to
                train on. Defaults to None.
            cfg: Configuration dictionary. Defaults to None.
        """
        # update the config
        _cfg = copy.deepcopy(SEQUENTIAL_TRAINER_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        # store agents scope
        agents_scope = agents_scope if agents_scope is not None else []
        # initialize the base class
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=_cfg)
        # init agents
        if self.env.num_agents > 1:
            for agent in self.agents:
                agent.init(trainer_cfg=self.cfg)
        else:
            self.agents.init(trainer_cfg=self.cfg)

        # if self.cfg.get("video_render", False):
        #     self.eval_timesteps_interval = self.cfg.get("eval_timesteps_interval", 5000)
        #     self.eval_timesteps = self.cfg.get("eval_timesteps", 300)
        #     self.video_folder = self.cfg.get("video_dir", None)
        #     self.video_recorder = None
        #     os.makedirs(self.video_folder, exist_ok=True)   


    def train(self):
        """Train the agents sequentially.

        This method executes the training loop for the agents. It performs the following steps:

        * Pre-interaction: Perform any pre-interaction operations.
        * Compute actions: Compute the actions for the agents.
        * Step the environments: Step the environments with the computed actions.
        * Record the environments' transitions: Record the transitions from the environments.
        * Log custom environment data: Log custom environment data.
        * Post-interaction: Perform any post-interaction operations.
        * Reset the environments: Reset the environments if they are terminated or truncated.

        """
        # init agent
        self.agents.init(trainer_cfg=self.cfg)
        self.agents.set_running_mode("train")
        # reset env
        states, infos = self.env.reset()
        # training loop
        
        n_env_interactions = 0        # multiples of 1e4
        fractional = 0
        for timestep in tqdm.tqdm(
            range(self.timesteps), disable=self.disable_progressbar
        ):    
            
            
            # add temperature check
            # if 'CHECK_OVERHEAT' in os.environ:
            
            
            
            if check_overheat:
                if timestep % 500 == 0:            
                    # print("#"*80)
                    # print(f"Timestep: {timestep}. Checking overheat status")
                    # print("#"*80)
                    if check_overheat.pause_needed():
                        check_overheat.pause()
                    
            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)
            # compute actions
            with torch.no_grad():
                actions = self.agents.act(
                    states, timestep=timestep, timesteps=self.timesteps
                )[0]
            # step the environments
            next_states, rewards, terminated, truncated, infos = self.env.step(actions)
            
            fractional += rewards.size(0)
            if fractional >= 1e5:
                n_env_interactions += (fractional // 1e5)
                fractional = fractional % 1e5
            
            # note: here we do not call render scene since it is done in the env.step() method
            # record the environments' transitions
            
            with torch.no_grad():
                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

            # log custom environment data            
            for env_name, info in infos.items():
                if "log" in info:
                    for k, v in info["log"].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            data_key = f"{env_name} / EpisodeInfo / {k}"
                            self.agents.track_data(data_key, v.item())
                            
            self.agents.track_data("n_env_interactions (1e5)", n_env_interactions + fractional * 1e-5)
            # self.agents.track_data("env_per_step", rewards.size(0))
 
            # post-interaction
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)
            # reset the environments
            # note: here we do not call reset scene since it is done in the env.step() method
            # update states
            states.copy_(next_states)
            
            # record an example trajectory for debugging after every some timesteps
            # if self.cfg.get("video_render", False) and (timestep % self.eval_timesteps_interval == 0) and (timestep > 0):
            #     print("Evaluating agent, timestep: ", timestep)
            #     states = self.run_evaluation(states, timestep) 
            
    def eval(self) -> None:
        """Evaluate the agents sequentially.

        This method executes the following steps in loop:

        * Compute actions: Compute the actions for the agents.
        * Step the environments: Step the environments with the computed actions.
        * Record the environments' transitions: Record the transitions from the environments.
        * Log custom environment data: Log custom environment data.
        """

        # set running mode
        self.agents.set_running_mode("eval")
        assert (
            self.num_simultaneous_agents == 1
        ), "This method is not allowed for simultaneous agents"
        assert self.env.num_agents == 1, "This method is not allowed for multi-agents"

        # reset env
        states, infos = self.env.reset()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps),
            disable=self.disable_progressbar,
            file=sys.stdout,
        ):
            # compute actions
            with torch.no_grad():
                # the last argument timesteps is not used in PPO
                actions = self.agents.act(
                    states, timestep=timestep, timesteps=self.timesteps
                )[0]

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(
                    actions
                )

                # render scene
                if not self.headless:
                    self.env.render()

                # write data to TensorBoard
                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                    # is_train=False,
                )
                # super(type(self.agents), self.agents).post_interaction(
                #     timestep=timestep, timesteps=self.timesteps
                # )

                if timestep > 1 and self.agents.write_interval > 0 and not timestep % self.agents.write_interval:
                    self.agents.write_tracking_data(timestep, self.timesteps)

                # log environment info
                # log custom environment data            
                for info in infos:
                    env_name = info.get('env_name', 'env')
                    if "log" in info:
                        for k, v in info["log"].items():
                            if isinstance(v, torch.Tensor) and v.numel() == 1:
                                data_key = f"{env_name} / EpisodeInfo / {k}"
                                self.agents.track_data(data_key, v.item())
 
            # reset environments
            states = next_states
            # if self.env.__getattr__('num_envs') > 1:
            #     states = next_states
            # else:
            #     raise NotImplementedError("have atleast 2 envs")
                

