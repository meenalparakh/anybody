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

from collections import deque

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config

from anybody.utils.utils import mydequedict
from skrl.agents.torch.sac import SAC
from anybody.cfg import cfg as global_cfg
from .base import AdaptiveRewardScaler, _my_record_transition, AdaptiveRewardScaler_MovingWindow
from .base import symexp, symlog

class MultiEnvSAC(SAC):
    """
    updates logging and tracking data for multi-environment training
    """

    def __init__(
        self,
        env_name_lst: list[str],
        models,
        memory=None,
        observation_space=None,
        action_space=None,
        device=None,
        cfg=None,
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
        print("MultiEnvSAC agent initialized")

        # initialize per env adaptive reward scaler
        # if global_cfg.AGENT.ADAPTIVE_REWARD_NORMALIZATION:
        #     if global_cfg.AGENT.SMOOTHING_WINDOW > 0:
        #         self.reward_scalers = [AdaptiveRewardScaler_MovingWindow(device=device) for _ in range(len(env_name_lst))]
        #     else:              
        #         self.reward_scalers = [AdaptiveRewardScaler(device=device) for _ in range(len(env_name_lst))]
        
    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        # gradient steps
        for gradient_step in range(self._gradient_steps):

            # sample a batch from memory
            (
                sampled_states,
                sampled_actions,
                sampled_rewards,
                sampled_next_states,
                sampled_terminated,
                sampled_truncated,
            ) = self.memory.sample(names=self._tensors_names, batch_size=self._batch_size)[0]

            # need to normalize rewards across the multiple-environments, by using a running mean and std per env.
            # if global_cfg.AGENT.ADAPTIVE_REWARD_NORMALIZATION:
            #     n_robos = len(self.reward_scalers)
            #     new_rewards = sampled_rewards.clone()   
            #     # assert n_robos * n_per_robo_env == rewards.shape[1], f"Number of environments: {rewards.shape[1]} is not a multiple of the number of tasks: {n_robos}"
                
            #     for i in range(n_robos):
            #         robo_indices = (sampled_states['robot_id'][:, 0, 0].long() == i)
            #         # self.reward_scalers[i].update(sampled_rewards[])
            #         selected_rewards = sampled_rewards[robo_indices, 0]
            #         self.reward_scalers[i].update(selected_rewards)
            #         new_rewards[robo_indices, 0] = self.reward_scalers[i].scale(selected_rewards)
                    
            #     sampled_rewards = new_rewards

            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):

                sampled_states = self._state_preprocessor(sampled_states, train=True)
                sampled_next_states = self._state_preprocessor(sampled_next_states, train=True)

                # compute target values
                with torch.no_grad():
                    next_actions, next_log_prob, _ = self.policy.act({"states": sampled_next_states}, role="policy")

                    target_q1_values, _, _ = self.target_critic_1.act(
                        {"states": sampled_next_states, "taken_actions": next_actions}, role="target_critic_1"
                    )
                    target_q2_values, _, _ = self.target_critic_2.act(
                        {"states": sampled_next_states, "taken_actions": next_actions}, role="target_critic_2"
                    )
                    
                    if global_cfg.AGENT.SYMLOG_RETURNS:
                        target_q_values = (
                            symexp(torch.min(target_q1_values, target_q2_values)) - self._entropy_coefficient * next_log_prob
                        )
                    else:
                        target_q_values = (
                            torch.min(target_q1_values, target_q2_values) - self._entropy_coefficient * next_log_prob
                        )
                        
                    target_values = (
                        sampled_rewards
                        + self._discount_factor
                        * (sampled_terminated | sampled_truncated).logical_not()
                        * target_q_values
                    )
                    
                    if global_cfg.AGENT.SYMLOG_RETURNS:
                        target_values = symlog(target_values)

                # compute critic loss
                critic_1_values, _, _ = self.critic_1.act(
                    {"states": sampled_states, "taken_actions": sampled_actions}, role="critic_1"
                )
                critic_2_values, _, _ = self.critic_2.act(
                    {"states": sampled_states, "taken_actions": sampled_actions}, role="critic_2"
                )

                critic_loss = (
                    F.mse_loss(critic_1_values, target_values) + F.mse_loss(critic_2_values, target_values)
                ) / 2

            # optimization step (critic)
            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()

            if config.torch.is_distributed:
                self.critic_1.reduce_parameters()
                self.critic_2.reduce_parameters()

            if self._grad_norm_clip > 0:
                self.scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(
                    itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()), self._grad_norm_clip
                )

            self.scaler.step(self.critic_optimizer)

            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                # compute policy (actor) loss
                actions, log_prob, _ = self.policy.act({"states": sampled_states}, role="policy")
                critic_1_values, _, _ = self.critic_1.act(
                    {"states": sampled_states, "taken_actions": actions}, role="critic_1"
                )
                critic_2_values, _, _ = self.critic_2.act(
                    {"states": sampled_states, "taken_actions": actions}, role="critic_2"
                )

                if global_cfg.AGENT.SYMLOG_RETURNS: 
                    policy_loss = (
                        self._entropy_coefficient * log_prob - symexp(torch.min(critic_1_values, critic_2_values))
                    ).mean()
                else:
                    policy_loss = (
                        self._entropy_coefficient * log_prob - torch.min(critic_1_values, critic_2_values)
                    ).mean()

            # optimization step (policy)
            self.policy_optimizer.zero_grad()
            self.scaler.scale(policy_loss).backward()

            if config.torch.is_distributed:
                self.policy.reduce_parameters()

            if self._grad_norm_clip > 0:
                self.scaler.unscale_(self.policy_optimizer)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)

            self.scaler.step(self.policy_optimizer)

            # entropy learning
            if self._learn_entropy:
                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    # compute entropy loss
                    entropy_loss = -(self.log_entropy_coefficient * (log_prob + self._target_entropy).detach()).mean()

                # optimization step (entropy)
                self.entropy_optimizer.zero_grad()
                self.scaler.scale(entropy_loss).backward()
                self.scaler.step(self.entropy_optimizer)

                # compute entropy coefficient
                self._entropy_coefficient = torch.exp(self.log_entropy_coefficient.detach())

            self.scaler.update()  # called once, after optimizers have been stepped

            # update target networks
            self.target_critic_1.update_parameters(self.critic_1, polyak=self._polyak)
            self.target_critic_2.update_parameters(self.critic_2, polyak=self._polyak)

            # update learning rate
            if self._learning_rate_scheduler:
                self.policy_scheduler.step()
                self.critic_scheduler.step()

            # record data
            if self.write_interval > 0:
                self.track_data("Loss / Policy loss", policy_loss.item())
                self.track_data("Loss / Critic loss", critic_loss.item())

                self.track_data("Q-network / Q1 (max)", torch.max(critic_1_values).item())
                self.track_data("Q-network / Q1 (min)", torch.min(critic_1_values).item())
                self.track_data("Q-network / Q1 (mean)", torch.mean(critic_1_values).item())

                self.track_data("Q-network / Q2 (max)", torch.max(critic_2_values).item())
                self.track_data("Q-network / Q2 (min)", torch.min(critic_2_values).item())
                self.track_data("Q-network / Q2 (mean)", torch.mean(critic_2_values).item())

                self.track_data("Target / Target (max)", torch.max(target_values).item())
                self.track_data("Target / Target (min)", torch.min(target_values).item())
                self.track_data("Target / Target (mean)", torch.mean(target_values).item())

                if self._learn_entropy:
                    self.track_data("Loss / Entropy loss", entropy_loss.item())
                    self.track_data("Coefficient / Entropy coefficient", self._entropy_coefficient.item())

                if self._learning_rate_scheduler:
                    self.track_data("Learning / Policy learning rate", self.policy_scheduler.get_last_lr()[0])
                    self.track_data("Learning / Critic learning rate", self.critic_scheduler.get_last_lr()[0])
