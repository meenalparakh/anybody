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
from copy import deepcopy

import random
# import collections
import torch
from torch import nn
from anybody.utils.utils import mydequedict
from skrl.agents.torch.ppo import PPO_RNN

from typing import Any
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl import config
from anybody.cfg import cfg as global_cfg

import torch.nn.functional as F
import itertools

from .base import AdaptiveRewardScaler, _my_record_transition, AdaptiveRewardScaler_MovingWindow, symlog, symexp
from anybody.algos.multi_task_rl.agents.pc_grad_optimizer import PCGrad

class MultiEnvPPO_RNN(PPO_RNN):
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
        print(
            "//////////////////////// MultiEnvPPO-RNN agent initialized ///////////////////"
        )

        # set up optimizer and learning rate scheduler
        if self.policy is not None and self.value is not None:
            if self.policy is self.value:
                self.optimizer = torch.optim.Adam(
                    self.policy.parameters(), lr=self._learning_rate
                )
            else:
                self.optimizer = torch.optim.Adam(
                    itertools.chain(self.policy.parameters(), self.value.parameters()),
                    lr=self._learning_rate,
                    # eps=1e-5,
                )
            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(
                    self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )

            self.checkpoint_modules["optimizer"] = self.optimizer

        if global_cfg.AGENT.PCGRAD:
            self.pcgrad_optimizer = PCGrad(self.optimizer)
            
            
        self.n_tasks = len(env_name_lst)
        # initialize per env adaptive reward scaler
        if global_cfg.AGENT.ADAPTIVE_REWARD_NORMALIZATION:
            if global_cfg.AGENT.SMOOTHING_WINDOW > 0:
                self.reward_scalers = [AdaptiveRewardScaler_MovingWindow(device=device) for _ in range(len(env_name_lst))]
            else:              
                self.reward_scalers = [AdaptiveRewardScaler(device=device) for _ in range(len(env_name_lst))]
        
        if global_cfg.AGENT.EMA_CRITIC:
            # Initialize EMA parameters
            self.ema_tau = 0.995  # EMA decay rate
            self._slow_value = deepcopy(self.value)
            self._slow_value.train(False)  # set to evaluation mode


    def update_ema_parameters(self, timestep):
        """Update EMA parameters for the critic network."""
        
        if timestep % global_cfg.AGENT.EMA_UPDATE_INTERVAL == 0:
            
            for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                d.data.mul_(self.ema_tau).add_(s.data, alpha=1 - self.ema_tau)
            
            
    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        rnn = {"rnn": self._rnn_initial_states["policy"]} if self._rnn else {}
        # robo_indices = torch.arange(len(self.reward_scalers)).unsqueeze(1).repeat(1, global_cfg.TRAIN.NUM_ENVS).reshape(-1)

        # sample random actions
        # TODO: fix for stochasticity, rnn and log_prob
        if timestep < self._random_timesteps:
            actions, log_prob, outputs = self.policy.random_act(
                {"states": self._state_preprocessor(states), **rnn}, role="policy"
            )

        else:
            # sample stochastic actions
            actions, log_prob, outputs = self.policy.act(
                {"states": self._state_preprocessor(states), **rnn}, role="policy"
            )

        self._current_log_prob = log_prob

        if self._rnn:
            self._rnn_final_states["policy"] = outputs.get("rnn", [])

        return actions, log_prob, outputs


    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        
        if global_cfg.AGENT.EMA_CRITIC:
            self.update_ema_parameters(timestep)
        
        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> torch.Tensor:
            """Compute the Generalized Advantage Estimator (GAE)

            :param rewards: Rewards obtained by the agent
            :type rewards: torch.Tensor
            :param dones: Signals to indicate that episodes have ended
            :type dones: torch.Tensor
            :param values: Values obtained by the agent
            :type values: torch.Tensor
            :param next_values: Next values obtained by the agent
            :type next_values: torch.Tensor
            :param discount_factor: Discount factor
            :type discount_factor: float
            :param lambda_coefficient: Lambda coefficient
            :type lambda_coefficient: float

            :return: Generalized Advantage Estimator
            :rtype: torch.Tensor
            """
            # need to normalize rewards across the multiple-environments, by using a running mean and std per env.
            # if global_cfg.AGENT.ADAPTIVE_REWARD_NORMALIZATION:
            #     n_per_robo_env = global_cfg.TRAIN.NUM_ENVS            
            #     n_robos = len(self.reward_scalers)
            #     new_rewards = rewards.clone()
            #     # assert n_robos * n_per_robo_env == rewards.shape[1], f"Number of environments: {rewards.shape[1]} is not a multiple of the number of tasks: {n_robos}"            
            #     for i in range(n_robos):
            #         self.reward_scalers[i].update(rewards[:, i * n_per_robo_env : (i + 1) * n_per_robo_env, :].reshape(-1))
            #         new_rewards[:, i * n_per_robo_env : (i + 1) * n_per_robo_env, : ] = self.reward_scalers[i].scale(rewards[:, i * n_per_robo_env : (i + 1) * n_per_robo_env, :])

            #     rewards = new_rewards
                
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages


        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            
            if global_cfg.AGENT.EMA_CRITIC:
                rnn = {"rnn": self._rnn_initial_states["value"]} if self._rnn else {}
                self._slow_value.train(False)
                last_values, _, _ = self._slow_value.act(
                    {"states": self._state_preprocessor(self._current_next_states.float()), **rnn}, role="value"
                )
                last_values = self._value_preprocessor(last_values, inverse=True)
            else:                
                self.value.train(False)
                rnn = {"rnn": self._rnn_initial_states["value"]} if self._rnn else {}
                last_values, _, _ = self.value.act(
                    {"states": self._state_preprocessor(self._current_next_states.float()), **rnn}, role="value"
                )
                self.value.train(True)
                last_values = self._value_preprocessor(last_values, inverse=True)

            if global_cfg.AGENT.SYMLOG_RETURNS:
                # the last values are actually the symlog of the actual values
                last_values = symexp(last_values)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)


        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(
            names=self._tensors_names, mini_batches=self._mini_batches, sequence_length=self._rnn_sequence_length
        )

        rnn_policy, rnn_value = {}, {}
        if self._rnn:
            sampled_rnn_batches = self.memory.sample_all(
                names=self._rnn_tensors_names,
                mini_batches=self._mini_batches,
                sequence_length=self._rnn_sequence_length,
            )

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for i, (
                sampled_states,
                sampled_actions,
                sampled_terminated,
                sampled_truncated,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in enumerate(sampled_batches):

                if self._rnn:
                    if self.policy is self.value:
                        rnn_policy = {
                            "rnn": [s.transpose(0, 1) for s in sampled_rnn_batches[i]],
                            "terminated": sampled_terminated | sampled_truncated,
                        }
                        rnn_value = rnn_policy
                    else:
                        rnn_policy = {
                            "rnn": [
                                s.transpose(0, 1)
                                for s, n in zip(sampled_rnn_batches[i], self._rnn_tensors_names)
                                if "policy" in n
                            ],
                            "terminated": sampled_terminated | sampled_truncated,
                        }
                        rnn_value = {
                            "rnn": [
                                s.transpose(0, 1)
                                for s, n in zip(sampled_rnn_batches[i], self._rnn_tensors_names)
                                if "value" in n
                            ],
                            "terminated": sampled_terminated | sampled_truncated,
                        }

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):

                    sampled_states = self._state_preprocessor(sampled_states, train=not epoch)

                    _, next_log_prob, _ = self.policy.act(
                        {"states": sampled_states, "taken_actions": sampled_actions, **rnn_policy}, role="policy"
                    )

                    # compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # early stopping with KL divergence
                    if self._kl_threshold and kl_divergence > self._kl_threshold:
                        break

                    # compute entropy loss
                    if self._entropy_loss_scale:
                        entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    # compute policy loss
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )
                    predicted_values, _, _ = self.value.act({"states": sampled_states, **rnn_value}, role="value")


                    # compute value loss
                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                        )
                        
                    if global_cfg.AGENT.SYMLOG_RETURNS:
                        with torch.no_grad():   
                            symlog_returns = symlog(sampled_returns)
                            
                        value_targets = symlog_returns
                    else:
                        value_targets = sampled_returns
                        
                    
                    if global_cfg.AGENT.PCGRAD:
                        robo_id_masks = sampled_states['robot_id'].squeeze(-1).long()
                        assert robo_id_masks.shape == surrogate.shape, f"robo_id_masks shape: {robo_id_masks.shape} is not equal to surrogate shape: {surrogate.shape}"
                        _per_policy_loss = -torch.min(surrogate, surrogate_clipped)
                        _per_value_loss = self._value_loss_scale * nn.MSELoss(reduction='none')(predicted_values, value_targets)
                        task_losses = []
                        for i in random.sample(range(self.n_tasks), 2):
                            task_mask = robo_id_masks == i
                            task_loss = _per_policy_loss[task_mask].mean() + _per_value_loss[task_mask].mean()
                            task_losses.append(task_loss)
                            
                        task_losses.append(entropy_loss)
                    
                    else:
                        
                        policy_loss = -torch.min(surrogate, surrogate_clipped).mean()
                        value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)


                if global_cfg.AGENT.PCGRAD:
                    self.pcgrad_optimizer.zero_grad()
                    self.pcgrad_optimizer.pc_backward(task_losses)

                    if config.torch.is_distributed:
                        self.policy.reduce_parameters()
                        if self.policy is not self.value:
                            self.value.reduce_parameters()

                    self.pcgrad_optimizer.step()
                    policy_loss = _per_policy_loss.mean()
                    value_loss = _per_value_loss.mean()
                
                else:
                    # optimization step
                    self.optimizer.zero_grad()
                    self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

                    if config.torch.is_distributed:
                        self.policy.reduce_parameters()
                        if self.policy is not self.value:
                            self.value.reduce_parameters()

                    if self._grad_norm_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        if self.policy is self.value:
                            nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                        else:
                            nn.utils.clip_grad_norm_(
                                itertools.chain(self.policy.parameters(), self.value.parameters()), self._grad_norm_clip
                            )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    # reduce (collect from all workers/processes) KL in distributed runs
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        # record data
        self.track_data(
            "Policy / Policy loss",
            cumulative_policy_loss / (self._learning_epochs * self._mini_batches),
        )
        self.track_data(
            "Policy / Value loss",
            cumulative_value_loss / (self._learning_epochs * self._mini_batches),
        )
        if self._entropy_loss_scale:
            self.track_data(
                "Policy / Entropy loss",
                cumulative_entropy_loss / (self._learning_epochs * self._mini_batches),
            )

        self.track_data(
            "Policy / Standard deviation",
            self.policy.distribution(role="policy").stddev.mean().item(),
        )

        if self._learning_rate_scheduler:
            self.track_data("Policy / Learning rate", self.scheduler.get_last_lr()[0])



    def record_transition(
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
    ) -> None:
        """Record an environment transition in memory

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
        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )


        self.set_running_mode("eval")

        if self.memory is not None:
            self._current_next_states = next_states

            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # compute values
            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                rnn = {"rnn": self._rnn_initial_states["value"]} if self._rnn else {}
                
                if global_cfg.AGENT.EMA_CRITIC:
                    values, _, outputs = self._slow_value.act({"states": self._state_preprocessor(states), **rnn}, role="value")
                else:                
                    values, _, outputs = self.value.act({"states": self._state_preprocessor(states), **rnn}, role="value")
                values = self._value_preprocessor(values, inverse=True)

                if global_cfg.AGENT.SYMLOG_RETURNS:
                    values = symexp(values)

            # time-limit (truncation) bootstrapping
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated

            # package RNN states
            rnn_states = {}
            if self._rnn:
                rnn_states.update(
                    {f"rnn_policy_{i}": s.transpose(0, 1) for i, s in enumerate(self._rnn_initial_states["policy"])}
                )
                if self.policy is not self.value:
                    rnn_states.update(
                        {f"rnn_value_{i}": s.transpose(0, 1) for i, s in enumerate(self._rnn_initial_states["value"])}
                    )

            # storage transition in memory
            self.memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                log_prob=self._current_log_prob,
                values=values,
                **rnn_states,
            )
            for memory in self.secondary_memories:
                memory.add_samples(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    log_prob=self._current_log_prob,
                    values=values,
                    **rnn_states,
                )

        # update RNN states
        if self._rnn:
            self._rnn_final_states["value"] = (
                self._rnn_final_states["policy"] if self.policy is self.value else outputs.get("rnn", [])
            )

            # reset states if the episodes have ended
            finished_episodes = (terminated | truncated).nonzero(as_tuple=False)
            if finished_episodes.numel():
                for rnn_state in self._rnn_final_states["policy"]:
                    rnn_state[:, finished_episodes[:, 0]] = 0
                if self.policy is not self.value:
                    for rnn_state in self._rnn_final_states["value"]:
                        rnn_state[:, finished_episodes[:, 0]] = 0

            self._rnn_initial_states = self._rnn_final_states