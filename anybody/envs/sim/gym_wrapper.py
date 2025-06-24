import gymnasium as gym
import torch
from typing import Any, Tuple
from anybody.cfg import cfg
# from gymnasium.wrappers.monitoring import video_recorder
import wandb
from tensordict import TensorDict

from skrl.envs.wrappers.torch.isaaclab_envs import IsaacLabWrapper
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space, unflatten_tensorized_space

def preprocess(observation):
    _observation = observation['policy']
    for k, v in _observation.items():
        if k in ['obs_mask', 'act_mask']:
            _observation[k] = v.bool()
        else:
            _observation[k] = v.float()
            
    return TensorDict(_observation)

class MT_SKRLWrapper(IsaacLabWrapper):

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        actions = unflatten_tensorized_space(self.action_space, actions)
        observations, reward, terminated, truncated, self._info = self._env.step(actions)
        self._observations = preprocess(observations)
        return self._observations, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), self._info

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        if self._reset_once:
            observations, self._info = self._env.reset()
            self._observations = preprocess(observations)
            self._reset_once = False
        return self._observations, self._info
    

class VideoWrapper(gym.wrappers.RecordVideo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def step(self, action):

        """Steps through the environment using action, recording observations if :attr:`self.recording`."""
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        ) = self.env.step(action)
        # increment steps and episodes
        self.step_id += 1
        # if not self.is_vector_env:
        #     if terminateds or truncateds:
        #         self.episode_id += 1
        #         self.terminated = terminateds
        #         self.truncated = truncateds
        # elif terminateds[0] or truncateds[0]:
        #     self.episode_id += 1
        #     self.terminated = terminateds[0]
        #     self.truncated = truncateds[0]

        if self.recording:
            assert self.video_recorder is not None
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            # print("recorded_frames: ", len(self.video_recorder.recorded_frames))
            # print("recorded_frames: ", self.recorded_frames)
            if self.video_length > 0:
                if self.recorded_frames > self.video_length:
                    self.close_video_recorder()
            # else:
            #     if not self.is_vector_env:
            #         if terminateds or truncateds:
            #             self.close_video_recorder()
            #     elif terminateds[0] or truncateds[0]:
            #         self.close_video_recorder()

        elif self._video_enabled():
            self.start_video_recorder()

        return observations, rewards, terminateds, truncateds, infos

    def close_video_recorder(self):
        """Closes the video recorder if currently recording."""
        if self.recording:
            assert self.video_recorder is not None
            self.video_recorder.close()
            wandb.log({f"video/{cfg.GROUP_RUN_NAME}_{cfg.RUN_SEED}": wandb.Video(self.video_recorder.path)})
            
        self.recording = False
        self.recorded_frames = 1