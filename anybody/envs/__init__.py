# registering stick-v1 env (as an example)
import gymnasium as gym
import os


##
# Register Gym environments.
##

# gym.register(
#     id="BenchmarkEnv",
#     entry_point="isaaclab.envs:ManagerBasedMTRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.sim.mtrl_cfg:BenchmarkRLCfg",
#         "render_mode": False,
#         "device": f"cuda:{os.getenv('LOCAL_RANK', 0)}",
#     },
# )