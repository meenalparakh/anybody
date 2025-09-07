import torch
import gymnasium as gym
from anybody.algos.multi_task_rl.agents.base import get_models
from anybody.cfg import cfg
from anybody.utils.path_utils import get_global_cfgs_dir

class DummyEnv:

# Observation space: Dict(
    # 'act_mask': Box(-inf, inf, (22, 1), float32), 
    # 'obj': Box(-inf, inf, (1, 13), float32), 
    # 'obs_mask': Box(-inf, inf, (26, 1), float32), 
    # 'obstacle': Box(-inf, inf, (1, 13), float32), 
    # 'robo_base': Box(-inf, inf, (1, 7), float32), 
    # 'robo_goal': Box(-inf, inf, (1, 7), float32), 
    # 'robo_link': Box(-inf, inf, (22, 48), float32), 
    # 'robot_id': Box(-inf, inf, (1, 1), float32))
# Action space: Box(-inf, inf, (22,), float32)

    observation_space = gym.spaces.Dict({
        'act_mask': gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(22, 1), dtype='float32'),
        'obj': gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(1, 13), dtype='float32'),
        'obs_mask': gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(26, 1), dtype='float32'),
        'obstacle': gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(1, 13), dtype='float32'),
        'robo_base': gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(1, 7), dtype='float32'),
        'robo_goal': gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(1, 7), dtype='float32'),
        'robo_link': gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(22, 48), dtype='float32'),
        'robot_id': gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(1, 1), dtype='float32')
    })
    action_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(22,), dtype='float32')


def set_cfg_tf():
    cfg.ACTION.DISCRETE = False
    cfg.merge_from_file(get_global_cfgs_dir() / "experiment_cfgs/mt_tf_reach.yaml")
    
    cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD = 256
    cfg.MODEL.TRANSFORMER.NLAYERS = 3
    cfg.MODEL.LIMB_EMBED_SIZE = 16

def set_cfg_mlp():
    cfg.ACTION.DISCRETE = False
    cfg.merge_from_file(get_global_cfgs_dir() / "experiment_cfgs/mt_mlp_reach.yaml")
    
    cfg.MODEL.MLP.EMBED_DIM = 72
    cfg.MODEL.MLP.N_LAYERS = 3
    cfg.MODEL.LIMB_EMBED_SIZE = 16


if __name__ == "__main__":
    dummy_env = DummyEnv()
    
    
    set_cfg_tf()
    models = get_models(env=dummy_env, device="cuda")
    total_params = sum(p.numel() for p in models['policy'].parameters())
    print(f"Total parameters Tf: {total_params}")

    set_cfg_mlp()
    models = get_models(env=dummy_env, device="cuda")
    total_params = sum(p.numel() for p in models['policy'].parameters())
    print(f"Total parameters Mlp: {total_params}")
