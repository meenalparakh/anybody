"""Configuration file (powered by YACS)."""
import torch
import copy
import os
from anybody.utils.yacs import CfgNode as CN

# Global config object
_C = CN()

# Example usage:
#   from core.config import cfg
cfg = _C

_C.IS_DISTRIBUTED = False
# _C.RAND = "0"  # string appended at the end of experiment name (to distinguish between runs started at the same time)
_C.OVERRIDE_CFGNAME = ''  # name to a config file (inside bench/train_cfgs) to override the default config
_C.PROJECT_NAME = 'scratch'     # name of the wandb project
_C.EXPERIMENT_NAME = "None"      # by default, "{group_run_name}_{seed}" is used. but useful for overriding the name
# _C.RUN_NAME_PREFIX = 'run'
_C.NUM_WORKERS = 40
_C.NUM_GPUS = 1
_C.CHECK_OVERHEATING = False
_C.ALLOW_PROBLEM_SPEC_GEN = True

_C.SEARCH_CHECKPOINT = False

_C.RECOMPUTE_GOAL_RANDOMIZATION_CFGS = False
_C.NUM_GOAL_RANDOMIZATIONS = 10
# _C.ROBOT_COLOR = (0.1, 0.2, 0.2)
_C.ROBOT_COLOR = (189/255.0, 191/255.0, 227/255.0)
_C.GOAL_MARKER_COLOR = (0.9, 0.0, 0.0)
_C.CURRENT_MARKER_COLOR = (0.1, 0.1, 0.1)
_C.GOAL_MARKER_TYPE = 'frame'
_C.FIXED_GOAL = False
_C.FIXED_INITIAL_STATE = False
_C.FORCE_ROBO_LINK_USD_CONVERSION = False
_C.FORCE_PROBLEM_SPEC_GEN = False
_C.FORCE_RECOMPUTE_LINK_INFO = True
_C.SAVE_DIFFIK_TRAJ = True

_C.GROUP_RUN_NAME = None

# negative value indicates that the damping and stiffness values are to be set automatically
_C.DAMPING = -1.0      
_C.STIFFNESS = -1.0
# ----------------------------------------------------------------------------#
# Misc Options
# ----------------------------------------------------------------------------#

# Config destination (in OUT_DIR)
_C.CFG_DEST = "config.yaml"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries. This is the only seed
# which will effect env variations.
_C.RUN_SEED = 1
_C.LOGGER = "wandb"

_C.EVAL = False
_C.EVAL_CHECKPOINT = None
_C.TRAIN_CHECKPOINT = None
_C.IS_FINETUNING = False
_C.EVAL_ON_TEST = False
_C.SCRATCH_TRAIN_TEST = False

# --------------------------------------------------------------------------- #
# Environment Options
# --------------------------------------------------------------------------- #
# used ony for single environment training
_C.ENV = CN()
_C.ENV.TASK = "reach"
_C.ENV.ROBOT = "stick"
_C.ENV.TARGET = "joint"
_C.ENV.VARIATION = "v1"
_C.ENV.NAME = None  # set by the environment
_C.ENV.SEED = 100


_C.BENCHMARK_TASK = None
_C.SE_TASK = None     # {robo_cat}/{robo_variation}/{task_name}

# used for multi environment training
_C.MULTIENV = CN()
_C.MULTIENV.TASKS = []
_C.MULTIENV.ROBOTS = []
_C.MULTIENV.VARIATIONS = []
_C.MULTIENV.SEEDS = []
_C.MULTIENV.TASK_RESET_STEP_EVERY = 10000
_C.MULTIENV.TASK_SPACING = 3.0
_C.MULTIENV.ENV_SPACING = 2.0


_C.CARTPOLE = CN()
_C.CARTPOLE.OBS_DIM = 6
_C.CARTPOLE.MLP_N_LAYERS = 1
_C.CARTPOLE.MLP_HIDDEN_DIMS = 128

# test environments
_C.TEST_ENVS = CN()
_C.TEST_ENVS.TASKS = []
_C.TEST_ENVS.ROBOTS = []
_C.TEST_ENVS.TARGETS = []
_C.TEST_ENVS.VARIATIONS = []
_C.TEST_ENVS.SEEDS = []


_C.BENCH = CN()
_C.BENCH.MAX_NUM_LINKS = 22
_C.BENCH.MAX_NUM_ROBOTS = 1
_C.BENCH.MAX_NUM_OBJECTS = 1
_C.BENCH.MAX_NUM_OBSTACLES = 1

# action dimension: max_robots * (1 + max_links) + max_objects
#                 = 2 * (1 + 32) + 10 = 76

_C.OBSERVATION = CN()
_C.OBSERVATION.MINIMAL = True
_C.OBSERVATION.LINK_POSE = False
_C.OBSERVATION.LINK_POSE_DIM = 7
_C.OBSERVATION.MASK_ROBO_MORPH = False
_C.OBSERVATION.COMPRESSED = True

_C.OBSERVATION.FILL_UP_VAL = -2.0   

# 2 link idx vecs + 6 link geom vec + 7 link origin vec
# _C.OBSERVATION.LINK_VEC_DIM = 2 * _C.OBSERVATION.ROBO_LINK_IDX_ENCODE_DIM + 6 + 7
# _C.OBSERVATION.JOINT_VEC_DIM = 3 + 2 + 3 + 7
# _C.OBSERVATION.GOAL_VEC_DIM = 1 + 8 + 1 + 7
_C.OBSERVATION.JOINT_VALUE_ENCODER = CN()
_C.OBSERVATION.JOINT_VALUE_ENCODER.TYPE = 'sinusoidal'
_C.OBSERVATION.JOINT_VALUE_ENCODER.DIM = 16
_C.OBSERVATION.JOINT_VALUE_ENCODER.MIN_WAVELENGTH = 0.1
_C.OBSERVATION.JOINT_VALUE_ENCODER.MAX_WAVELENGTH = 12.0

_C.OBSERVATION.ROBO_LINK_IDX_ENCODE_DIM = 1

_C.OBSERVATION.ROBO_BASE_DIM = 7
_C.OBSERVATION.LINK_VEC_DIM = 2 * _C.OBSERVATION.ROBO_LINK_IDX_ENCODE_DIM + 6 + 7
_C.OBSERVATION.JOINT_VEC_DIM = 3 + 3 + 7     # joint type, joint axis, joint origin
_C.OBSERVATION.EE_FLAG_DIM = 4    # repeat the ee flag 8 times
_C.OBSERVATION.ROBO_LINK_DIM = _C.OBSERVATION.LINK_VEC_DIM + _C.OBSERVATION.JOINT_VEC_DIM + _C.OBSERVATION.JOINT_VALUE_ENCODER.DIM + cfg.OBSERVATION.EE_FLAG_DIM
_C.OBSERVATION.ROBO_GOAL_DIM = 7

_C.OBSERVATION.OBJ_DIM = 6 + 7   # 6 for shape, 7 for pose

_C.OBSERVATION.ADDITIONAL_DIM = 4      # dims beside the robot links: robo_base, goal, object, obstacle
_C.OBSERVATION.PREV_ACTION = False
_C.OBSERVATION.NORM_EPSILON = 1e-4
_C.OBSERVATION.ROBOT_ID = True
_C.OBSERVATION.VOXEL_SIZE = 0.1    # 10 cm voxel size
_C.OBSERVATION.PCD_POINTS = 2048
_C.OBSERVATION.PCD_NOISE = 0.01
_C.OBSERVATION.WORKPACE_DIMS = [-0.5, 0.5, -0.5, 0.5, 0.0, 1.0]
_C.OBSERVATION.HIGH_DIM = False

_C.COMMAND = CN()
_C.COMMAND.RESAMPLING_TIME_RANGE = (4, 6)
_C.COMMAND.DEBUG_VIS = True
_C.COMMAND.FULL_VISUALIZATION = False
# _C.COMMAND.JOINT_GOAL_EMBED_DIM = _C.OBSERVATION.JOINT_VALUE_ENCODER.DIM

_C.REWARD = CN()
_C.REWARD.JOINT_POS_WEIGHT = 10.0
_C.REWARD.JOINT_POS_THRESHOLD = 100.0
_C.REWARD.EE_POSE_POS_ONLY = True
_C.REWARD.SPARSE_REWARD = False
_C.REWARD.EE_POSE_WEIGHT = 50.0
_C.REWARD.EE_POSE_THRESHOLD = 100.0
_C.REWARD.OBJ_POSE_WEIGHT = 10.0
_C.REWARD.OBJ_POSE_POS_ONLY = True 
_C.REWARD.OBJ_POSE_THRESHOLD = 100.0
_C.REWARD.OBJ_SIMPLE_REWARD = False           # set by the train script, by checking if the task is push-simple
_C.REWARD.COMPLETE_REWARD = 50.0    #5.0   for push - where the env terminates with a complete reward
_C.REWARD.REACH_REWARD = 4.0
_C.REWARD.EE_OBJ_WEIGHT = 2.5
# _C.REWARD.EE_OBJ_THRESHOLD = 0.00001
_C.REWARD.OBJ_DIST_WEIGHT = 100.0
_C.REWARD.JOINT_LIMITS_WEIGHT = 0.05
_C.REWARD.JOINT_ACC_WEIGHT = 1e-5
_C.REWARD.JOINT_MAG_WEIGHT = 1.0
_C.REWARD.NORMALIZE = True

_C.ACTION = CN()
_C.ACTION.SCALE = 1.0
_C.ACTION.ABSOLUTE = False
_C.ACTION.DISCRETE = False
_C.ACTION.NUM_BINS = 11       # one bin is for sign, the other 10 are for the magnitude
_C.ACTION.BIN_BASE = 2.0      # exponentially based bins with base 2.0

_C.TERMINATION = CN()
_C.TERMINATION.SUCCESS_POSE_THRESHOLD = 0.0009
_C.TERMINATION.SUCCESS_JPOS_THRESHOLD = 0.01
_C.TERMINATION.SUCCESS_OBJ_POSE_THRESHOLD = 0.01
_C.TERMINATION.SUCCESS_OBJ_X_DIST_THRESHOLD = 0.2
_C.TERMINATION.REACH_EARLY_TERMINATION = False 
_C.TERMINATION.PUSH_EARLY_TERMINATION = True


_C.CURRICULUM = CN()
_C.CURRICULUM.ACTIVE = True
_C.CURRICULUM.START_TERMINATION_THD = 0.1   # this is the factor by which the actual threshold is multiplied
_C.CURRICULUM.N_UPDATES = 10
_C.CURRICULUM.NUM_STEPS = 0 # NEED TO BE FILLED BY THE TRAIN SCRIPT

_C.TRAIN = CN()
_C.TRAIN.NUM_ENVS_PER_TASK = 64
_C.TRAIN.EPISODE_LENGTH_S = 6.0
_C.TRAIN.DECIMATION = 4
_C.TRAIN.SIM_DT = 0.005
# _C.TRAIN.EPISODE_LENGTH_S = 20.0
# _C.TRAIN.DECIMATION = 2
# _C.TRAIN.SIM_DT = 0.005

_C.EVAL_EPISODE_LENGTH_S = 5.0
_C.EVAL_VIDEO_LENGTH = 1000
_C.THRESHOLD_GROUND_HT = -2.0

# --------------------------------------------------------------------------- #
# SKRL PPO settings
# --------------------------------------------------------------------------- #
# PPO agent configuration (field names are from PPO_DEFAULT_CONFIG)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html

_C.AGENT_NAME = 'ppo'

_C.AGENT = CN()
_C.AGENT.ROLLOUTS = 4096      # 15625
_C.AGENT.RANDOM_TIMESTEPS = 1000
_C.AGENT.LEARNING_STARTS = 1000
_C.AGENT.REWARD_SHAPER = False
_C.AGENT.ADAPTIVE_REWARD_NORMALIZATION = False
_C.AGENT.TASK_GRAD_NORM = False
_C.AGENT.SMOOTHING_WINDOW = 10000
_C.AGENT.EMA_CRITIC = False
_C.AGENT.PCGRAD = False
_C.AGENT.TASK_REWEIGHTING = False
_C.AGENT.TASK_WEIGHTS_RANGE = (-5.0, 5.0)
_C.AGENT.EMA_UPDATE_INTERVAL = 500
_C.AGENT.SYMLOG_RETURNS = False
# _C.AGENT.PPO = get_agent_cfg("ppo")
# _C.AGENT.SAC = get_agent_cfg("sac")
# _C.AGENT.DDPG = get_agent_cfg("ddpg")
# _C.AGENT.TRPO = get_agent_cfg("trpo")
# _C.AGENT.TD3 = get_agent_cfg("td3")
# initialized at the end of this file
_C.AGENT.PPO = None
_C.AGENT.SAC = None

# agent specific settings loaded from algos folder

_C.AGENT.EXPERIMENT = CN()
_C.AGENT.EXPERIMENT.DIRECTORY = "Reach-Stick-Joint-V1-100"
_C.AGENT.EXPERIMENT.BASE_DIRECTORY = "Reach-Stick-Joint-V1"
_C.AGENT.EXPERIMENT.EXPERIMENT_NAME = ""
_C.AGENT.EXPERIMENT.WRITE_INTERVAL = 1000
_C.AGENT.EXPERIMENT.CHECKPOINT_INTERVAL = 2000
_C.AGENT.EXPERIMENT.WANDB = None  # to be set by the train script
_C.AGENT.EXPERIMENT.WANDB_KWARGS = CN()
_C.AGENT.EXPERIMENT.WANDB_KWARGS.PROJECT = None  # to be set by the train script
_C.AGENT.EXPERIMENT.WANDB_KWARGS.DIR = None  # to be set by the train script

_C.TRAINER = CN()
_C.TRAINER.TIMESTEPS = 1000000 + 2
_C.TRAINER.EVAL_TIMESTEPS_INTERVAL = 2000
_C.TRAINER.EVAL_TIMESTEPS = 300
_C.TRAINER.ENVIRONMENT_INFO = "log"
_C.TRAINER.VIDEO_DIR = ""
_C.TRAINER.VIDEO_RENDER = False

# --------------------------------------------------------------------------- #
# Model Options for RL transformer model
# --------------------------------------------------------------------------- #
_C.MODEL = CN()

_C.MODEL.NORMALIZE_OBS = False
_C.MODEL.IMPLICIT_OBS = True
_C.MODEL.POLICY = CN()
_C.MODEL.POLICY.CLIP_ACTIONS = True
_C.MODEL.POLICY.CLIP_LOG_STD = True
_C.MODEL.POLICY.INITIAL_LOG_STD = 1.0
_C.MODEL.POLICY.INITIAL_TEMPERATURE = 20.0
_C.MODEL.POLICY.MIN_LOG_STD = -20.0
_C.MODEL.POLICY.MAX_LOG_STD = 5.0
_C.MODEL.POLICY.ACTION_MIN = -10.0
_C.MODEL.POLICY.ACTION_MAX = 10.0
_C.MODEL.POLICY.MASK_ACTIONS = True

_C.MODEL.VALUE = CN()
_C.MODEL.VALUE.CLIP_ACTIONS = False
_C.MODEL.VALUE.VALUE_MIN = -10.0
_C.MODEL.VALUE.VALUE_MAX = 10.0
_C.MODEL.VALUE.MASK_VALUES = True

_C.MODEL.USE_CACHED_OBS = False
_C.MODEL.LIMB_EMBED_SIZE = 32
# Fixed std value
_C.MODEL.ACTION_STD = 0.9

# Use fixed or learnable std
_C.MODEL.ACTION_STD_FIXED = True

# --------------------------------------------------------------------------- #
# lr Options
# --------------------------------------------------------------------------- #
_C.LEARNING_RATE = CN()
_C.LEARNING_RATE.BASE_LR = 3e-4
_C.LEARNING_RATE.WARMUP_STEPS = 5
_C.LEARNING_RATE.WARMUP_FACTOR = 0.1


# --------------------------------------------------------------------------- #
# Transformer Options
# --------------------------------------------------------------------------- #
_C.MODEL.TYPE = "transformer"  # other option is "mlp"

_C.MODEL.TRANSFORMER = CN()

# Number of attention heads in TransformerEncoderLayer (nhead)
_C.MODEL.TRANSFORMER.NHEAD = 4

# TransformerEncoderLayer (dim_feedforward)
_C.MODEL.TRANSFORMER.DIM_FEEDFORWARD = 64

# TransformerEncoderLayer (dropout)
_C.MODEL.TRANSFORMER.DROPOUT = 0.0

# Number of TransformerEncoderLayer in TransformerEncoder
_C.MODEL.TRANSFORMER.NLAYERS = 3

# Init for input embedding
_C.MODEL.TRANSFORMER.EMBED_INIT = 0.1

# Init for output decoder embodedding
_C.MODEL.TRANSFORMER.DECODER_INIT = 0.01

_C.MODEL.TRANSFORMER.DECODER_DIM = 64
_C.MODEL.TRANSFORMER.N_DECODER_LAYERS = 2

_C.MODEL.TRANSFORMER.EXT_HIDDEN_DIMS = []

# Early vs late fusion of exterioceptive observation
_C.MODEL.TRANSFORMER.EXT_MIX = "none"

# Type of position embedding to use: None, learnt
_C.MODEL.TRANSFORMER.POS_EMBEDDING = "learnt"

_C.MODEL.TRANSFORMER.POS_EMBEDDING_LINK = "link"  # other option is 'robo_link'

_C.MODEL.RNN = CN()
_C.MODEL.RNN.FEATURE_DIM = 16
_C.MODEL.RNN.HIDDEN_SIZE = 32
_C.MODEL.RNN.NUM_LAYERS = 1
_C.MODEL.RNN.SEQUENCE_LENGTH = 32        # need to be a multiple of mini batch size


_C.MODEL.MLP = CN()
_C.MODEL.MLP.EMBED_DIM = 64
_C.MODEL.MLP.N_LAYERS = 3
_C.MODEL.MLP.DROPOUT = 0.1
_C.MODEL.MLP.OLD_VERSION = False
_C.MODEL.MLP_LATENT = "l"   # or "s" for small, "l" for large

_C.MODEL.OUTPUT_STD = False

# --------------------------------------------------------------------------- #
# CUDNN options
# --------------------------------------------------------------------------- #
_C.CUDNN = CN()

_C.CUDNN.BENCHMARK = False
_C.CUDNN.DETERMINISTIC = True


def dump_cfg(cfg_name=None):
    """Dumps the config to the output directory."""
    if not cfg_name:
        cfg_name = _C.CFG_DEST

    # current time
    exp_dir = _C.AGENT.EXPERIMENT.DIRECTORY
    # d = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    # the experiment directory is assumed to be a subdirectory of OUT_DIR
    # so no need to prepend OUT_DIR

    dirname = os.path.join(exp_dir, _C.AGENT.EXPERIMENT.EXPERIMENT_NAME)

    # dirname = os.path.join(_C.OUT_DIR, exp_dir)
    os.makedirs(dirname, exist_ok=True)

    cfg_file = os.path.join(dirname, cfg_name)
    with open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    _C.merge_from_file(cfg_file)


def get_default_cfg():
    return copy.deepcopy(cfg)


def get_lower_case_cfg(my_cfg):
    """
    Get a small case version of the cfg
    """
    small_case_cfg = {}
    for key, value in my_cfg.items():
        if isinstance(value, dict):
            small_case_cfg[key.lower()] = get_lower_case_cfg(value)
        else:
            if key.lower() in ['state_preprocessor', 'learning_rate_scheduler', 'value_preprocessor'] and value is None:
                small_case_cfg[key.lower()] = "None"
            else:
                small_case_cfg[key.lower()] = value
            
    return small_case_cfg


def update_values(cfg):
    """
    Update the values of the config based on the current values
    """        
    if cfg.OBSERVATION.LINK_POSE:
        cfg.OBSERVATION.ROBO_LINK_DIM += cfg.OBSERVATION.LINK_POSE_DIM
        

    if cfg.ACTION.ABSOLUTE:    
        cfg.MODEL.POLICY.ACTION_MIN = -1.0
        cfg.MODEL.POLICY.ACTION_MAX = 1.0

    else:
        cfg.MODEL.POLICY.ACTION_MIN = -2.0
        cfg.MODEL.POLICY.ACTION_MAX = 2.0


    # if only one task, disable pcgrad
    if len(cfg.MULTIENV.ROBOTS) == 1:
        cfg.AGENT.PCGRAD = False

    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

update_values(_C)



def get_ppo_cfg():
    PPO = CN()
    PPO.ROLLOUTS = 64
    PPO.LEARNING_EPOCHS = 4
    PPO.MINI_BATCHES = 4
    PPO.DISCOUNT_FACTOR = 0.99
    PPO.LAMBDA = 0.95
    PPO.LEARNING_RATE = 3 * 1e-4
    PPO.LEARNING_RATE_SCHEDULER = "LinearWarmstartCosineLR"
    PPO.LEARNING_RATE_SCHEDULER_KWARGS = ["start_factor", 1.0, "end_factor", 1e-7, "total_iters", 2e4]

    PPO.STATE_PREPROCESSOR = "None"
    PPO.STATE_PREPROCESSOR_KWARGS = None
    PPO.VALUE_PREPROCESSOR = "None"
    PPO.VALUE_PREPROCESSOR_KWARGS = None
    PPO.RANDOM_TIMESTEPS = 1000
    
    PPO.LEARNING_STARTS = 0
    PPO.GRAD_NORM_CLIP = 1.0
    PPO.RATIO_CLIP = 0.2
    PPO.VALUE_CLIP = 10.0
    PPO.CLIP_PREDICTED_VALUES = False
    PPO.ENTROPY_LOSS_SCALE = 0.01
    PPO.VALUE_LOSS_SCALE = 0.02
    PPO.KL_THRESHOLD = 0.0
    PPO.REWARDS_SHAPER = None
    PPO.REWARDS_SHAPER_SCALE = 1.0
    PPO.MIXED_PRECISION = False
    PPO.TIME_LIMIT_BOOTSTRAP = True

    return PPO


def get_sac_cfg():
    SAC = CN()
    SAC.GRADIENT_STEPS = 1
    SAC.BATCH_SIZE = 2048
    SAC.DISCOUNT_FACTOR = 0.99
    SAC.POLYAK = 0.005

    SAC.ACTOR_LEARNING_RATE = 5 * 1e-4
    SAC.CRITIC_LEARNING_RATE = 5 * 1e-4

    SAC.LEARNING_RATE_SCHEDULER = "LinearWarmstartCosineLR"
    SAC.LEARNING_RATE_SCHEDULER_KWARGS = ["start_factor", 1.0, "end_factor", 1e-2, "total_iters", 1e5]

    SAC.STATE_PREPROCESSOR = "None"
    SAC.STATE_PREPROCESSOR_KWARGS = CN()

    SAC.RANDOM_TIMESTEPS = 1000
    SAC.LEARNING_STARTS = 500

    SAC.GRAD_NORM_CLIP = 0.1

    SAC.LEARN_ENTROPY = True
    SAC.ENTROPY_LEARNING_RATE = 1e-3
    SAC.INITIAL_ENTROPY_VALUE = 1.0
    SAC.TARGET_ENTROPY = None

    SAC.REWARDS_SHAPER = None

    return SAC
    
    
_C.AGENT.PPO = get_ppo_cfg()
_C.AGENT.SAC = get_sac_cfg()
# _C.TRPO = get_trpo_cfg()
# _C.TD3 = get_td3_cfg()