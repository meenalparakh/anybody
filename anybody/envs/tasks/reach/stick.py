import numpy as np
from numpy.random import uniform as U

import anybody.envs.tasks.env_utils as eu
from anybody.utils.path_utils import get_robot_morphs_dir
import random
from copy import deepcopy

# size from l1 to l5
# pregenerated urdfs
# allowed_sizes = [f"{i}" for i in range(5)]
# allowed_lengths = np.linspace(0.2, 0.7, 5)

# allowed_sizes = ['l1', 'l2', 'l3', 'l4', 'l5']
# length goes from 0.2 to 0.7


def get_v4_env(robo_type, robo_fname, seed=0):
    np.random.seed(seed)
    random.seed(seed)

    # s = random.choice(allowed_sizes)
    obj_shape_z_dim = U(0.3, 0.5)
    obj_shape_x_dim = U(0.1, 0.6)

    robo_x_joint_i = U(-0.6, -obj_shape_x_dim / 2)
    robo_x_joint_g = U(obj_shape_x_dim / 2, 0.6)
    robo_y_joint_i, robo_y_joint_g = U(-np.pi, np.pi, 2)
    robo_z_joint_i, robo_z_joint_g = U(-np.pi / 6, np.pi / 6, 2) + np.pi / 2

    robot = eu.get_robot(robo_type="stick", robo_fname=robo_fname, robo_id=0)

    joint_names = robot.act_info["joint_names"]
    # initialize the obstacles
    p1 = np.eye(4)
    p1[:3, 3] = [0.0, 0.0, 0.3]
    cuboid1 = eu.Prim(
        obj_type="box",
        obj_id=1,
        obj_shape=[obj_shape_x_dim, 0.3, obj_shape_z_dim],
        pose=p1,
        static=True,
        friction=0.8,
    )

    robo_dict = {robot.robot_id: robot}
    obj_dict = {cuboid1.obj_id: cuboid1}

    # define the initial state
    init_robo_state_dict = {
        robot.robot_id: eu.RobotState(
            joint_pos=dict(
                zip(joint_names, [robo_x_joint_i, robo_y_joint_i, robo_z_joint_i])
            ),
        )
    }

    # init_obj_state_dict = {
    #     cuboid1.obj_id: PrimState(
    #         pose=p1,
    #     )
    # }

    # define the goal state
    goal_robo_state_dict = {
        robot.robot_id: eu.RobotState(
            joint_pos=dict(
                zip(joint_names, [robo_x_joint_g, robo_y_joint_g, robo_z_joint_g])
            ),
        ),
    }

    def get_cfg_fn():
        
        _robo_x_joint_i = U(-0.6, -obj_shape_x_dim / 2)
        _robo_x_joint_g = U(obj_shape_x_dim / 2, 0.6)
        _robo_y_joint_i, _robo_y_joint_g = U(-np.pi, np.pi, 2)
        _robo_z_joint_i, _robo_z_joint_g = U(-np.pi / 6, np.pi / 6, 2) + np.pi / 2

        _init_joints = [_robo_x_joint_i, _robo_y_joint_i, _robo_z_joint_i]
        _goal_joints = [_robo_x_joint_g, _robo_y_joint_g, _robo_z_joint_g]
        
        return {robot.robot_id: _init_joints}, {robot.robot_id: _goal_joints}, {}, {}
    
    cfg_name = eu.save_randomized_configs(
        robo_dict=robo_dict,
        get_cfg_fn=get_cfg_fn,
        task_type="reach",
        robo_type='stick',
        robo_name=robo_fname,
        variation="v4",
        seed=seed,
    )   


    return eu.ProblemSpec(
        robot_dict=robo_dict,
        obj_dict=obj_dict,
        init_robo_state_dict=init_robo_state_dict,
        goal_robo_state_dict=goal_robo_state_dict,
        init_obj_state_dict={},
        goal_obj_state_dict={},
        ground=-1.0,
        additional_configs=cfg_name,
    )


def get_v2_env(robo_type, robo_fname, seed=0):
    np.random.seed(seed)
    random.seed(seed)

    # s = random.choice(allowed_sizes)
    obj_shape_z_dim = U(0.3, 0.5)
    obj_shape_x_dim = U(0.1, 0.6)

    robo_x_joint_i = U(-0.6, -obj_shape_x_dim / 2)
    robo_x_joint_g = U(obj_shape_x_dim / 2, 0.6)
    robo_y_joint_i, robo_y_joint_g = U(-np.pi, np.pi, 2)
    robo_z_joint_i, robo_z_joint_g = U(-np.pi / 6, np.pi / 6, 2) + np.pi / 2

    # robot = get_robot(size=s)
    robot = eu.get_robot(robo_type="stick", robo_fname=robo_fname, robo_id=0)

    joint_names = robot.act_info["joint_names"]
    # initialize the obstacles
    p1 = np.eye(4)
    p1[:3, 3] = [0.0, 0.0, 0.3]
    cuboid1 = eu.Prim(
        obj_type="box",
        obj_id=1,
        obj_shape=[obj_shape_x_dim, 0.3, obj_shape_z_dim],
        pose=p1,
        static=True,
        friction=0.8,
    )

    p2 = np.eye(4)
    p2[:3, 3] = [0.0, 0.0, -0.3]
    cuboid2 = deepcopy(cuboid1)
    cuboid2.obj_id = 2
    cuboid2.pose = p2

    robo_dict = {robot.robot_id: robot}
    obj_dict = {cuboid1.obj_id: cuboid1, cuboid2.obj_id: cuboid2}

    # define the initial state
    init_robo_state_dict = {
        robot.robot_id: eu.RobotState(
            joint_pos=dict(
                zip(joint_names, [robo_x_joint_i, robo_y_joint_i, robo_z_joint_i])
            ),
        )
    }
    goal_robo_state_dict = {
        robot.robot_id: eu.RobotState(
            joint_pos=dict(
                zip(joint_names, [robo_x_joint_g, robo_y_joint_g, robo_z_joint_g])
            ),
        ),
    }


    def get_cfg_fn():
        
        _robo_x_joint_i = U(-0.6, -obj_shape_x_dim / 2)
        _robo_x_joint_g = U(obj_shape_x_dim / 2, 0.6)
        _robo_y_joint_i, _robo_y_joint_g = U(-np.pi, np.pi, 2)
        _robo_z_joint_i, _robo_z_joint_g = U(-np.pi / 6, np.pi / 6, 2) + np.pi / 2

        _init_joints = [_robo_x_joint_i, _robo_y_joint_i, _robo_z_joint_i]
        _goal_joints = [_robo_x_joint_g, _robo_y_joint_g, _robo_z_joint_g]
        
        return {robot.robot_id: _init_joints}, {robot.robot_id: _goal_joints}, {}, {}
    
    cfg_name = eu.save_randomized_configs(
        robo_dict=robo_dict,
        get_cfg_fn=get_cfg_fn,
        task_type="reach",
        robo_type='stick',
        robo_name=robo_fname,
        variation="v2",
        seed=seed,
    )   


    return eu.ProblemSpec(
        robot_dict=robo_dict,
        obj_dict=obj_dict,
        init_robo_state_dict=init_robo_state_dict,
        goal_robo_state_dict=goal_robo_state_dict,
        init_obj_state_dict={},
        goal_obj_state_dict={},
        ground=-1.0,
        additional_configs=cfg_name,
    )


def get_v3_env(robo_type, robo_fname, seed):
    np.random.seed(seed)
    random.seed(seed)

    # s = random.choice(allowed_sizes)
    robot = eu.get_robot(robo_type="stick", robo_fname=robo_fname, robo_id=0)

    # robot = get_robot(size=s)
    robo_y_joint = U(-np.pi / 6, np.pi / 6, 2) + np.pi / 2
    robot_init_joint_pos = [0.0, robo_y_joint[0], np.pi / 2]

    robo_goal_joint_pos = [0.2, robo_y_joint[1], np.pi / 2]

    p1 = np.eye(4)
    p1[:3, 3] = [-0.1, 0.0, 0.1]
    cuboid1 = eu.Prim(
        obj_type="box",
        obj_id=1,
        obj_shape=[0.15, 0.2, 0.1],
        pose=p1,
        static=True,
        friction=0.8,
    )

    p2 = np.eye(4)
    p2[:3, 3] = [0.1, 0.0, 0.1]
    cuboid2 = deepcopy(cuboid1)
    cuboid2.obj_id = 2
    cuboid2.pose = p2

    p3 = np.eye(4)
    p3[:3, 3] = [0.3, 0.0, 0.1]
    cuboid3 = deepcopy(cuboid1)
    cuboid3.obj_id = 3
    cuboid3.pose = p3

    p4 = np.eye(4)
    p4[:3, 3] = [U(-0.7, 0.7), 0.0, -0.1]
    cuboid4 = eu.Prim(
        obj_type="box",
        obj_id=4,
        obj_shape=[U(0.1, 0.4), U(0.1, 0.4), U(0.1, 0.13)],
        pose=p4,
        static=True,
    )

    p5 = np.eye(4)
    p5[:3, 3] = [U(-0.4, 0.4), U(0.3, 0.5), 0.0]
    cuboid5 = deepcopy(cuboid1)
    cuboid5.obj_id = 5
    cuboid5.pose = p5

    robo_dict = {robot.robot_id: robot}
    obj_dict = {
        cuboid1.obj_id: cuboid1,
        cuboid2.obj_id: cuboid2,
        cuboid3.obj_id: cuboid3,
        cuboid4.obj_id: cuboid4,
        cuboid5.obj_id: cuboid5,
    }

    joint_names = robot.act_info["joint_names"]

    # define the initial state
    init_robo_state_dict = {
        robot.robot_id: eu.RobotState(
            joint_pos=dict(zip(joint_names, robot_init_joint_pos)),
        )
    }
    goal_robo_state_dict = {
        robot.robot_id: eu.RobotState(
            joint_pos=dict(zip(joint_names, robo_goal_joint_pos)),
        ),
    }


    def get_cfg_fn():
        _robo_y_joint = U(-np.pi / 6, np.pi / 6, 2) + np.pi / 2
        _robot_init_joint_pos = [0.0, _robo_y_joint[0], np.pi / 2]

        _robo_goal_joint_pos = [0.2, _robo_y_joint[1], np.pi / 2]

        return {robot.robot_id: _robot_init_joint_pos}, {robot.robot_id: _robo_goal_joint_pos}, {}, {}
    
    cfg_name = eu.save_randomized_configs(
        robo_dict=robo_dict,
        get_cfg_fn=get_cfg_fn,
        task_type="reach",
        robo_type='stick',
        robo_name=robo_fname,
        variation="v3",
        seed=seed,
    )   

    return eu.ProblemSpec(
        robot_dict=robo_dict,
        obj_dict=obj_dict,
        init_robo_state_dict=init_robo_state_dict,
        goal_robo_state_dict=goal_robo_state_dict,
        init_obj_state_dict={},
        goal_obj_state_dict={},
        ground=-1.0,
        additional_configs=cfg_name,
    )


def get_v1_env(robo_type, robo_fname, seed):
    # robo_type is discarded, as it is always 'stick' for this function

    np.random.seed(seed)
    random.seed(seed)

    robot = eu.get_robot(robo_type="stick", robo_fname=robo_fname, robo_id=0)
    joint_names = robot.act_info["joint_names"]

    joint_x = U(-1.0, 1, 2)
    joint_y = U(-np.pi, np.pi, 2)
    joint_z = U(-np.pi, np.pi, 2)

    robo_dict = {robot.robot_id: robot}
    obj_dict = {}

    # define the initial state
    init_robo_state_dict = {
        robot.robot_id: eu.RobotState(
            joint_pos=dict(zip(joint_names, [joint_x[0], joint_y[0], joint_z[0]])),
        )
    }

    goal_robo_state_dict = {
        robot.robot_id: eu.RobotState(
            joint_pos=dict(zip(joint_names, [joint_x[1], joint_y[1], joint_z[1]])),
        ),
    }


    def get_cfg_fn():
        _joint_x = U(-1.0, 1, 2)
        _joint_y = U(-np.pi, np.pi, 2)
        _joint_z = U(-np.pi, np.pi, 2)
        _init_joints = [_joint_x[0], _joint_y[0], _joint_z[0]]
        _goal_joints = [_joint_x[1], _joint_y[1], _joint_z[1]]

        return {robot.robot_id: _init_joints}, {robot.robot_id: _goal_joints}, {}, {}
    
    cfg_name = eu.save_randomized_configs(
        robo_dict=robo_dict,
        get_cfg_fn=get_cfg_fn,
        task_type="reach",
        robo_type='stick',
        robo_name=robo_fname,
        variation="v1",
        seed=seed,
    )   

    return eu.ProblemSpec(
        robot_dict=robo_dict,
        obj_dict=obj_dict,
        init_robo_state_dict=init_robo_state_dict,
        goal_robo_state_dict=goal_robo_state_dict,
        init_obj_state_dict={},
        goal_obj_state_dict={},
        ground=-1.0,
        additional_configs=cfg_name,
    )


# get robo_fnames in stick directory


def get_v1_pose_env(robo_type, robo_fname, seed):
    return eu.get_pose_env(robo_type, robo_fname, seed, get_v1_env)

def get_v2_pose_env(robo_type, robo_fname, seed):
    return eu.get_pose_env(robo_type, robo_fname, seed, get_v2_env)

def get_v3_pose_env(robo_type, robo_fname, seed):
    return eu.get_pose_env(robo_type, robo_fname, seed, get_v3_env)

def get_v4_pose_env(robo_type, robo_fname, seed):
    return eu.get_pose_env(robo_type, robo_fname, seed, get_v4_env)



stick_joint_envs = {}
stick_pose_envs = {}

stick_dir = get_robot_morphs_dir() / "stick"
stick_fnames = [f.name for f in stick_dir.iterdir() if f.is_dir()]

for stick_fname in stick_fnames:
    potential_stick_urdf = stick_dir / stick_fname / f"{stick_fname}.urdf"
    if not potential_stick_urdf.exists():
        continue

    # matches the description in base.py and __init__.py for general robots

    stick_joint_envs.update(
        {
            f"{stick_fname}_v1": (get_v1_env, "stick", stick_fname),
            f"{stick_fname}_v2": (get_v2_env, "stick", stick_fname),
            f"{stick_fname}_v3": (get_v3_env, "stick", stick_fname),
            f"{stick_fname}_v4": (get_v4_env, "stick", stick_fname),
        }
    )

    stick_pose_envs.update(
        {
            f"{stick_fname}_v1": (get_v1_pose_env, "stick", stick_fname),
            f"{stick_fname}_v2": (get_v2_pose_env, "stick", stick_fname),
            f"{stick_fname}_v3": (get_v3_pose_env, "stick", stick_fname),
            f"{stick_fname}_v4": (get_v4_pose_env, "stick", stick_fname),
        }
    )


# stick_pose_envs = {
#     "v1": lambda seed: eu.get_pose_env("v1", stick_joint_envs, seed),
#     "v2": lambda seed: eu.get_pose_env("v2", stick_joint_envs, seed),
#     "v3": lambda seed: eu.get_pose_env("v3", stick_joint_envs, seed),
#     "v4": lambda seed: eu.get_pose_env("v4", stick_joint_envs, seed),
# }

# target type
envs = {"joint": stick_joint_envs, "pose": stick_pose_envs}
