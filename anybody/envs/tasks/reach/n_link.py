import numpy as np
from numpy.random import uniform as U

from urdfpy import URDF

import anybody.envs.tasks.env_utils as eu
from anybody.utils.path_utils import get_robot_morphs_dir
from anybody.cfg import cfg
from anybody.utils import utils

# from anybody.utils.vis_server import VizServer
import random
from copy import deepcopy

import trimesh

allowed_sizes = [2, 3, 4, 5]
allowed_lengths = np.linspace(0.5, 0.2, 4)


def get_non_colliding_pos(robot, obstacles={}, max_tries=100):
    joint_names = robot.act_info["joint_names"]

    for _ in range(max_tries):
        # pos = [U(-np.pi + np.pi / 6, np.pi - np.pi / 6) for _ in range(len(joint_names))]
        pos = [U(-np.pi + np.pi / 2, np.pi - np.pi / 2) for _ in range(len(joint_names))]
        # cum angles for each link
        is_collision = eu.check_collision(robot, dict(zip(joint_names, pos)), obstacles)
        if not is_collision:
            return pos
    return None


def get_v1_env(robo_type, robo_name, seed):
    np.random.seed(seed)
    random.seed(seed)

    robot = eu.get_robot(robo_type=robo_type, robo_fname=robo_name, robo_id=0)
    joint_names = robot.act_info["joint_names"]

    robo_dict = {robot.robot_id: robot}


    def get_cfg_fn():
        
        for _ in range(20):
            n_joints = len(joint_names)
            jpos = U(-np.pi / 2, np.pi / 2, n_joints)
            is_collision = eu.check_collision(robot, dict(zip(joint_names, jpos)), {})
            if not is_collision:
                r_init_dict = {robot.robot_id: jpos}
                # returning robot init, robo goal, obj init, obj goal
                return r_init_dict, r_init_dict, {}, {}

        return None

    cfg_name = eu.save_randomized_configs(
        robo_dict=robo_dict,
        get_cfg_fn=get_cfg_fn,
        task_type="reach",
        robo_type='nlink',
        robo_name=robo_name,
        variation="v1",
        seed=seed,
    )

    init_joints = get_non_colliding_pos(robot, obstacles={})
    assert init_joints is not None
    init_robo_state_dict = {
        robot.robot_id: eu.RobotState(
            joint_pos=dict(zip(joint_names, init_joints)),
        )
    }

    goal_joints = get_non_colliding_pos(robot, obstacles={})
    assert goal_joints is not None
    goal_robo_state_dict = {
        robot.robot_id: eu.RobotState(
            joint_pos=dict(zip(joint_names, goal_joints)),
        ),
    }

    return eu.ProblemSpec(
        robot_dict=robo_dict,
        obj_dict={},
        init_robo_state_dict=init_robo_state_dict,
        goal_robo_state_dict=goal_robo_state_dict,
        init_obj_state_dict={},
        goal_obj_state_dict={},
        additional_configs=cfg_name,
    )


def get_v2_env(robo_type, robo_name, seed=0, n_cfgs=500):
    np.random.seed(seed)
    random.seed(seed)

    robot = eu.get_robot(robo_type="nlink", robo_fname=robo_name, robo_id=0)
    joint_names = robot.act_info["joint_names"]

    robo_dict = {robot.robot_id: robot}

    while True:
        n_obstacles = random.choice([0, 1, 2, 3])

        obj_dict = {}
        for idx in range(n_obstacles):
            p1 = np.eye(4)
            p1[:3, 3] = [U(-0.6, -0.1), U(-0.6, 0.6), 0.0]
            cuboid1 = eu.Prim(
                obj_type="box",
                obj_id=idx + 2,
                obj_shape=[U(0.1, 0.4), U(0.1, 0.4), U(0.1, 0.13)],
                pose=p1,
                static=True,
            )
            obj_dict[cuboid1.obj_id] = cuboid1

        init_joints = get_non_colliding_pos(robot, obstacles=obj_dict)
        goal_joints = get_non_colliding_pos(robot, obstacles=obj_dict)
        # goal_joints = eu.get_non_colliding_jpos_general(
        #     robot, obj_dict=obj_dict, return_dict=False
        # )

        # init_joints = get_non_colliding_pos(robot, s, obstacles=obj_dict)
        # goal_joints = get_non_colliding_pos(robot, s, obstacles=obj_dict)
        if (init_joints is not None) and (goal_joints is not None):
            break

    # define the initial state
    init_robo_state_dict = {
        robot.robot_id: eu.RobotState(
            joint_pos=dict(zip(joint_names, init_joints)),
        )
    }

    # define the goal state
    goal_robo_state_dict = {
        robot.robot_id: eu.RobotState(
            joint_pos=dict(zip(joint_names, goal_joints)),
        ),
    }

    def get_cfg_fn():
        
        for _ in range(20):
            n_joints = len(joint_names)
            jpos = U(-np.pi + np.pi / 2, np.pi - np.pi / 2, n_joints)
            is_collision = eu.check_collision(robot, dict(zip(joint_names, jpos)), {})
            if not is_collision:
                r_init_dict = {robot.robot_id: jpos}
                # returning robot init, robo goal, obj init, obj goal
                return r_init_dict, r_init_dict, {}, {}

        return None
        
    cfg_name = eu.save_randomized_configs(
        robo_dict=robo_dict,
        get_cfg_fn=get_cfg_fn,
        task_type="reach",
        robo_type='nlink',
        robo_name=robo_name,
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
        additional_configs=cfg_name,
    )


def get_v3_env(robo_type, robo_name, seed=0):
    np.random.seed(seed)
    random.seed(seed)

    # s_idx = np.random.choice(len(allowed_sizes))

    s_idx = allowed_sizes.index(int(robo_name.split("_")[0]))
    _l = allowed_lengths[s_idx]

    robot = eu.get_robot(robo_type="nlink", robo_fname=robo_name, robo_id=0)
    joint_names = robot.act_info["joint_names"]

    robo_dict = {robot.robot_id: robot}

    p1 = np.eye(4)
    p1[:3, 3] = [0, -U(_l + 0.1, 1.5 * _l), 0.0]
    cuboid1 = eu.Prim(
        obj_type="box",
        obj_id=1,
        obj_shape=[U(0.1, 0.4), 0.1, U(0.1, 0.13)],
        pose=p1,
        static=True,
    )

    p2 = np.eye(4)
    p2[:3, 3] = [0.0, U(_l + 0.1, 1.5 * _l), 0.0]
    cuboid2 = deepcopy(cuboid1)
    cuboid2.obj_id = 2
    cuboid2.pose = p2

    obj_dict = {cuboid1.obj_id: cuboid1, cuboid2.obj_id: cuboid2}

    init_joints = np.pi * 5 / 6 * np.ones(allowed_sizes[s_idx])
    init_joints[::2] = -np.pi * 5 / 6
    init_joints[0] = -U(0.0, np.pi / 2)

    goal_joints = np.pi * 5 / 6 * np.ones(allowed_sizes[s_idx])
    goal_joints[1::2] = -np.pi * 5 / 6
    goal_joints[0] = U(0.0, np.pi / 2) + np.pi

    # define the initial state
    init_robo_state_dict = {
        robot.robot_id: eu.RobotState(
            joint_pos=dict(zip(joint_names, init_joints)),
        )
    }

    # define the goal state
    goal_robo_state_dict = {
        robot.robot_id: eu.RobotState(
            joint_pos=dict(zip(joint_names, goal_joints)),
        ),
    }


    def get_cfg_fn():
        _init_joints = np.pi * 5 / 6 * np.ones(allowed_sizes[s_idx])
        _init_joints[::2] = -np.pi * 5 / 6
        _init_joints[0] = -U(0.0, np.pi / 2)

        _goal_joints = np.pi * 5 / 6 * np.ones(allowed_sizes[s_idx])
        _goal_joints[1::2] = -np.pi * 5 / 6
        _goal_joints[0] = U(0.0, np.pi / 2) + np.pi

        _init_dict = {robot.robot_id: _init_joints}
        _goal_dict = {robot.robot_id: _goal_joints}
        return _init_dict, _goal_dict, {}, {}
        
    cfg_name = eu.save_randomized_configs(
        robo_dict=robo_dict,
        get_cfg_fn=get_cfg_fn,
        task_type="reach",
        robo_type='nlink',
        robo_name=robo_name,
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
        additional_configs=cfg_name,
    )


def get_v4_env(robo_type, robo_name, seed=0):
    np.random.seed(seed)
    random.seed(seed)

    robot = eu.get_robot(robo_type="nlink", robo_fname=robo_name, robo_id=0)
    s_idx = allowed_sizes.index(int(robo_name.split("_")[0]))
    _l = allowed_lengths[s_idx]

    joint_names = robot.act_info["joint_names"]
    robo_dict = {robot.robot_id: robot}

    p1 = np.eye(4)
    p1[:3, 3] = [0, -U(_l + 0.1, 1.5 * _l), 0.0]
    cuboid1 = eu.Prim(
        obj_type="box",
        obj_id=1,
        obj_shape=[U(0.1, 0.4), 0.1, U(0.1, 0.13)],
        pose=p1,
        static=True,
    )

    obj_dict = {cuboid1.obj_id: cuboid1}

    init_joints = np.zeros(allowed_sizes[s_idx])

    while True:
        goal_joints = -U(np.pi / 6, np.pi / 3) * np.ones(allowed_sizes[s_idx])
        # check if this is in collision
        is_collision = eu.check_collision(
            robot, dict(zip(joint_names, goal_joints)), obj_dict
        )
        if not is_collision:
            break

    # define the initial state
    init_robo_state_dict = {
        robot.robot_id: eu.RobotState(
            joint_pos=dict(zip(joint_names, init_joints)),
        )
    }

    # define the goal state
    goal_robo_state_dict = {
        robot.robot_id: eu.RobotState(
            joint_pos=dict(zip(joint_names, goal_joints)),
        ),
    }

    def get_cfg_fn():
        
        found = False
        for _ in range(20):
            _goal_joints = -U(np.pi / 6, np.pi / 3) * np.ones(allowed_sizes[s_idx])
            # check if this is in collision
            is_collision = eu.check_collision(
                robot, dict(zip(joint_names, _goal_joints)), obj_dict
            )
            if not is_collision:
                found = True
                break

        if not found:
            return None
        
        _goal_dict = {robot.robot_id: _goal_joints}
        return {}, _goal_dict, {}, {}
        
    cfg_name = eu.save_randomized_configs(
        robo_dict=robo_dict,
        get_cfg_fn=get_cfg_fn,
        task_type="reach",
        robo_type='nlink',
        robo_name=robo_name,
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
        additional_configs=cfg_name,
    )



def get_v1_pose_env(robo_type, robo_fname, seed):
    return eu.get_pose_env(robo_type, robo_fname, seed, get_v1_env)

def get_v2_pose_env(robo_type, robo_fname, seed):
    return eu.get_pose_env(robo_type, robo_fname, seed, get_v2_env)

def get_v3_pose_env(robo_type, robo_fname, seed):
    return eu.get_pose_env(robo_type, robo_fname, seed, get_v3_env)

def get_v4_pose_env(robo_type, robo_fname, seed):
    return eu.get_pose_env(robo_type, robo_fname, seed, get_v4_env)



nlink_joint_envs = {}

nlink_pose_envs = {}

nlink_dir = get_robot_morphs_dir() / "nlink"
nlink_fnames = [f.name for f in nlink_dir.iterdir() if f.is_dir()]

for nlink_fname in nlink_fnames:
    potential_nlink_urdf = nlink_dir / nlink_fname / f"{nlink_fname}.urdf"
    if not potential_nlink_urdf.exists():
        continue

    # matches the description in base.py and __init__.py for general robots

    nlink_joint_envs.update(
        {
            f"{nlink_fname}_v1": (get_v1_env, "nlink", nlink_fname),
            f"{nlink_fname}_v2": (get_v2_env, "nlink", nlink_fname),
            f"{nlink_fname}_v3": (get_v3_env, "nlink", nlink_fname),
            f"{nlink_fname}_v4": (get_v4_env, "nlink", nlink_fname),
        }
    )
    nlink_pose_envs.update(
        {
            f"{nlink_fname}_v1": (get_v1_pose_env, "nlink", nlink_fname),
            f"{nlink_fname}_v2": (get_v2_pose_env, "nlink", nlink_fname),
            f"{nlink_fname}_v3": (get_v3_pose_env, "nlink", nlink_fname),
            f"{nlink_fname}_v4": (get_v4_pose_env, "nlink", nlink_fname),
        }
    )
    


# nlink_pose_envs = {
#     "v1": lambda seed: eu.get_pose_env("v1", nlink_joint_envs, seed),
#     "v2": lambda seed: eu.get_pose_env("v2", nlink_joint_envs, seed),
#     "v3": lambda seed: eu.get_pose_env("v3", nlink_joint_envs, seed),
#     "v4": lambda seed: eu.get_pose_env("v4", nlink_joint_envs, seed),
# }

# target type
envs = {"joint": nlink_joint_envs, "pose": nlink_pose_envs}
