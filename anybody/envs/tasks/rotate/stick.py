import numpy as np
from numpy.random import uniform as U
import trimesh
from urdfpy import URDF
from scipy.spatial.transform import Rotation as R

import anybody.envs.tasks.env_utils as eu
from anybody.utils.path_utils import get_robot_morphs_dir

# import anybody.articulation_utils.load as load_articulations


door_fnames = eu.get_door_fnames()
drawer_fnames = eu.get_drawer_fnames()
turner_fnames = eu.get_turner_fnames()


import random
from copy import deepcopy


allowed_sizes = [f"{i}" for i in range(5)]
allowed_lengths = np.linspace(0.2, 0.7, len(allowed_sizes))

# allowed_sizes = ['l1', 'l2', 'l3', 'l4', 'l5']
# length goes from 0.2 to 0.7


def get_robot(size="0", control_type="joint"):
    # initialize the stick model

    stick_urdf = (
        get_robot_morphs_dir() / "stick" / f"stick_{size}" / f"stick_{size}.urdf"
    )

    # urdfpy_robot = URDF.load(stick_urdf)
    joint_names = ["joint_px", "joint_rx", "joint_rz"]
    urdfpy_robot = URDF.load(stick_urdf)
    jlimits = {j.name: j.limit for j in urdfpy_robot.joints}
    joint_lb = [jlimits[jname].lower for jname in joint_names]
    joint_ub = [jlimits[jname].upper for jname in joint_names]

    robot = eu.Robot(
        control_type=control_type,
        act_info={
            "joint_names": joint_names,
            "joint_lb": joint_lb,
            "joint_ub": joint_ub,
        },
        robot_id=0,
        robot_urdf=str(stick_urdf),
        ee_link="link_ball",
        pose=np.eye(4),
        joint_pos=dict(zip(joint_names, np.zeros(3))),
        joint_vel=dict(zip(joint_names, np.zeros(3))),
    )
    return robot


def get_drawer_envs(seed=0, rot_variation=False):
    import anybody.articulation_utils.load as load_articulations

    np.random.seed(seed)
    random.seed(seed)

    s_idx = np.random.choice(len(allowed_sizes))
    length = allowed_lengths[s_idx]
    s = allowed_sizes[s_idx]
    robot = get_robot(size=s)
    robo_dict = {robot.robot_id: robot}

    joint_names = robot.act_info["joint_names"]

    dz = U(-length, -length / 3)
    y_max = max(length**2 - dz**2, 0.0)
    y_max = np.sqrt(y_max)

    if rot_variation:
        y_max = 0.0

    d_pos = [U(-0.1, 0.1), U(-y_max - 0.001, y_max + 0.001), dz]
    drawer_idx = np.random.choice(len(drawer_fnames))

    if rot_variation:
        rotz = U(-np.pi, np.pi)
    else:
        rotz = 0.0

    drawer, ht = load_articulations.get_drawer(drawer_idx, pos=d_pos, rotz=rotz)

    d_joint_names = drawer.act_info["joint_names"]
    drawer_joint_lb = drawer.act_info["joint_lb"]
    drawer_joint_ub = drawer.act_info["joint_ub"]

    articulation_dict = {
        drawer.articulation_id: drawer,
    }

    init_d_jpos = dict(zip(d_joint_names, U(drawer_joint_lb, drawer_joint_ub)))
    goal_d_jpos = dict(zip(d_joint_names, U(drawer_joint_lb, drawer_joint_ub)))

    init_articulation_dict = {
        drawer.articulation_id: eu.ArticulationState(joint_pos=init_d_jpos)
    }
    goal_articulation_dict = {
        drawer.articulation_id: eu.ArticulationState(joint_pos=goal_d_jpos)
    }

    robo_init_jpos = [d_pos[0] + U(0.3 + 0.5), U(-np.pi, np.pi), U(-np.pi, np.pi)]
    init_robo_state_dict = {
        robot.robot_id: eu.RobotState(joint_pos=dict(zip(joint_names, robo_init_jpos)))
    }

    print("height of drawer", ht)
    return eu.ProblemSpec(
        robot_dict=robo_dict,
        obj_dict={},
        articulation_dict=articulation_dict,
        init_robo_state_dict=init_robo_state_dict,
        goal_robo_state_dict={r.robot_id: eu.RobotState() for r in robo_dict.values()},
        init_articulation_state_dict=init_articulation_dict,
        goal_articulation_state_dict=goal_articulation_dict,
        init_obj_state_dict={},
        goal_obj_state_dict={},
        ground=min(-length - 0.01, d_pos[2] - ht),
    )


# the stick is along x axis
# the door should be some distance away from the stick
# the possible locations for door could be
# r radius from the stick
# facing the origin


def get_door_envs(seed=0, rotx_variation=False):
    import anybody.articulation_utils.load as load_articulations

    np.random.seed(seed)
    random.seed(seed)

    s_idx = np.random.choice(len(allowed_sizes))
    length = allowed_lengths[s_idx]
    s = allowed_sizes[s_idx]
    robot = get_robot(size=s)
    robo_dict = {robot.robot_id: robot}

    joint_names = robot.act_info["joint_names"]

    theta = U(0, 2 * np.pi)
    r = U(length / 2, length)
    pos = [r * np.cos(theta), r * np.sin(theta), 0.0]
    rotz = theta + np.pi
    if rotx_variation:
        rotx = U(-np.pi / 2, np.pi / 2)
    else:
        rotx = 0.0

    door_idx = np.random.choice(len(door_fnames))
    door, ht = load_articulations.get_door(door_idx, pos=pos, rotz=rotz, rotx=rotx)
    pos_z = U(-ht / 2, ht / 2)
    door.pose[2, 3] = pos_z

    d_joint_names = door.act_info["joint_names"]
    d_joint_lb = door.act_info["joint_lb"]
    d_joint_ub = door.act_info["joint_ub"]

    articulation_dict = {
        door.articulation_id: door,
    }

    init_d_jpos = dict(zip(d_joint_names, U(d_joint_lb, d_joint_ub)))
    goal_d_jpos = dict(zip(d_joint_names, U(d_joint_lb, d_joint_ub)))

    init_articulation_dict = {
        door.articulation_id: eu.ArticulationState(joint_pos=init_d_jpos)
    }
    goal_articulation_dict = {
        door.articulation_id: eu.ArticulationState(joint_pos=goal_d_jpos)
    }

    robo_init_jpos = eu.get_non_colliding_jpos_general(
        robot, articulation_dict, art_jdict={door.articulation_id: init_d_jpos}
    )

    init_robo_state_dict = {robot.robot_id: eu.RobotState(joint_pos=robo_init_jpos)}

    return eu.ProblemSpec(
        robot_dict=robo_dict,
        obj_dict={},
        articulation_dict=articulation_dict,
        init_robo_state_dict=init_robo_state_dict,
        goal_robo_state_dict={r.robot_id: eu.RobotState() for r in robo_dict.values()},
        init_articulation_state_dict=init_articulation_dict,
        goal_articulation_state_dict=goal_articulation_dict,
        init_obj_state_dict={},
        goal_obj_state_dict={},
        ground=min(pos_z - ht - 0.1 * (1 if rotx_variation else 0), -length - 0.01),
    )


def get_turner_envs(seed=0, rot_variation=False):
    import anybody.articulation_utils.load as load_articulations

    np.random.seed(seed)
    random.seed(seed)

    s_idx = np.random.choice(len(allowed_sizes))
    length = allowed_lengths[s_idx]
    s = allowed_sizes[s_idx]
    robot = get_robot(size=s)
    robo_dict = {robot.robot_id: robot}

    joint_names = robot.act_info["joint_names"]

    theta = U(0, 2 * np.pi)
    r = U(length / 3, length * 0.8)

    pz = length**2 - r**2
    pz = np.sqrt(max(pz, 0.0))

    pos = [r * np.cos(theta), r * np.sin(theta), 0.0]
    rotz = theta + np.pi

    turner_idx = np.random.choice(len(turner_fnames))
    turner, _ = load_articulations.get_turner(turner_idx, pos=pos, rotz=rotz)
    pos_z = U(-pz - 0.001, pz + 0.001)
    turner.pose[2, 3] = pos_z

    d_joint_names = turner.act_info["joint_names"]
    d_joint_lb = turner.act_info["joint_lb"]
    d_joint_ub = turner.act_info["joint_ub"]

    articulation_dict = {
        turner.articulation_id: turner,
    }

    init_d_jpos = dict(zip(d_joint_names, U(d_joint_lb, d_joint_ub)))
    goal_d_jpos = dict(zip(d_joint_names, U(d_joint_lb, d_joint_ub)))

    init_articulation_dict = {
        turner.articulation_id: eu.ArticulationState(joint_pos=init_d_jpos)
    }
    goal_articulation_dict = {
        turner.articulation_id: eu.ArticulationState(joint_pos=goal_d_jpos)
    }

    robo_init_jpos = eu.get_non_colliding_jpos_general(
        robot, articulation_dict, art_jdict={turner.articulation_id: init_d_jpos}
    )

    init_robo_state_dict = {robot.robot_id: eu.RobotState(joint_pos=robo_init_jpos)}

    return eu.ProblemSpec(
        robot_dict=robo_dict,
        obj_dict={},
        articulation_dict=articulation_dict,
        init_robo_state_dict=init_robo_state_dict,
        goal_robo_state_dict={r.robot_id: eu.RobotState() for r in robo_dict.values()},
        init_articulation_state_dict=init_articulation_dict,
        goal_articulation_state_dict=goal_articulation_dict,
        init_obj_state_dict={},
        goal_obj_state_dict={},
        ground=min(-length - 0.01, pos_z - 0.11),
    )


def get_v1_env(seed=0):
    return get_drawer_envs(seed=seed, rot_variation=False)


def get_v2_env(seed=0):
    return get_drawer_envs(seed=seed, rot_variation=True)


def get_v3_env(seed=0):
    return get_door_envs(seed=seed)


def get_v4_env(seed=0):
    return get_door_envs(seed=seed, rotx_variation=True)


def get_v5_env(seed=0):
    return get_turner_envs(seed=seed)


# def get_v5_env(seed=0):
#     return get_turner_envs(seed=seed, rot_variation=False)

# def get_v6_env(seed=0):
#     return get_turner_envs(seed=seed, rot_variation=True)


# rotate for the stick robot should be
# rotate wheel
# rotate window [ | ] to [ ||| ] (the projection of the window)
# open drawer
# open dishwasher kind mechanism
# closing cupboard door

# for this I need to make procedural articulations and
# find how to represent them


stick_joint_envs = {
    "v1": get_v1_env,
    "v2": get_v2_env,
    "v3": get_v3_env,
    "v4": get_v4_env,
    "v5": get_v5_env,
}

envs = {
    "joint": stick_joint_envs,
}
