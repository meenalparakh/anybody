import numpy as np
from numpy.random import uniform as U

from urdfpy import URDF

import anybody.envs.tasks.env_utils as eu

from anybody.cfg import cfg
# from anybody.utils.vis_server import VizServer
from anybody.utils.path_utils import get_robot_morphs_dir
from anybody.utils import utils
import random
from scipy.optimize import minimize
import typing as T

def get_v1_env(robo_type, robo_name, seed=0):
    np.random.seed(seed)
    random.seed(seed)

    robot = eu.get_robot(robo_type=robo_type, robo_fname=robo_name, robo_id=0)  

    robo_dict = {robot.robot_id: robot}

    obj_dict = {}

    ground = U(-0.2, -0.03)
    obj_dict["ground"] = eu.add_ground_prim(ground, obj_id=2)

    # joint_names = robot.act_info["joint_names"]
    # joint_lb = robot.act_info["joint_lb"]
    # joint_ub = robot.act_info["joint_ub"]

    # define the initial state
    # init_joints = U(joint_lb, joint_ub)
    init_joints = eu.get_non_colliding_jpos(robot, obj_dict)
    goal_joints = eu.get_non_colliding_jpos(robot, obj_dict)
    if (init_joints is None) or (goal_joints is None):
        raise ValueError("No non-colliding init joint found")

    init_robo_state_dict = {
        robot.robot_id: eu.RobotState(
            joint_pos=init_joints,
        )
    }
    goal_robo_state_dict = {
        robot.robot_id: eu.RobotState(
            joint_pos=goal_joints,
        ),
    }


    def get_cfg_fn():
        _init_joints = eu.get_non_colliding_jpos(robot, obj_dict, return_dict=False)
        if _init_joints is None:
            return None
        _init_dict = {robot.robot_id: _init_joints}

        return _init_dict, _init_dict, {}, {}
    
    cfg_name = eu.save_randomized_configs(
        robo_dict=robo_dict,
        get_cfg_fn=get_cfg_fn,
        task_type="reach",
        robo_type='simple_bot',
        robo_name=robo_name,
        variation="v1",
        seed=seed,
    )   

    obj_dict.pop("ground")

    return eu.ProblemSpec(
        robot_dict=robo_dict,
        obj_dict=obj_dict,
        init_robo_state_dict=init_robo_state_dict,
        goal_robo_state_dict=goal_robo_state_dict,
        init_obj_state_dict={},
        goal_obj_state_dict={},
        ground=ground,
        additional_configs=cfg_name,
    )


def get_v2_env(robo_type, robo_name, seed=0):
    np.random.seed(seed)
    random.seed(seed)

    robot = eu.get_robot(robo_type=robo_type, robo_fname=robo_name, robo_id=0)

    robo_dict = {robot.robot_id: robot}

    p1 = np.eye(4)
    p1[:3, 3] = [U(-0.1, 0.1), U(0.15, 0.6), U(0.05, 0.2)]
    cuboid1 = eu.Prim(
        obj_type="box",
        obj_id=1,
        obj_shape=[
            U(0.02, 0.05),
            U(0.02, 2 * (p1[1, 3] - 0.15)),
            (p1[2, 3] + U(-0.05, 0.05)) * 2,
        ],
        pose=p1,
        static=True,
    )
    obj_dict: T.Dict[T.Any, eu.Prim] = {cuboid1.obj_id: cuboid1}

    pdict = {
        "joint_base_rev": [0, np.pi / 3],
        "joint_1_rev": [-np.pi / 2, 0.0],
        "joint_2_rev": [0, np.pi / 2],
    }

    ground = U(-0.2, -0.03)
    obj_dict["ground"] = eu.add_ground_prim(ground, obj_id=2)

    init_joints = eu.get_non_colliding_jpos(robot, obj_dict, partial_dict=pdict)
    pdict["joint_base_rev"] = [np.pi - np.pi / 3, np.pi]
    goal_joints = eu.get_non_colliding_jpos(robot, obj_dict, partial_dict=pdict)

    if (init_joints is None) or (goal_joints is None):
        raise ValueError("No non-colliding init joint found")

    init_robo_state_dict = {
        robot.robot_id: eu.RobotState(
            joint_pos=init_joints,
        )
    }
    goal_robo_state_dict = {
        robot.robot_id: eu.RobotState(
            joint_pos=goal_joints,
        ),
    }

    
    
    def get_cfg_fn():

        pdict = {
            "joint_base_rev": [0, np.pi / 3],
            "joint_1_rev": [-np.pi / 2, 0.0],
            "joint_2_rev": [0, np.pi / 2],
        }
        _init_joints = eu.get_non_colliding_jpos(robot, obj_dict, partial_dict=pdict, return_dict=False)
        pdict["joint_base_rev"] = [np.pi - np.pi / 3, np.pi]
        _goal_joints = eu.get_non_colliding_jpos(robot, obj_dict, partial_dict=pdict, return_dict=False)

        if (_init_joints is None) or (_goal_joints is None):
            return None

        _init_dict = {robot.robot_id: _init_joints}
        _goal_dict = {robot.robot_id: _goal_joints} 

        return _init_dict, _goal_dict, {}, {}
    
    cfg_name = eu.save_randomized_configs(
        robo_dict=robo_dict,
        get_cfg_fn=get_cfg_fn,
        task_type="reach",
        robo_type='simple_bot',
        robo_name=robo_name,
        variation="v2",
        seed=seed,
    )   

    obj_dict.pop("ground")
    return eu.ProblemSpec(
        robot_dict=robo_dict,
        obj_dict=obj_dict,
        init_robo_state_dict=init_robo_state_dict,
        goal_robo_state_dict=goal_robo_state_dict,
        init_obj_state_dict={},
        goal_obj_state_dict={},
        ground=ground,
        additional_configs=cfg_name,
    )


def optimize_jpos(robot, obj_dict, bb: np.ndarray, init_jpos=None):
    robot_urdfpy = URDF.load(robot.robot_urdf)
    joint_names = robot.act_info["joint_names"]
    joint_lb = robot.act_info["joint_lb"]
    joint_ub = robot.act_info["joint_ub"]
    ee_link = robot.ee_link

    def loss_fn(jpos):
        jdict = dict(zip(joint_names, jpos))
        # get ee position
        link_fk = robot_urdfpy.link_fk(cfg=jdict)
        for l in link_fk:
            if l.name == ee_link:
                ee_pose = link_fk[l]
                break
        ee_pos = ee_pose[:3, 3]

        d1 = np.abs(bb[0] - ee_pos)
        d2 = np.abs(bb[1] - ee_pos)
        d3 = np.abs(bb[1] - bb[0])
        d_from_center = (2 * ee_pos - bb[0] - bb[1]) / 2

        _l = d1 + d2 - d3
        l1 = 0.1 * np.linalg.norm(d_from_center) + np.linalg.norm(_l)
        loss = 10 * l1

        # get collision distance
        # collision_fk = robot_urdfpy.collision_trimesh_fk(cfg=jdict)
        # min_distance = []

        # collision_manager = trimesh.collision.CollisionManager()
        # for obj in obj_dict.values():
        #     collision_manager.add_object(
        #         f"{obj.obj_id}", obj.mesh, transform=obj.pose
        #     )

        # for mesh, pose in collision_fk.items():
        #     dist = collision_manager.min_distance_single(
        #         mesh, transform=pose
        #     )
        #     dist = dist - 0.01
        #     if dist > 0.0:
        #         min_distance.append(0.0)
        #     else:
        #         min_distance.append(dist)

        # l2 = -np.sum(min_distance)
        # loss += l2

        return loss

    if init_jpos is not None:
        x0 = init_jpos
    else:
        x0 = U(joint_lb, joint_ub)
    b = [(joint_lb[i], joint_ub[i]) for i in range(len(joint_lb))]

    result = minimize(loss_fn, x0, method="L-BFGS-B", bounds=b, options={"disp": False})
    return result.x, result.fun


def get_ee_pose(robot, jpos):
    robot_urdfpy = URDF.load(robot.robot_urdf)
    joint_names = robot.act_info["joint_names"]
    jdict = dict(zip(joint_names, jpos))

    link_fk = robot_urdfpy.link_fk(cfg=jdict)
    for l in link_fk:
        if l.name == robot.ee_link:
            ee_pose = link_fk[l]
            break
    return ee_pose


def get_v3_env(robo_type, robo_name, seed=0):
    # shelf kind of variation

    np.random.seed(seed)
    random.seed(seed)

    robot = eu.get_robot(robo_type=robo_type, robo_fname=robo_name, robo_id=0)

    robo_dict = {robot.robot_id: robot}

    p1 = np.eye(4)
    p1[:3, 3] = [0.0, 0.4, U(0.1, 0.2)]
    cuboid1 = eu.Prim(
        obj_type="box",
        obj_id=1,
        obj_shape=[U(0.1, 0.6), U(0.2, 0.3), U(0.01, 0.04)],
        pose=p1,
        static=True,
    )

    obj_dict: T.Dict[T.Any, eu.Prim] = {cuboid1.obj_id: cuboid1}

    # dummy box
    bounds = np.array([[0.1, 0.3, 0.01], [0.2, 0.5, 0.1]])

    # while True:
    #     init_joints = [np.pi/2, -U(np.pi/6, np.pi/2), 2 * np.pi/3, -np.pi/3, 0.5, -0.5, -0.5, 0.5]
    #     init_joints[2] = -init_joints[1] * 2
    pdict = {
        "joint_base_rev": [np.pi / 2 - np.pi / 6, np.pi / 2 + np.pi / 6],
        "joint_1_rev": [-np.pi / 2, -np.pi / 6],
        "joint_2_rev": [np.pi / 2, 5 * np.pi / 6],
        "joint_3_rev": [-np.pi / 2, -np.pi / 6],
    }

    ground = U(-0.2, -0.03)
    obj_dict["ground"] = eu.add_ground_prim(ground, obj_id=2)

    init_joints = eu.get_non_colliding_jpos(robot, obj_dict, partial_dict=pdict)
    init_robo_state_dict = {
        robot.robot_id: eu.RobotState(
            joint_pos=init_joints,
        )
    }

    pdict = {
        "joint_base_rev": [np.pi / 2 - np.pi / 6, np.pi / 2 + np.pi / 6],
        "joint_1_rev": [-np.pi / 2, -np.pi / 6],
        "joint_2_rev": [0, np.pi / 2],
    }
    goal_joints = eu.get_non_colliding_jpos(robot, obj_dict, partial_dict=pdict)
    goal_robo_state_dict = {
        robot.robot_id: eu.RobotState(
            joint_pos=goal_joints,
        ),
    }



    def get_cfg_fn():
        pdict = {
                "joint_base_rev": [np.pi / 2 - np.pi / 6, np.pi / 2 + np.pi / 6],
                "joint_1_rev": [-np.pi / 2, -np.pi / 6],
                "joint_2_rev": [np.pi / 2, 5 * np.pi / 6],
                "joint_3_rev": [-np.pi / 2, -np.pi / 6],
            }
        _init_joints = eu.get_non_colliding_jpos(robot, obj_dict, partial_dict=pdict, return_dict=False)

        pdict = {
            "joint_base_rev": [np.pi / 2 - np.pi / 6, np.pi / 2 + np.pi / 6],
            "joint_1_rev": [-np.pi / 2, -np.pi / 6],
            "joint_2_rev": [0, np.pi / 2],
        }        
        _goal_joints = eu.get_non_colliding_jpos(robot, obj_dict, partial_dict=pdict, return_dict=False)

        if (_init_joints is None) or (_goal_joints is None):
            return None

        _init_dict = {robot.robot_id: _init_joints}
        _goal_dict = {robot.robot_id: _goal_joints} 

        return _init_dict, _goal_dict, {}, {}
    
    cfg_name = eu.save_randomized_configs(
        robo_dict=robo_dict,
        get_cfg_fn=get_cfg_fn,
        task_type="reach",
        robo_type='simple_bot',
        robo_name=robo_name,
        variation="v3",
        seed=seed,
    )   

    obj_dict.pop("ground")
    return eu.ProblemSpec(
        robot_dict=robo_dict,
        obj_dict=obj_dict,
        init_robo_state_dict=init_robo_state_dict,
        goal_robo_state_dict=goal_robo_state_dict,
        init_obj_state_dict={},
        goal_obj_state_dict={},
        ground=ground,
        additional_configs=cfg_name,
    )


def get_v4_env(robo_type, robo_name, seed=0):
    np.random.seed(seed)
    random.seed(seed)

    robot = eu.get_robot(robo_type=robo_type, robo_fname=robo_name, robo_id=0)

    robo_dict = {robot.robot_id: robot}

    joint_names = robot.act_info["joint_names"]

    # while True:
    global_found = False
    for _ in range(100):
        n_obs = np.random.randint(1, 10)
        obj_dict = {}
        for idx in range(n_obs):
            p = np.eye(4)
            p[:3, 3] = [U(-0.5, 0.5), U(0.2, 0.5), U(-0.1, 0.5)]

            cuboid1 = eu.Prim(
                obj_type="box",
                obj_id=idx,
                obj_shape=[U(0.05, 0.07), U(0.05, 0.07), U(0.02, 0.05)],
                pose=p,
                static=True,
            )
            obj_dict[cuboid1.obj_id] = cuboid1

        pdict = {
            "joint_base_rev": [0, np.pi / 3],
            "joint_1_rev": [-np.pi / 2, 0.0],
            "joint_2_rev": [0, np.pi / 2],
        }

        ground = U(-0.2, -0.03)
        obj_dict["ground"] = eu.add_ground_prim(ground, obj_id=n_obs + 2)

        init_joints = eu.get_non_colliding_jpos(robot, obj_dict, partial_dict=pdict)
        pdict["joint_base_rev"] = [np.pi - np.pi / 3, np.pi]
        goal_joints = eu.get_non_colliding_jpos(robot, obj_dict, partial_dict=pdict)

        if (init_joints is None) or (goal_joints is None):
            # goal_joints = eu.get_non_colliding_jpos(robot, obj_dict, view=True, max_tries=1)
            # raise ValueError("No non-colliding init joint found")
            continue
        else:
            global_found = True
            break

    if not global_found:
        raise ValueError("No non-colliding init joint found")

    init_robo_state_dict = {
        robot.robot_id: eu.RobotState(
            joint_pos=init_joints,
        )
    }
    goal_robo_state_dict = {
        robot.robot_id: eu.RobotState(
            joint_pos=goal_joints,
        ),
    }
    
    

    def get_cfg_fn():
        pdict = {
            "joint_base_rev": [0, np.pi / 3],
            "joint_1_rev": [-np.pi / 2, 0.0],
            "joint_2_rev": [0, np.pi / 2],
        }  
        _init_joints = eu.get_non_colliding_jpos(robot, obj_dict, partial_dict=pdict, return_dict=False)

        pdict["joint_base_rev"] = [np.pi - np.pi / 3, np.pi]

        _goal_joints = eu.get_non_colliding_jpos(robot, obj_dict, partial_dict=pdict, return_dict=False)

        if (_init_joints is None) or (_goal_joints is None):
            return None

        _init_dict = {robot.robot_id: _init_joints}
        _goal_dict = {robot.robot_id: _goal_joints} 

        return _init_dict, _goal_dict, {}, {}
    
    cfg_name = eu.save_randomized_configs(
        robo_dict=robo_dict,
        get_cfg_fn=get_cfg_fn,
        task_type="reach",
        robo_type='simple_bot',
        robo_name=robo_name,
        variation="v4",
        seed=seed,
    )   


    obj_dict.pop("ground")
    return eu.ProblemSpec(
        robot_dict=robo_dict,
        obj_dict=obj_dict,
        init_robo_state_dict=init_robo_state_dict,
        goal_robo_state_dict=goal_robo_state_dict,
        init_obj_state_dict={},
        goal_obj_state_dict={},
        ground=ground,
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



simple_bot_joint_envs = {}
simple_bot_pose_envs = {}


simple_bot_dir = get_robot_morphs_dir() / "simple_bot"
simple_bot_fnames = [f.name for f in simple_bot_dir.iterdir() if f.is_dir()]

for simple_bot_fname in simple_bot_fnames:
    potential_simple_bot_urdf = simple_bot_dir / simple_bot_fname / f"{simple_bot_fname}.urdf"
    if not potential_simple_bot_urdf.exists():
        continue

    # matches the description in base.py and __init__.py for general robots

    simple_bot_joint_envs.update(
        {
            f"{simple_bot_fname}_v1": (get_v1_env, "simple_bot", simple_bot_fname),
            f"{simple_bot_fname}_v2": (get_v2_env, "simple_bot", simple_bot_fname),
            f"{simple_bot_fname}_v3": (get_v3_env, "simple_bot", simple_bot_fname),
            f"{simple_bot_fname}_v4": (get_v4_env, "simple_bot", simple_bot_fname),
        }
    )
    simple_bot_pose_envs.update(
        {
            f"{simple_bot_fname}_v1": (get_v1_pose_env, "simple_bot", simple_bot_fname),
            f"{simple_bot_fname}_v2": (get_v2_pose_env, "simple_bot", simple_bot_fname),
            f"{simple_bot_fname}_v3": (get_v3_pose_env, "simple_bot", simple_bot_fname),
            f"{simple_bot_fname}_v4": (get_v4_pose_env, "simple_bot", simple_bot_fname),
        }
    )
    



# bot_joint_envs = {
#     "v1": get_v1_env,
#     "v2": get_v2_env,
#     "v3": get_v3_env,
#     "v4": get_v4_env,
# }


# bot_pose_envs = {
#     "v1": lambda seed: eu.get_pose_env("v1", bot_joint_envs, seed),
#     "v2": lambda seed: eu.get_pose_env("v2", bot_joint_envs, seed),
#     "v3": lambda seed: eu.get_pose_env("v3", bot_joint_envs, seed),
#     "v4": lambda seed: eu.get_pose_env("v4", bot_joint_envs, seed),
# }

envs = {
    "joint": simple_bot_joint_envs,
    "pose": simple_bot_pose_envs,
}
