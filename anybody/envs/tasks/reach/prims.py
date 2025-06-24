import numpy as np
from numpy.random import uniform as U

from urdfpy import URDF

import anybody.envs.tasks.env_utils as eu

# from anybody.utils.vis_server import VizServer
from anybody.utils.path_utils import get_robot_morphs_dir
import random
from scipy.spatial.transform import Rotation as R



def get_v1_env(robo_type, robo_name, seed=0):
    np.random.seed(seed)
    random.seed(seed)

    robot = eu.get_robot(robo_type=robo_type, robo_fname=robo_name, robo_id=0)

    robo_dict = {robot.robot_id: robot}

    while True:
        ground = U(-0.05, 0.05)

        n_obs = 1
        # n_obs = np.random.randint(1, 20)
        obj_dict = {}
        for idx in range(n_obs):
            p = np.eye(4)
            p[:3, 3] = [U(-0.5, 0.5), U(-0.5, 0.5), U(-0.1, 0.9)]

            cuboid1 = eu.Prim(
                obj_type="box",
                obj_id=idx,
                obj_shape=[U(0.02, 0.2), U(0.05, 0.2), U(0.02, 0.2)],
                pose=p,
                static=True,
            )
            obj_dict[cuboid1.obj_id] = cuboid1

        obj_dict["ground"] = eu.add_ground_prim(ground, obj_id=n_obs + 1)

        init_joints = eu.get_non_colliding_jpos_general(robot, obj_dict)
        goal_joints = eu.get_non_colliding_jpos_general(robot, obj_dict)
        if (init_joints is not None) and (goal_joints is not None):
            break

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
        _init_joints = eu.get_non_colliding_jpos_general(robot, obj_dict, return_dict=False)
        if _init_joints is None:
            return None
        _init_dict = {robot.robot_id: _init_joints}
        return _init_dict, _init_dict, {}, {}
    
    cfg_name = eu.save_randomized_configs(
        robo_dict=robo_dict,
        get_cfg_fn=get_cfg_fn,
        task_type="reach",
        robo_type='prims',
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

    robot = eu.get_robot(robo_type="prims", robo_fname=robo_name, robo_id=0)
    
    robo_dict = {robot.robot_id: robot}

    p = np.eye(4)
    p[:3, 3] = [0, 0.07, U(0.2, 0.6)]
    p[:3, :3] = R.from_euler(
        "xyz", [U(-np.pi / 6, np.pi / 6), U(-np.pi / 6, np.pi / 6), 0]
    ).as_matrix()
    cuboid = eu.Prim(
        obj_type="box",
        obj_id=0,
        obj_shape=[U(0.05, 0.4), U(0.05, 0.4), U(0.01, 0.08)],
        pose=p,
        static=True,
    )

    while True:
        obj_dict = {cuboid.obj_id: cuboid}

        if np.random.rand() > 0.3:
            p2 = np.eye(4)
            p2[:3, 3] = [0, 0.07, U(0.2, p[2, 3])]
            p2[:3, :3] = R.from_euler(
                "xyz", [U(-np.pi / 6, np.pi / 6), U(-np.pi / 6, np.pi / 6), 0]
            ).as_matrix()
            cuboid2 = eu.Prim(
                obj_type="box",
                obj_id=1,
                obj_shape=[U(0.05, 0.4), U(0.05, 0.4), U(0.01, 0.08)],
                pose=p2,
                static=True,
            )

            obj_dict[cuboid2.obj_id] = cuboid2

        if np.random.rand() > 0.3:
            p3 = np.eye(4)
            p3[:3, 3] = [0, 0.07, 0.05]
            p3[:3, :3] = R.from_euler(
                "xyz", [U(-np.pi / 6, np.pi / 6), U(-np.pi / 6, np.pi / 6), 0]
            ).as_matrix()
            cuboid3 = eu.Prim(
                obj_type="box",
                obj_id=2,
                obj_shape=[U(0.02, 0.2), U(0.05, 0.2), U(0.02, 0.05)],
                pose=p3,
                static=True,
            )
            obj_dict[cuboid3.obj_id] = cuboid3

        ground = U(-0.05, 0.05)
        obj_dict["ground"] = eu.add_ground_prim(ground, obj_id=5)

        lb = robot.act_info["joint_lb"]
        ub = robot.act_info["joint_ub"]
        init_joints = U(lb, ub)
        init_joints[:3] = [0.0, U(0.05, 0.09), U(p[2, 3], min(p[2, 3] + 0.2, 1.0))]
        init_dict = dict(zip(robot.act_info["joint_names"], init_joints))

        goal_joints = U(lb, ub)
        goal_joints[:3] = [0.0, U(0.05, 0.09), U(0.2, p[2, 3])]
        goal_dict = dict(zip(robot.act_info["joint_names"], goal_joints))

        robo_urdfpy = URDF.load(robot.robot_urdf)
        init_is_collision = eu.check_is_collision(robo_urdfpy, init_dict, obj_dict)
        goal_is_collision = eu.check_is_collision(robo_urdfpy, goal_dict, obj_dict)

        if not (init_is_collision or goal_is_collision):
            break

    init_robo_state_dict = {
        robot.robot_id: eu.RobotState(
            joint_pos=init_dict,
        )
    }
    goal_robo_state_dict = {
        robot.robot_id: eu.RobotState(
            joint_pos=goal_dict,
        ),
    }



    def get_cfg_fn():

        found = False
        
        for _ in range(20):
            _init_joints = U(lb, ub)
            _init_joints[:3] = [0.0, U(0.05, 0.09), U(p[2, 3], min(p[2, 3] + 0.2, 1.0))]
            _init_dict = dict(zip(robot.act_info["joint_names"], _init_joints))
            _init_is_collision = eu.check_is_collision(robo_urdfpy, _init_dict, obj_dict)
            if not _init_is_collision:
                found = True
                break
            
        if not found:
            return None
        
        _init_dict = {robot.robot_id: _init_joints}


        found = False
        
        for _ in range(20):
            _goal_joints = U(lb, ub)
            _goal_joints[:3] = [0.0, U(0.05, 0.09), U(0.2, p[2, 3])]
            _goal_dict = dict(zip(robot.act_info["joint_names"], _goal_joints))

            _goal_is_collision = eu.check_is_collision(robo_urdfpy, _goal_dict, obj_dict)
            if not _goal_is_collision:
                found = True
                break
            
        if not found:
            return None
        
        _goal_dict = {robot.robot_id: _goal_joints}
        
        return _init_dict, _goal_dict, {}, {}
    
    cfg_name = eu.save_randomized_configs(
        robo_dict=robo_dict,
        get_cfg_fn=get_cfg_fn,
        task_type="reach",
        robo_type='prims',
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


# def get_v3_env(seed=0):
#     np.random.seed(seed)
#     random.seed(seed)

#     rob_idx = np.random.randint(len(urdf_names))

#     robot = get_robot(idx=rob_idx)
#     robo_dict = {robot.robot_id: robot}

#     # same as v2 but with variations in horizontal positions
#     raise NotImplementedError


# def get_ee_pose(robot, jpos):

#     robot_urdfpy = URDF.load(robot.robot_urdf)
#     joint_names = robot.act_info["joint_names"]
#     jdict = dict(zip(joint_names, jpos))

#     link_fk = robot_urdfpy.link_fk(cfg=jdict)
#     for l in link_fk:
#         if l.name == robot.ee_link:
#             ee_pose = link_fk[l]
#             break
#     return ee_pose


def get_v1_pose_env(robo_type, robo_name, seed):
    return eu.get_pose_env(robo_type, robo_name, seed, get_v1_env)

def get_v2_pose_env(robo_type, robo_name, seed):
    return eu.get_pose_env(robo_type, robo_name, seed, get_v2_env)


prim_joint_envs = {}
prim_pose_envs = {}

prims_dir = get_robot_morphs_dir() / "prims"    
prims_fnames = [f.name for f in prims_dir.iterdir() if f.is_dir()]

for prims_fname in prims_fnames:
    potential_prims_urdf = prims_dir / prims_fname / f"{prims_fname}.urdf"
    if not potential_prims_urdf.exists():
        continue
    
    prim_joint_envs[f'{prims_fname}_v1'] = (get_v1_env, 'prims', prims_fname)
    prim_joint_envs[f'{prims_fname}_v2'] = (get_v2_env, 'prims', prims_fname)

    prim_pose_envs[f'{prims_fname}_v1'] = (get_v1_pose_env, 'prims', prims_fname)
    prim_pose_envs[f'{prims_fname}_v2'] = (get_v2_pose_env, 'prims', prims_fname)
    

# bot_joint_envs = {
#     "v1": get_v1_env,
#     "v2": get_v2_env,
# }


# bot_pose_envs = {
#     "v1": lambda seed: eu.get_pose_env("v1", bot_joint_envs, seed),
#     "v2": lambda seed: eu.get_pose_env("v2", bot_joint_envs, seed),
# }

envs = {
    "joint": prim_joint_envs,
    "pose": prim_pose_envs,
}
