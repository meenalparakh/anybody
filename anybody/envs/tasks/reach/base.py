# general script for creating any reach task for a given robot
import numpy as np
from numpy.random import uniform as U
import trimesh

import anybody.envs.tasks.env_utils as eu

import random
import typing as T


def get_v1_env(robo_type, robo_name, seed=0, save_dirname=None, save_fname=None):
    np.random.seed(seed)
    random.seed(seed)

    robot: eu.Robot = eu.get_robot(robo_type, robo_name, robo_id=0)

    robo_dict = {robot.robot_id: robot}

    obj_dict = {}

    ground = U(-0.2, -0.03)
    obj_dict["ground"] = eu.add_ground_prim(ground, obj_id=2)

    # joint_names = robot.act_info["joint_names"]
    # joint_lb = robot.act_info["joint_lb"]
    # joint_ub = robot.act_info["joint_ub"]

    # define the initial state
    # init_joints = U(joint_lb, joint_ub)
    
    
    
    init_joints = eu.get_non_colliding_jpos_general(robot, obj_dict, robo_type=robo_type)
    goal_joints = eu.get_non_colliding_jpos_general(robot, obj_dict, robo_type=robo_type)
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
        _init_joints = eu.get_non_colliding_jpos_general(robot, obj_dict, return_dict=False, robo_type=robo_type)
        if (_init_joints is None):
            return None

        _init_dict = {robot.robot_id: _init_joints}
        return _init_dict, _init_dict, {}, {}, {}
    
    cfg_name = eu.save_randomized_configs(
        robo_dict=robo_dict,
        get_cfg_fn=get_cfg_fn,
        task_type="reach",
        robo_type=robo_type,
        robo_name=robo_name,
        variation="v1",
        seed=seed,
        save_dirname=save_dirname,
        save_fname=save_fname,
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


def get_v2_env(robo_type, robo_name, seed=0, save_dirname=None, save_fname=None):
    np.random.seed(seed)
    random.seed(seed)

    robot: eu.Robot = eu.get_robot(robo_type, robo_name, robo_id=0)
    joint_lb = robot.act_info["joint_lb"]
    joint_ub = robot.act_info["joint_ub"]
    
    robo_dict = {robot.robot_id: robot}

    obst_bounds = np.array([[-0.1, 0.15, 0.05], [0.1, 0.6, 0.2]])

    p1 = np.eye(4)
    p1[:3, 3] = U(obst_bounds[0], obst_bounds[1])
    cuboid1 = eu.Prim(
        obj_type="box",
        obj_id=1,
        obj_shape=[
            U(0.06, 0.15),
            U(0.06, 0.15),
            U(0.06, 0.15),
            # U(0.02, 2 * (p1[1, 3] - 0.15)),
            # (p1[2, 3] + U(-0.05, 0.05)) * 2,
        ],
        pose=p1,
        static=True,
    )
    obj_dict: T.Dict[T.Any, eu.Prim] = {cuboid1.obj_id: cuboid1}

    # pdict = {
    #     "joint_base_rev": [0, np.pi / 3],
    #     "joint_1_rev": [-np.pi / 2, 0.0],
    #     "joint_2_rev": [0, np.pi / 2],
    # }

    ground = U(-0.2, -0.03)
    obj_dict["ground"] = eu.add_ground_prim(ground, obj_id=2)

    init_joints = eu.get_non_colliding_jpos_general(robot, obj_dict, robo_type=robo_type) #, partial_dict=pdict)
    # pdict["joint_base_rev"] = [np.pi - np.pi / 3, np.pi]
    goal_joints = eu.get_non_colliding_jpos_general(robot, obj_dict, robo_type=robo_type) #, partial_dict=pdict)

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

    
    obj_dict.pop("ground")

    def obj_sampler(obstacle_bounds, flag="init"):
        _p = trimesh.transformations.random_rotation_matrix()
        _p[:3, 3] = U(obstacle_bounds[0], obstacle_bounds[1])

        return _p
        
    def robo_jpos_sampler(joint_lb, joint_ub):
        return U(joint_lb, joint_ub)

    def get_cfg_fn():
        return eu.get_randomized_reach_cfg(
            obj_dict,
            movable_obj_dict={},
            robot_dict=robo_dict,
            obj_sampler=obj_sampler,
            robot_sampler=robo_jpos_sampler,
            z_ground=ground,
            obj_sampler_args=[obst_bounds],
            robo_sampler_args=[joint_lb, joint_ub],
        )

    
    
    cfg_name = eu.save_randomized_configs(
        robo_dict=robo_dict,
        get_cfg_fn=get_cfg_fn,
        task_type="reach",
        robo_type=robo_type,
        robo_name=robo_name,
        variation="v2",
        seed=seed,
        save_dirname=save_dirname,
        save_fname=save_fname,
    )   

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

def get_v3_env(robo_type, robo_name, seed=0, save_dirname=None, save_fname=None):
    np.random.seed(seed)
    random.seed(seed)

    robot = eu.get_robot(robo_type, robo_name, robo_id=0)

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
    # bounds = np.array([[0.1, 0.3, 0.01], [0.2, 0.5, 0.1]])

    # while True:
    #     init_joints = [np.pi/2, -U(np.pi/6, np.pi/2), 2 * np.pi/3, -np.pi/3, 0.5, -0.5, -0.5, 0.5]
    #     init_joints[2] = -init_joints[1] * 2
    # pdict = {
    #     "joint_base_rev": [np.pi / 2 - np.pi / 6, np.pi / 2 + np.pi / 6],
    #     "joint_1_rev": [-np.pi / 2, -np.pi / 6],
    #     "joint_2_rev": [np.pi / 2, 5 * np.pi / 6],
    #     "joint_3_rev": [-np.pi / 2, -np.pi / 6],
    # }

    ground = U(-0.2, -0.03)
    obj_dict["ground"] = eu.add_ground_prim(ground, obj_id=2)

    init_joints = eu.get_non_colliding_jpos_general(robot, obj_dict, robo_type=robo_type) #, partial_dict=pdict)
    init_robo_state_dict = {
        robot.robot_id: eu.RobotState(
            joint_pos=init_joints,
        )
    }

    # pdict = {
    #     "joint_base_rev": [np.pi / 2 - np.pi / 6, np.pi / 2 + np.pi / 6],
    #     "joint_1_rev": [-np.pi / 2, -np.pi / 6],
    #     "joint_2_rev": [0, np.pi / 2],
    # }
    goal_joints = eu.get_non_colliding_jpos_general(robot, obj_dict, robo_type=robo_type) #, partial_dict=pdict)
    goal_robo_state_dict = {
        robot.robot_id: eu.RobotState(
            joint_pos=goal_joints,
        ),
    }



    def get_cfg_fn():
        _init_joints = eu.get_non_colliding_jpos_general(robot, obj_dict, return_dict=False, robo_type=robo_type)
        if (_init_joints is None):
            return None

        _init_dict = {robot.robot_id: _init_joints}
        return _init_dict, _init_dict, {}, {}
    
    cfg_name = eu.save_randomized_configs(
        robo_dict=robo_dict,
        get_cfg_fn=get_cfg_fn,
        task_type="reach",
        robo_type=robo_type,
        robo_name=robo_name,
        variation="v3",
        seed=seed,
        save_dirname=save_dirname,
        save_fname=save_fname,
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


def get_v4_env(robo_type, robo_name, seed=0, save_dirname=None, save_fname=None):
    np.random.seed(seed)
    random.seed(seed)

    robot = eu.get_robot(robo_type, robo_name, robo_id=0)

    robo_dict = {robot.robot_id: robot}

    # joint_names = robot.act_info["joint_names"]

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

        # pdict = {
        #     "joint_base_rev": [0, np.pi / 3],
        #     "joint_1_rev": [-np.pi / 2, 0.0],
        #     "joint_2_rev": [0, np.pi / 2],
        # }

        ground = U(-0.2, -0.03)
        obj_dict["ground"] = eu.add_ground_prim(ground, obj_id=n_obs + 2)

        init_joints = eu.get_non_colliding_jpos_general(robot, obj_dict, robo_type=robo_type) #, partial_dict=pdict)
        # pdict["joint_base_rev"] = [np.pi - np.pi / 3, np.pi]
        goal_joints = eu.get_non_colliding_jpos_general(robot, obj_dict, robo_type=robo_type) #, partial_dict=pdict)

        if (init_joints is None) or (goal_joints is None):
            # goal_joints = eu.get_non_colliding_jpos_general(robot, obj_dict, view=True, max_tries=1)
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
        _init_joints = eu.get_non_colliding_jpos_general(robot, obj_dict, return_dict=False, robo_type=robo_type)
        if (_init_joints is None):
            return None

        _init_dict = {robot.robot_id: _init_joints}
        return _init_dict, _init_dict, {}, {}
    
    cfg_name = eu.save_randomized_configs(
        robo_dict=robo_dict,
        get_cfg_fn=get_cfg_fn,
        task_type="reach",
        robo_type=robo_type,
        robo_name=robo_name,
        variation="v4",
        seed=seed,
        save_dirname=save_dirname,  
        save_fname=save_fname,
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



def get_v1_pose_env(robo_type, robo_name, seed, save_dirname=None, save_fname=None):
    return eu.get_pose_env(get_v1_env, robo_type, robo_name, seed, save_dirname, save_fname)

def get_v2_pose_env(robo_type, robo_name, seed, save_dirname=None, save_fname=None):
    return eu.get_pose_env(get_v2_env, robo_type, robo_name, seed, save_dirname, save_fname)

def get_v3_pose_env(robo_type, robo_name, seed, save_dirname=None, save_fname=None):
    return eu.get_pose_env(get_v3_env, robo_type, robo_name, seed, save_dirname, save_fname)

def get_v4_pose_env(robo_type, robo_name, seed, save_dirname=None, save_fname=None):
    return eu.get_pose_env(get_v4_env, robo_type, robo_name, seed, save_dirname, save_fname)


bot_joint_envs = {
    "v1": get_v1_env,
    "v2": get_v2_env,
    "v3": get_v3_env,
    "v4": get_v4_env,
}

bot_pose_envs = {   
    "v1": get_v1_pose_env,
    "v2": get_v2_pose_env,
    "v3": get_v3_pose_env,
    "v4": get_v4_pose_env,
}



# bot_pose_envs = {
#     "v1": lambda seed: eu.get_pose_env("v1", bot_joint_envs, seed),
#     "v2": lambda seed: eu.get_pose_env("v2", bot_joint_envs, seed),
#     "v3": lambda seed: eu.get_pose_env("v3", bot_joint_envs, seed),
#     "v4": lambda seed: eu.get_pose_env("v4", bot_joint_envs, seed),
# }

envs = {
    "joint": bot_joint_envs,
    "pose": bot_pose_envs,
}
