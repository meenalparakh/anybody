import numpy as np
from numpy.random import uniform as U

from urdfpy import URDF

import anybody.envs.tasks.env_utils as eu
import anybody.utils.utils as u
from anybody.utils.path_utils import get_robot_morphs_dir
import random
from copy import deepcopy
from anybody.utils.utils import transform_pcd, save_pickle

import trimesh
from scipy.spatial.transform import Rotation as R

# allowed_sizes = [2, 3, 4, 5]
# allowed_lengths = np.linspace(0.5, 0.2, 4)

# copied from cube_urdf.py
# n_cubes = [1, 2, 3, 4, 5]
# fnames = [f"{n}_{i}" for i in range(2) for n in n_cubes]

fnames = [f"cube_{i}" for i in range(5)]


def get_robot_height(robot):
    robo_urdfpy = URDF.load(robot.robot_urdf)
    link_fk = robo_urdfpy.collision_trimesh_fk(cfg=robot.joint_pos)
    min_z = []
    for m, p in link_fk.items():
        vs = m.vertices
        vs = transform_pcd(vs, p)
        min_z.append(vs[:, 2].min())

    min_z = np.array(min_z).min()
    ht = -min_z
    return ht


def get_all_obstacle_pos(robo_dict, obj_dict, z_ground, obstacle_dict=None):
    max_tries = 100
    pose_dict = {}

    # initialize the collision manager with obj dict
    collision_manager = trimesh.collision.CollisionManager()
    if obstacle_dict is not None:
        collision_manager = eu.add_to_collision_manager(
            collision_manager, obstacle_dict
        )

    # add robo to collision manager
    for robot in robo_dict.values():
        collision_manager = eu.add_robo_to_collision_manager(
            collision_manager, robot, robot.joint_pos
        )

    obj_pos_bounds = np.array([[-0.3, -0.3], [0.3, 0.3]])

    # find obstacle poses
    for obj_id, obj in obj_dict.items():
        found = False
        obj_ht = obj.obj_shape[2]
        for _ in range(max_tries):
            p = np.eye(4)
            p[:2, 3] = U(obj_pos_bounds[0], obj_pos_bounds[1])
            p[:3, :3] = R.from_euler("xyz", [0, 0, U(-np.pi, np.pi)]).as_matrix()
            p[2, 3] = z_ground + obj_ht / 2

            dist = collision_manager.min_distance_single(obj.mesh, transform=p)
            is_collision = dist < 0.03
            if not is_collision:
                collision_manager.add_object(f"{obj_id}", obj.mesh, transform=p)
                pose_dict[obj_id] = p
                found = True
                break
        if not found:
            return None

    return pose_dict


def get_v1_env(robo_type, robo_fname, seed=0, version_name="v1"):
    np.random.seed(seed)
    random.seed(seed)

    robo_dict = {}

    robot = eu.get_robot(robo_type='cubes', robo_fname=robo_fname, robo_id=0)
    ht = get_robot_height(robot)
    
    robot.pose[2, 3] = ht

    robo_dict[robot.robot_id] = robot
    joint_names = robot.act_info["joint_names"]
    lb = robot.act_info["joint_lb"]
    ub = robot.act_info["joint_ub"]
    init_joints = U(lb, ub)
    robot.joint_pos = dict(zip(joint_names, init_joints))

    obst_pos_bounds = np.array([[-0.3, -0.3], [0.3, 0.3]])

    while True:
        # add static obstacles
        static_obs_dict = {}
        n_obstacles = np.random.choice(5)

        for idx in range(n_obstacles):
            p = np.eye(4)
            p[:2, 3] = U(obst_pos_bounds[0], obst_pos_bounds[1])
            obj_shape = U(0.1, 0.2, 3).tolist()
            p[2, 3] = obj_shape[2] / 2
            p[:3, :3] = R.from_euler("xyz", [0, 0, U(-np.pi, np.pi)]).as_matrix()
            cuboid = eu.Prim(
                obj_type="box",
                obj_id=idx,
                obj_shape=obj_shape,
                pose=p,
                static=True,
            )
            static_obs_dict[cuboid.obj_id] = cuboid

        collision_manager = trimesh.collision.CollisionManager()
        collision_manager = eu.add_to_collision_manager(
            collision_manager, static_obs_dict
        )

        # now adding movable obstacles

        z_ground = 0.0
        obj_dict = {}
        n_obs = np.random.choice(4)
        for idx in range(n_obs):
            obj_shape = U(0.1, 0.2, 3).tolist()
            cuboid = eu.Prim(
                obj_type="box",
                obj_id=idx + n_obstacles,
                obj_shape=obj_shape,
                pose=np.eye(4),
                static=False,
            )
            obj_dict[cuboid.obj_id] = cuboid

        # find non-colliding poses for obstacles
        init_pose_dict = get_all_obstacle_pos(
            robo_dict, obj_dict, z_ground=z_ground, obstacle_dict=static_obs_dict
        )
        goal_pose_dict = get_all_obstacle_pos(
            robo_dict, obj_dict, z_ground=z_ground, obstacle_dict=static_obs_dict
        )

        if not ((init_pose_dict is None) or (goal_pose_dict is None)):
            break

    init_obj_state_dict = {k: eu.PrimState(pose=v) for k, v in init_pose_dict.items()}

    goal_obj_state_dict = {k: eu.PrimState(pose=v) for k, v in goal_pose_dict.items()}

    # define the initial state for robot
    init_robo_state_dict = {
        robot.robot_id: eu.RobotState(joint_pos=robot.joint_pos),
    }

    obj_pos_bounds = np.array([[-0.3, -0.3], [0.3, 0.3]])

    def obj_sampler(_obj_pos_bounds):
        p = np.eye(4)
        p[:2, 3] = U(_obj_pos_bounds[0], _obj_pos_bounds[1])
        p[:3, :3] = R.from_euler("xyz", [0, 0, U(-np.pi, np.pi)]).as_matrix()
    
        return p
        # return get_all_obstacle_pos(
        #     robo_dict, obj_dict, z_ground=z_ground, obstacle_dict=static_obs_dict
        # )
        
    def robo_jpos_sampler(lb, ub):
        return U(lb, ub)

    def get_cfg_fn():
        return eu.get_randomized_push_cfg(
            static_obs_dict,
            movable_obj_dict=obj_dict,
            robot_dict=robo_dict,
            obj_sampler=obj_sampler,
            robot_sampler=robo_jpos_sampler,
            obj_sampler_args=[obj_pos_bounds],
            robo_sampler_args=[lb, ub],
            z_ground=z_ground,
        )

    cfg_name = eu.save_randomized_configs(
        robo_dict=robo_dict,
        get_cfg_fn=get_cfg_fn,
        task_type="push",
        robo_type="cubes",
        robo_name=robo_fname,
        variation=version_name,
        seed=seed,
    )


    obj_dict.update(static_obs_dict)

    return eu.ProblemSpec(
        robot_dict=robo_dict,
        obj_dict=obj_dict,
        init_robo_state_dict=init_robo_state_dict,
        goal_robo_state_dict={robot.robot_id: eu.RobotState()},
        init_obj_state_dict=init_obj_state_dict,
        goal_obj_state_dict=goal_obj_state_dict,
        ground=z_ground,
        additional_configs=cfg_name,
    )


def get_v1_pose_env(robo_type, robo_fname, seed=0):
    return eu.get_pose_env(robo_type, robo_fname, seed, env_fn=get_v1_env)

cube_joint_envs = {}
cube_pose_envs = {}

cube_dir = get_robot_morphs_dir() / "cubes"
cube_fnames = [f.name for f in cube_dir.iterdir() if f.is_dir()]

for cube_fname in cube_fnames:
    potential_cube_urdf = cube_dir / cube_fname / f"{cube_fname}.urdf"
    if not potential_cube_urdf.exists():
        continue

    cube_joint_envs[f'{cube_fname}_v1'] = (get_v1_env, 'cubes', cube_fname)
    cube_pose_envs[f'{cube_fname}_v1'] = (get_v1_pose_env, 'cubes', cube_fname)

# cube_pose_envs = {
#     "v1": lambda seed: eu.get_pose_env("v1", cube_joint_envs, seed),
# }

cube_pose_envs = {}

# target type
envs = {
    "joint": cube_joint_envs,
    "pose": cube_pose_envs,
}
