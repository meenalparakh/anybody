import numpy as np
from numpy.random import uniform as U

from urdfpy import URDF

import anybody.envs.tasks.env_utils as eu

# from anybody.utils.vis_server import VizServer
from anybody.utils.path_utils import get_robot_morphs_dir
import random
from copy import deepcopy
from scipy.spatial.transform import Rotation as R

import trimesh


# urdf_names = [f"planar_{i}" for i in range(3)]


def get_non_colliding_jpos(
    robot, obj_dict, partial_dict=None, view=False, max_tries=100
):
    # joint_pos = {}

    joint_names = robot.act_info["joint_names"]
    joint_lb = robot.act_info["joint_lb"]
    joint_ub = robot.act_info["joint_ub"]

    robo_urdfpy = URDF.load(robot.robot_urdf)

    for _ in range(max_tries):
        # get robo mesh
        jpos = U(joint_lb, joint_ub)
        # jpos[-2:] = -jpos[-4:-2]     # symmetric end effector joints
        jdict = dict(zip(joint_names, jpos))

        if partial_dict is not None:
            for k in partial_dict:
                jdict[k] = U(partial_dict[k][0], partial_dict[k][1])

        is_collision = eu.check_is_collision(robo_urdfpy, jdict, obj_dict, view=view)
        if not is_collision:
            return jdict

    return None


# push env for prims - same as the reach env with the bot


# the push envs for simple robot - would involve cube pushing on a planar surface that may be tilted
def get_base_env(robo_type, robo_fname, n_obs, n_blocks, seed=0, version_name='v1'):
    np.random.seed(seed)
    random.seed(seed)

    robot = eu.get_robot(robo_type="prims", robo_fname=robo_fname, robo_id=0)

    robo_dict = {robot.robot_id: robot}

    joint_names = robot.act_info["joint_names"]
    joint_lb = robot.act_info["joint_lb"]
    joint_ub = robot.act_info["joint_ub"]

    obst_shape_bounds = np.array([[0.04, 0.04, 0.04], [0.2, 0.2, 0.2]])

    block_shape_bounds = np.array([[0.02, 0.02, 0.04], [0.07, 0.07, 0.07]])

    max_tries = 100
    env_found = False

    for _ in range(max_tries):
        # the robot is at 0, 0, 0
        z_ground = U(0.0, 0.3)

        obst_pos_bounds = np.array([[-0.5, -0.5, z_ground], [0.5, 0.5, 0.5]])

        block_pos_bounds = np.array([[-0.5, -0.5, 0.0], [0.5, 0.5, 1.0]])

        obstacle_dict = {}

        # add obstacles
        for idx in range(n_obs):
            p1 = np.eye(4)
            p1[:3, 3] = U(obst_pos_bounds[0], obst_pos_bounds[1])
            cuboid1 = eu.Prim(
                obj_type="box",
                obj_id=idx + 1,
                obj_shape=U(obst_shape_bounds[0], obst_shape_bounds[1]).tolist(),
                pose=p1,
                static=True,
                friction=0.8,
            )
            obstacle_dict[cuboid1.obj_id] = cuboid1

        # initialize collision manager
        collision_manager = trimesh.collision.CollisionManager()
        for obj_id, obj in obstacle_dict.items():
            collision_manager.add_object(
                f"obstacle_{obj_id}", obj.mesh, transform=obj.pose
            )

        block_dict = {}
        # add blocks that do not collide with obstacles
        for idx in range(n_blocks):
            for _ in range(100):
                block_shape = U(block_shape_bounds[0], block_shape_bounds[1])
                block = eu.Prim(
                    obj_type="box",
                    obj_id=n_obs + 1 + idx,
                    obj_shape=block_shape.tolist(),
                    pose=np.eye(4),
                    static=False,
                )

                p = np.eye(4)
                p[:3, 3] = U(block_pos_bounds[0], block_pos_bounds[1])
                # p[:3, :3] = trimesh.transformations.random_rotation_matrix()
                p[2, 3] = z_ground + block_shape[2] / 2 + 0.001

                dist = collision_manager.min_distance_single(block.mesh, transform=p)

                is_collision = dist < 0.0
                if not is_collision:
                    block.pose = p
                    block_dict[block.obj_id] = block

                    collision_manager.add_object(
                        f"block_{idx}", block.mesh, transform=block.pose
                    )
                    break

        if len(block_dict) == 0:
            continue

        all_obs = {**obstacle_dict, **block_dict}

        found = False
        

        all_obs["ground"] = eu.add_ground_prim(z_ground, obj_id=n_obs + n_blocks + 2)
        for _ in range(100):
            init_jpos = U(joint_lb, joint_ub)
            is_collision = eu.check_collision(
                robot, dict(zip(joint_names, init_jpos)), all_obs, view=False
            )
            if not is_collision:
                found = True
                break
        # found 'init_jpos'

        if not found:
            continue

        # found a target block positions
        target_block_pos = {}
        collision_manager = trimesh.collision.CollisionManager()
        for obs_id, obst in obstacle_dict.items():
            collision_manager.add_object(
                f"obstacle_{obs_id}", obst.mesh, transform=obst.pose
            )

        block_goals_found = True
        for block_id, block in block_dict.items():
            found = False
            for k in range(100):
                p = np.eye(4)
                p[:3, 3] = U(block_pos_bounds[0], block_pos_bounds[1])
                p[2, 3] = z_ground + block_shape[2] / 2 + 0.001

                dist = collision_manager.min_distance_single(block.mesh, transform=p)
                is_collision = dist < 0.0
                if not is_collision:
                    target_block_pos[block_id] = p
                    collision_manager.add_object(
                        f"block_{block_id}", block.mesh, transform=p
                    )
                    found = True
                    break
            if not found:
                block_goals_found = False
                break

        if not block_goals_found:
            continue

        env_found = True
        break

    if not env_found:
        return None

    init_robo_state_dict = {
        robot.robot_id: eu.RobotState(joint_pos=dict(zip(joint_names, init_jpos)))
    }
    goal_obj_state_dict = {k: eu.PrimState(pose=v) for k, v in target_block_pos.items()}
    init_obj_state_dict = {
        obj_id: eu.PrimState(pose=obj.pose) for obj_id, obj in block_dict.items()
    }



    def obj_sampler(_block_pos_bounds):
        _p = np.eye(4)
        _p[:3, 3] = U(_block_pos_bounds[0], _block_pos_bounds[1])
        return _p

    def robo_jpos_sampler(joint_lb, joint_ub):
        return U(joint_lb, joint_ub)

    def get_cfg_fn():
        return eu.get_randomized_push_cfg(
            obstacle_dict,
            movable_obj_dict=block_dict,
            robot_dict=robo_dict,
            obj_sampler=obj_sampler,
            robot_sampler=robo_jpos_sampler,
            z_ground=z_ground,
            obj_sampler_args=[block_pos_bounds],
            robo_sampler_args=[joint_lb, joint_ub],
        )

    cfg_name = eu.save_randomized_configs(
        robo_dict=robo_dict,
        get_cfg_fn=get_cfg_fn,
        task_type="push",
        robo_type="prims",
        robo_name=robo_fname,
        variation=version_name,
        seed=seed,
    )

    # remove ground
    all_obs.pop("ground")

    return eu.ProblemSpec(
        robot_dict=robo_dict,
        obj_dict=all_obs,
        init_robo_state_dict=init_robo_state_dict,
        goal_robo_state_dict={r.robot_id: eu.RobotState() for r in robo_dict.values()},
        init_obj_state_dict=init_obj_state_dict,
        goal_obj_state_dict=goal_obj_state_dict,
        ground=z_ground,
        additional_configs=cfg_name,
    )


def get_v1_env(robo_type, robo_fname, seed=0):
    return get_base_env(robo_type, robo_fname, 0, 1, seed, version_name='v1')


def get_v2_env(robo_type, robo_fname, seed=0):
    n_blocks = np.random.randint(1, 5)
    return get_base_env(robo_type, robo_fname, 0, n_blocks=n_blocks, seed=seed, version_name='v2')


def get_v3_env(robo_type, robo_fname, seed=0):
    n_obs = np.random.randint(1, 10)
    n_blocks = np.random.randint(1, 5)
    return get_base_env(robo_type, robo_fname, n_obs, n_blocks, seed=seed, version_name='v3')


def get_v1_pose_env(robo_type, robo_fname, seed=0):
    return eu.get_pose_env(robo_type, robo_fname, seed, env_fn=get_v1_env)

def get_v2_pose_env(robo_type, robo_fname, seed=0):
    return eu.get_pose_env(robo_type, robo_fname, seed, env_fn=get_v2_env)

def get_v3_pose_env(robo_type, robo_fname, seed=0):
    return eu.get_pose_env(robo_type, robo_fname, seed, env_fn=get_v3_env)



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
    prim_joint_envs[f'{prims_fname}_v3'] = (get_v3_env, 'prims', prims_fname)

    prim_pose_envs[f'{prims_fname}_v1'] = (get_v1_pose_env, 'prims', prims_fname)
    prim_pose_envs[f'{prims_fname}_v2'] = (get_v2_pose_env, 'prims', prims_fname)
    prim_pose_envs[f'{prims_fname}_v3'] = (get_v3_pose_env, 'prims', prims_fname)
    


envs = {
    "joint": prim_joint_envs,
    "pose": prim_pose_envs,
}
